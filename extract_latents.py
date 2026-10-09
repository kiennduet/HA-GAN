#!/usr/bin/env python
# ============================================================================
# FILE MỚI - trích xuất latent Z từ ảnh thật bằng encoder phân cấp của HA-GAN
# ============================================================================
# Bám theo mục III-C "Incorporating the Encoder" của bài báo (Sun et al., 2022):
#
#   Ẑ = E^G( concat({ E^H( S^H(X^H; T_v) ) }_{v=1..V} ) )
#
# trong đó X^H được phân hoạch thành V lát cắt KHÔNG CHỒNG LẤN với chỉ số bắt
# đầu T_v. Lưu ý khác biệt so với lúc train: khi train, r được lấy NGẪU NHIÊN
# và các lát cắt có thể chồng lấn (bài báo nói điều này phủ tốt hơn các chỗ
# giáp ranh); còn khi inference thì phân hoạch đều, không chồng lấn.
#
# Với cấu hình 160x192x224: V = 8 lát cắt dày 20, mỗi lát qua E^H ra
# (64, 5, 48, 56), ghép theo trục 0 thành Â = (64, 40, 48, 56), rồi E^G -> Z 1024 chiều.
#
# Đây là bản amortized: MỘT forward pass cho mỗi ảnh, khác hẳn projector.py bên
# StyleGAN (1200 bước tối ưu gradient / ảnh).
#
# Downstream: bài báo (Bảng VI) dùng Ridge regression với lambda = 1e-4 trên Z
# để dự đoán chỉ số lâm sàng, báo cáo R² trên tập held-out.
#
# Ví dụ:
#   python extract_latents.py \
#     --ckpt-dir checkpoint/brainage_dof6 --iter 80000 \
#     --data-dir .../hagan_npy_160x192x224/ \
#     --out output/latents_brainage_80000.npz
# ============================================================================

import argparse
import glob
import os

import numpy as np
import torch

from utils import trim_state_dict_name

parser = argparse.ArgumentParser(description='Trích xuất latent Z từ ảnh thật (HA-GAN)')
parser.add_argument('--ckpt-dir', required=True, help='thư mục checkpoint/<exp-name>')
parser.add_argument('--iter', type=int, required=True, help='mốc iteration của checkpoint')
parser.add_argument('--data-dir', required=True, help='thư mục .npy (giá trị trong [-1,1])')
parser.add_argument('--out', required=True, help='file .npz đầu ra')
parser.add_argument('--img-size', default='160x192x224')
parser.add_argument('--latent-dim', type=int, default=1024)
parser.add_argument('--subjects', default=None,
                    help='CSV danh sách subject cần trích (mặc định: mọi .npy trong data-dir)')
parser.add_argument('--id-col', default='subject_id')
args = parser.parse_args()

_dims = (tuple(int(v) for v in args.img_size.lower().split('x'))
         if 'x' in args.img_size.lower() else (int(args.img_size),) * 3)
if _dims == (256, 256, 256):
    from models.Model_HA_GAN_256 import Encoder, Sub_Encoder
elif _dims == (160, 192, 224):
    from models.Model_HA_GAN_160_192_224 import Encoder, Sub_Encoder
else:
    from models.Model_HA_GAN_128 import Encoder, Sub_Encoder

V = 8                        # số lát cắt trong phân hoạch, cố định bởi kiến trúc
CROP = _dims[0] // V         # độ dày mỗi lát ở full resolution


def load(name):
    p = os.path.join(args.ckpt_dir, f'{name}_iter{args.iter}.pth')
    return trim_state_dict_name(torch.load(p, map_location='cuda')['model'])


@torch.no_grad()
def encode(x, E, E_G):
    """x: tensor (1,1,D0,D1,D2) trong [-1,1] -> Z (latent_dim,)"""
    # Â = concat({E^H(S^H(X^H; T_v))}) theo trục lát cắt, T_v không chồng lấn
    A = torch.cat([E(x[:, :, t:t + CROP, :, :]) for t in range(0, _dims[0], CROP)], dim=2)
    return E_G(A).flatten(1)[0].cpu().numpy()   # Ẑ = E^G(Â)


def main():
    E = Encoder().cuda();                             E.load_state_dict(load('E'));      E.eval()
    E_G = Sub_Encoder(latent_dim=args.latent_dim).cuda(); E_G.load_state_dict(load('Sub_E')); E_G.eval()

    if args.subjects:
        import csv
        ids = [r[args.id_col].strip() for r in csv.DictReader(open(args.subjects))]
        files = [os.path.join(args.data_dir, s + '.npy') for s in ids]
        files = [f for f in files if os.path.exists(f)]
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, '*.npy')))
    print(f'{len(files)} ảnh | phân hoạch {V} lát dày {CROP} | latent {args.latent_dim} chiều')

    sids, Z = [], []
    for i, f in enumerate(files, 1):
        vol = np.load(f)
        if vol.shape != _dims:
            print(f'  bỏ qua {os.path.basename(f)}: shape {vol.shape}')
            continue
        x = torch.from_numpy(vol).float()[None, None].cuda()
        Z.append(encode(x, E, E_G))
        sids.append(os.path.basename(f)[:-4])
        if i % 50 == 0 or i == len(files):
            print(f'\r  {i}/{len(files)}', end='', flush=True)
    print()

    Z = np.stack(Z)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, Z=Z, subject_id=np.array(sids))
    print(f'Đã lưu {Z.shape} -> {args.out}')

    # Kiểm tra nhanh: latent có phân biệt được các subject không. Nếu tương quan
    # trung bình ~1 thì model đã mode collapse, latent không dùng được cho regression.
    c = np.corrcoef(Z[:min(50, len(Z))])
    off = c[~np.eye(len(c), dtype=bool)].mean()
    print(f'Tương quan TB giữa các latent = {off:.4f}'
          + ('   [!] gần 1 -> latent gần như đồng nhất, kiểm tra mode collapse' if off > 0.95 else ''))


if __name__ == '__main__':
    main()
