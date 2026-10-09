#!/usr/bin/env python
# ============================================================================
# FILE MỚI - projection/reconstruction: ảnh thật -> Ẑ -> ảnh tái tạo
# ============================================================================
# Đây là "projection" kiểu amortized của HA-GAN (mục III-C bài báo):
#     Ẑ = E^G(concat({E^H(S^H(X^H; T_v))}))     rồi     X̂^H = G^H(G^A(Ẑ))
# MỘT forward pass, khác projector.py bên StyleGAN (1200 bước tối ưu gradient).
#
# Xuất ra, cho mỗi subject:
#   <sid>_real.nii.gz  <sid>_recon.nii.gz   (mở bằng FSLeyes/ITK-SNAP để so)
#   <sid>_compare.png  - 3 lát cắt trực giao, hàng trên real / hàng dưới recon
# và in bảng chỉ số fidelity (tương quan, MAE) chỉ tính trên vùng não.
#
# Ví dụ:
#   python reconstruct.py --ckpt-dir checkpoint/brainage_dof6 --iter 80000 \
#     --data-dir .../hagan_npy_160x192x224/ --subjects sub-BrainAge000019,sub-BrainAge000020 \
#     --out-dir output/recon_iter80000
# ============================================================================

import argparse, glob, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from utils import trim_state_dict_name

p = argparse.ArgumentParser(description='Tái tạo ảnh thật qua encoder+generator của HA-GAN')
p.add_argument('--ckpt-dir', required=True)
p.add_argument('--iter', type=int, required=True)
p.add_argument('--data-dir', required=True)
p.add_argument('--out-dir', required=True)
p.add_argument('--subjects', default=None, help='danh sách sid ngăn bởi dấu phẩy; mặc định lấy 5 ảnh đầu')
p.add_argument('--img-size', default='160x192x224')
p.add_argument('--latent-dim', type=int, default=1024)
args = p.parse_args()

_d = (tuple(int(v) for v in args.img_size.lower().split('x'))
      if 'x' in args.img_size.lower() else (int(args.img_size),) * 3)
if _d == (256, 256, 256):
    from models.Model_HA_GAN_256 import Generator, Encoder, Sub_Encoder
elif _d == (160, 192, 224):
    from models.Model_HA_GAN_160_192_224 import Generator, Encoder, Sub_Encoder
else:
    from models.Model_HA_GAN_128 import Generator, Encoder, Sub_Encoder

V = 8
CROP = _d[0] // V


def load(n):
    return trim_state_dict_name(torch.load(
        os.path.join(args.ckpt_dir, f'{n}_iter{args.iter}.pth'), map_location='cuda')['model'])


def save_png(real, rec, path, sid):
    """3 lát cắt giữa: hàng trên = thật, hàng dưới = tái tạo."""
    cuts = [(real[real.shape[0] // 2], rec[rec.shape[0] // 2]),
            (real[:, real.shape[1] // 2], rec[:, rec.shape[1] // 2]),
            (real[:, :, real.shape[2] // 2], rec[:, :, rec.shape[2] // 2])]
    fig, ax = plt.subplots(2, 3, figsize=(11, 8))
    for j, (a, b) in enumerate(cuts):
        for i, img in enumerate((a, b)):
            ax[i, j].imshow(img.T, cmap='gray', origin='lower', vmin=0, vmax=1)
            ax[i, j].axis('off')
    ax[0, 0].set_title('THẬT', loc='left'); ax[1, 0].set_title('TÁI TẠO', loc='left')
    fig.suptitle(sid)
    plt.tight_layout(); plt.savefig(path, dpi=130, bbox_inches='tight'); plt.close(fig)


@torch.no_grad()
def main():
    os.makedirs(args.out_dir, exist_ok=True)
    E = Encoder().cuda();                                  E.load_state_dict(load('E')); E.eval()
    EG = Sub_Encoder(latent_dim=args.latent_dim).cuda();    EG.load_state_dict(load('Sub_E')); EG.eval()
    G = Generator(mode='eval', latent_dim=args.latent_dim).cuda(); G.load_state_dict(load('G')); G.eval()

    if args.subjects:
        files = [os.path.join(args.data_dir, s.strip() + '.npy') for s in args.subjects.split(',')]
    else:
        files = sorted(glob.glob(os.path.join(args.data_dir, '*.npy')))[:5]

    print(f"{'subject':<24} {'corr(não)':>10} {'MAE(não)':>10}")
    reals, recs, sids = [], [], []
    for f in files:
        sid = os.path.basename(f)[:-4]
        vol = np.load(f)                                     # [-1,1]
        x = torch.from_numpy(vol).float()[None, None].cuda()
        A = torch.cat([E(x[:, :, t:t + CROP, :, :]) for t in range(0, _d[0], CROP)], dim=2)
        rec = G(EG(A)).cpu().numpy()[0, 0]                   # [-1,1]

        m = vol > -0.99                                      # chỉ tính trên mô não
        corr = np.corrcoef(vol[m], rec[m])[0, 1]
        mae = np.abs(vol[m] - rec[m]).mean()
        print(f'{sid:<24} {corr:>10.4f} {mae:>10.4f}')

        r01, c01 = (vol + 1) / 2, (rec + 1) / 2
        nib.save(nib.Nifti1Image(r01.astype(np.float32), np.eye(4)), f'{args.out_dir}/{sid}_real.nii.gz')
        nib.save(nib.Nifti1Image(c01.astype(np.float32), np.eye(4)), f'{args.out_dir}/{sid}_recon.nii.gz')
        save_png(r01, c01, f'{args.out_dir}/{sid}_compare.png', sid)
        reals.append(vol); recs.append(rec); sids.append(sid)

    # Phép thử danh tính: bản tái tạo có khớp với ĐÚNG người của nó không?
    if len(files) > 1:
        m = np.any(np.stack(reals) > -0.99, 0)
        C = np.array([[np.corrcoef(recs[i][m], reals[j][m])[0, 1] for j in range(len(sids))]
                      for i in range(len(sids))])
        hit = sum(int(np.argmax(C[i]) == i) for i in range(len(sids)))
        print(f'\nPhép thử danh tính: {hit}/{len(sids)} bản tái tạo khớp nhất với đúng người của nó')
        print(f'  đúng người (đường chéo) TB = {np.mean(np.diag(C)):.4f}')
        print(f'  sai người (ngoài chéo)  TB = {C[~np.eye(len(C), dtype=bool)].mean():.4f}')
    print(f'\nĐã lưu vào {args.out_dir}')


if __name__ == '__main__':
    main()
