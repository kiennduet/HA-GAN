#!/usr/bin/env python
# ============================================================================
# FILE MỚI - quét TẤT CẢ checkpoint G_iter*.pth và tính FID theo từng mốc
# ============================================================================
# Vì sao không dùng evaluation/fid_score.py: file đó lỗi cú pháp ở dòng 60
# (`Sub_Encoder(args.latent_dim=args.latent_dim)`), hardcode Model_HA_GAN_256,
# và hardcode đường dẫn resnet ("../gnn_shared/..."). Script này viết lại phần
# cần dùng, giữ nguyên công thức Fréchet distance chuẩn.
#
# Đặc điểm dùng 3D feature (MedicalNet ResNet50) -> FID tính trên cả khối,
# KHÁC với StyleGAN của bạn đang đo FID 2D theo lát cắt (fid250_both_sag/cor/ax).
# Hai con số này KHÔNG so sánh trực tiếp với nhau được.
#
# Feature của real chỉ trích xuất MỘT LẦN rồi cache lại, các checkpoint sau
# dùng chung -> quét nhiều mốc rất nhanh.
#
# Quét đều toàn bộ dải checkpoint, thưa dần bằng --stride (vd --stride 3 = lấy
# 1/3 số mốc, trải đều từ đầu đến cuối). Mốc cuối cùng luôn được giữ lại.
#
# Ví dụ (100 checkpoint -> đánh giá 34 mốc):
#   python evaluation/eval_checkpoints.py \
#     --ckpt-dir    checkpoint/brainage_dof6 \
#     --data-dir    /home/dknguyen/Documents/2_Works/1_FLAIR_MS/data/brain_age/hagan_npy_160x192x224/ \
#     --resnet-ckpt /home/dknguyen/Documents/2_Works/1_FLAIR_MS/StyleGAN3D/3DStyleGAN/metrics/resnet_50_medicalnet.pth \
#     --img-size 160x192x224 --fold -1 \
#     --stride 3 --num-samples 512 --batch-size 4 \
#     --out-dir evaluation/results_fid
# ============================================================================

import argparse
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import linalg

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils import trim_state_dict_name, Flatten
from volume_dataset import Volume_Dataset
from resnet3D import resnet50

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Tính FID cho từng checkpoint HA-GAN')
parser.add_argument('--ckpt-dir', required=True, help='thư mục checkpoint/<exp-name>')
parser.add_argument('--data-dir', required=True, help='thư mục .npy real (như train.py)')
parser.add_argument('--resnet-ckpt', required=True, help='MedicalNet resnet_50.pth')
parser.add_argument('--img-size', default='160x192x224', help='phải khớp checkpoint')
parser.add_argument('--latent-dim', type=int, default=1024)
parser.add_argument('--fold', type=int, default=-1,
                    help='fold lấy real; -1 = toàn bộ (khớp với lúc train)')
parser.add_argument('--num-samples', type=int, default=2048,
                    help='số mẫu mỗi bên. Nên >= 2048: covariance 2048 chiều cần '
                         'nhiều mẫu, ít mẫu thì FID bị chệch (cao giả tạo)')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--stride', type=int, default=1,
                    help='lấy mỗi checkpoint thứ k, trải đều toàn dải. '
                         'vd 3 = 1/3 số mốc. Mặc định 1 = quét hết')
parser.add_argument('--iters', default=None,
                    help='chỉ chạy đúng các mốc này, vd "61500,63500,72500". '
                         'Nếu có thì bỏ qua --stride')
parser.add_argument('--noise-seed', type=int, default=0,
                    help='MỌI checkpoint dùng chung đúng bộ noise này -> so sánh theo cặp, '
                         'nhiễu lấy mẫu triệt tiêu bớt, xếp hạng đáng tin hơn hẳn khi --num-samples nhỏ')
parser.add_argument('--out-dir', default='./evaluation/results_fid')
args = parser.parse_args()

_dims = (tuple(int(v) for v in args.img_size.lower().split('x'))
         if 'x' in args.img_size.lower() else (int(args.img_size),) * 3)
if _dims == (256, 256, 256):
    from models.Model_HA_GAN_256 import Generator
elif _dims == (160, 192, 224):
    from models.Model_HA_GAN_160_192_224 import Generator
else:
    from models.Model_HA_GAN_128 import Generator


def get_feature_extractor():
    model = resnet50(shortcut_type='B')
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten())
    ckpt = torch.load(args.resnet_ckpt, map_location='cuda')
    ckpt = trim_state_dict_name(ckpt['state_dict'])
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


def real_features(fx):
    """Trích xuất feature của ảnh thật (cache lại để các checkpoint dùng chung)."""
    cache = os.path.join(args.out_dir, f'pred_arr_Real_fold{args.fold}_n{args.num_samples}.npz')
    if os.path.exists(cache):
        print(f'[real] dùng cache {cache}')
        return np.load(cache)['arr_0']

    ds = Volume_Dataset(data_dir=args.data_dir, fold=args.fold)
    n = min(args.num_samples, len(ds))
    if n < args.num_samples:
        print(f'[real] CHỈ có {n} volume thật (< --num-samples {args.num_samples}) -> FID sẽ chệch')
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, drop_last=False,
                                         shuffle=True, num_workers=args.workers)
    feats, filled = [], 0
    with torch.no_grad():
        for i, (batch, _) in enumerate(loader):
            if filled >= n:
                break
            print(f'\r[real] batch {i+1}', end='', flush=True)
            pred = fx(batch.float().cuda()).cpu().numpy()
            take = min(pred.shape[0], n - filled)
            feats.append(pred[:take])
            filled += take
    print(' xong')
    out = np.concatenate(feats)
    np.savez(cache, out)
    return out


def fake_features(fx, ckpt_path):
    G = Generator(mode='eval', latent_dim=args.latent_dim).cuda()
    G.load_state_dict(trim_state_dict_name(torch.load(ckpt_path, map_location='cuda')['model']))
    G = nn.DataParallel(G).cuda()
    G.eval()

    # Generator RNG khởi tạo lại cho MỖI checkpoint bằng cùng một seed -> tất cả
    # checkpoint nhìn thấy ĐÚNG cùng bộ z. So sánh trở thành so sánh theo cặp:
    # nhiễu do lấy mẫu triệt tiêu phần lớn, nên xếp hạng vẫn tin được dù
    # --num-samples nhỏ (dùng cho vòng quét thô).
    g = torch.Generator(device='cuda').manual_seed(args.noise_seed)

    n_batches = args.num_samples // args.batch_size
    feats = []
    with torch.no_grad():
        for i in range(n_batches):
            print(f'\r[fake] batch {i+1}/{n_batches}', end='', flush=True)
            noise = torch.randn((args.batch_size, args.latent_dim), device='cuda', generator=g)
            feats.append(fx(G(noise)).cpu().numpy())   # G đầu ra [-1,1], như real .npy
    print(' xong')
    del G
    torch.cuda.empty_cache()
    return np.concatenate(feats)


def frechet_distance(a1, a2, eps=1e-6):
    mu1, s1 = np.mean(a1, axis=0), np.cov(a1, rowvar=False)
    mu2, s2 = np.mean(a2, axis=0), np.cov(a2, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        print('  (covariance suy biến, cộng eps vào đường chéo)')
        offset = np.eye(s1.shape[0]) * eps
        covmean = linalg.sqrtm((s1 + offset).dot(s2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def main():
    os.makedirs(args.out_dir, exist_ok=True)

    ckpts = []
    for p in glob.glob(os.path.join(args.ckpt_dir, 'G_iter*.pth')):
        m = re.search(r'G_iter(\d+)\.pth', os.path.basename(p))
        if m:
            ckpts.append((int(m.group(1)), p))
    ckpts.sort()
    if not ckpts:
        raise SystemExit(f'Không tìm thấy G_iter*.pth trong {args.ckpt_dir}')
    if args.iters:
        want = {int(x) for x in args.iters.split(',')}
        ckpts = [c for c in ckpts if c[0] in want]
        missing = want - {c[0] for c in ckpts}
        if missing:
            raise SystemExit(f'Không có checkpoint cho mốc: {sorted(missing)}')
    elif args.stride > 1:
        # Giữ luôn mốc cuối cùng: đó thường là mốc bạn muốn đối chiếu nhất
        ckpts = sorted(set(ckpts[::args.stride] + [ckpts[-1]]))
    print(f'Sẽ đánh giá {len(ckpts)} checkpoint: {[c[0] for c in ckpts]}\n')

    fx = get_feature_extractor()
    real = real_features(fx)
    print(f'[real] feature shape {real.shape}\n')

    rows = []
    for it, path in ckpts:
        print(f'--- iter {it} ---')
        fake = fake_features(fx, path)
        fid = frechet_distance(real, fake)
        print(f'  FID = {fid:.3f}\n')
        rows.append((it, fid))

        # ghi CSV sau mỗi mốc để ngắt giữa chừng vẫn còn kết quả
        with open(os.path.join(args.out_dir, 'fid_by_iter.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['iter', 'fid'])
            w.writerows(rows)

    best = min(rows, key=lambda r: r[1])
    print('=' * 40)
    print(f'{"iter":>10} | {"FID":>10}')
    for it, fid in rows:
        print(f'{it:>10} | {fid:>10.3f}' + ('   <- tốt nhất' if it == best[0] else ''))

    if len(rows) > 1:
        plt.figure(figsize=(7, 4))
        plt.plot([r[0] for r in rows], [r[1] for r in rows], marker='o')
        plt.xlabel('iteration'); plt.ylabel('FID (3D, MedicalNet ResNet50)')
        plt.grid(alpha=.3); plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, 'fid_curve.png'), dpi=150)
        print('\nĐã lưu', os.path.join(args.out_dir, 'fid_curve.png'))


if __name__ == '__main__':
    main()
