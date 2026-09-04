# Compare the feature distribution of real vs. HA-GAN-generated volumes with MDS.
# Adapted from visualize_feature_MDS.ipynb to the current repo layout:
#   - models.Model_HA_GAN_{128,256} (old notebook used a stale "Model_Alpha_GAN_..." module)
#   - checkpoint/<exp-name>/G_iter<N>.pth (matches train.py's own naming)
#   - Volume_Dataset for real .npy volumes (same loader train.py uses)
#   - baseline-GAN comparison branches removed (those checkpoints don't exist in this repo);
#     any extra "pred_arr_<name>.npz" dropped into --results-dir is picked up automatically.
#
# Requires a MedicalNet pretrained ResNet50-3D checkpoint (resnet_50.pth) as the feature
# extractor: https://github.com/Tencent/MedicalNet
#
# Usage:
#   python evaluation/visualize_feature_MDS.py \
#       --data-dir /path/to/real_npy/ --ckpt checkpoint/EXP/G_iter80000.pth \
#       --img-size 128 --resnet-ckpt /path/to/resnet_50.pth

import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.manifold import MDS

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import trim_state_dict_name, Flatten
from volume_dataset import Volume_Dataset
from resnet3D import resnet50

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', required=True, help='folder of real .npy volumes (same format Volume_Dataset expects)')
parser.add_argument('--ckpt', required=True, help='path to trained G_iter*.pth')
parser.add_argument('--resnet-ckpt', required=True, help='path to MedicalNet resnet_50.pth')
parser.add_argument('--img-size', type=str, default='128',
                    help='"128", "256" or "160x192x224", must match --ckpt')
parser.add_argument('--latent-dim', type=int, default=1024)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--num-samples', type=int, default=512)
parser.add_argument('--batch-size', type=int, default=2)
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--results-dir', default='./evaluation/results', help='where features/plot are written and read from')
parser.add_argument('--skip-extract', action='store_true',
                     help='skip feature extraction, only redo the MDS plot from existing pred_arr_*.npz in --results-dir')
args = parser.parse_args()

# THAY DOI: them nhanh volume bat doi xung (xem train.py)
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
    model.conv_seg = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)), Flatten())  # (N, 512)
    ckpt = torch.load(args.resnet_ckpt, map_location='cuda')
    ckpt = trim_state_dict_name(ckpt['state_dict'])
    model.load_state_dict(ckpt)
    model = nn.DataParallel(model).cuda()
    model.eval()
    return model


def extract_fake_features(model):
    G = Generator(mode='eval', latent_dim=args.latent_dim).cuda()
    ckpt = torch.load(args.ckpt, map_location='cuda')
    G.load_state_dict(trim_state_dict_name(ckpt['model']))
    G = nn.DataParallel(G).cuda()
    G.eval()

    n_batches = args.num_samples // args.batch_size
    feats = []
    with torch.no_grad():
        for i in range(n_batches):
            print(f'\r[fake] batch {i+1}/{n_batches}', end='', flush=True)
            noise = torch.randn((args.batch_size, args.latent_dim)).cuda()
            x_fake = G(noise)  # range [-1, 1], same as real .npy volumes
            pred = model(x_fake)
            feats.append(pred.cpu().numpy())
    print(' done')
    return np.concatenate(feats)


def extract_real_features(model):
    dataset = Volume_Dataset(data_dir=args.data_dir, fold=args.fold)
    n = min(args.num_samples, len(dataset))
    if n < args.num_samples:
        print(f'[real] only {n} real volumes available, using all of them')
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, drop_last=False,
                                          shuffle=True, num_workers=args.workers)
    feats = []
    filled = 0
    with torch.no_grad():
        for i, (batch, _) in enumerate(loader):
            if filled >= n:
                break
            print(f'\r[real] batch {i+1}', end='', flush=True)
            batch = batch.float().cuda()
            pred = model(batch).cpu().numpy()
            take = min(pred.shape[0], n - filled)
            feats.append(pred[:take])
            filled += take
    print(' done')
    return np.concatenate(feats)


def fit_ellipse(x, y):
    # SỬA: lấy trung bình TRƯỚC khi trừ, đúng như fit_eclipse() trong
    # visualize_feature_MDS.ipynb gốc. Bản trước tính xmean/ymean SAU khi đã
    # căn giữa nên luôn ra ~0 -> ellipse bị vẽ ở gốc toạ độ thay vì ở trọng
    # tâm cụm điểm.
    N = 300
    xmean, ymean = x.mean(), y.mean()
    x = x - xmean
    y = y - ymean
    U, S, _ = np.linalg.svd(np.stack((x, y)))
    tt = np.linspace(0, 2 * np.pi, 1000)
    circle = np.stack((np.cos(tt), np.sin(tt)))
    transform = np.sqrt(2 / N) * U.dot(np.diag(S))
    return transform.dot(circle) + np.array([[xmean], [ymean]])


def main():
    os.makedirs(args.results_dir, exist_ok=True)

    if not args.skip_extract:
        model = get_feature_extractor()
        real_feat = extract_real_features(model)
        np.savez(os.path.join(args.results_dir, 'pred_arr_Real.npz'), real_feat)
        fake_feat = extract_fake_features(model)
        np.savez(os.path.join(args.results_dir, 'pred_arr_HA_GAN.npz'), fake_feat)

    # auto-discover every pred_arr_<name>.npz present (so extra baselines just work if added later)
    group_names, group_arrays = [], []
    for path in sorted(glob.glob(os.path.join(args.results_dir, 'pred_arr_*.npz'))):
        name = os.path.basename(path)[len('pred_arr_'):-len('.npz')]
        group_names.append(name)
        group_arrays.append(np.load(path)['arr_0'])

    if len(group_names) < 2:
        raise RuntimeError(f'Need at least 2 feature groups in {args.results_dir}, found: {group_names}')

    sizes = [a.shape[0] for a in group_arrays]
    combined = np.concatenate(group_arrays)

    mds = MDS(n_components=2, n_jobs=8, random_state=0, normalized_stress='auto')
    embedded = mds.fit_transform(combined)

    colors = plt.get_cmap('tab10').colors
    fig, ax = plt.subplots(figsize=(8, 8))
    start = 0
    for i, (name, size) in enumerate(zip(group_names, sizes)):
        pts = embedded[start:start + size]
        start += size
        c = colors[i % len(colors)]
        ax.scatter(pts[:, 0], pts[:, 1], c=[c], s=10, label=name, alpha=0.4, edgecolors='none')
        if size >= 3:
            fit = fit_ellipse(pts[:, 0].copy(), pts[:, 1].copy())
            ax.scatter(fit[0, :], fit[1, :], c=[c], s=2.5, alpha=0.4)

    leg = ax.legend(prop={'size': 12}, markerscale=3.)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    plt.tight_layout()
    out_path = os.path.join(args.results_dir, 'MDS_real_vs_generated.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print('Saved plot to', out_path)


if __name__ == '__main__':
    main()
