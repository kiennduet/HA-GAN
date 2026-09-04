# Generate samples from a trained HA-GAN Generator checkpoint and visualize them:
#   - .nii.gz volume (viewable in any NIfTI viewer, e.g. FSLeyes/ITK-SNAP/MRIcroGL)
#   - orthogonal 2D slice montage (axial/coronal/sagittal)
#   - true 3D surface render (marching cubes isosurface)
#
# Usage:
#   python generate_samples.py --ckpt checkpoint/EXP/G_iter80000.pth --img-size 128 \
#       --seeds 0,7,42 --out-dir samples/

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import nibabel as nib
import numpy as np
import torch
from skimage import filters, measure

from utils import trim_state_dict_name

parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, help='path to G_iter*.pth')
# THAY ĐỔI: nhận thêm volume bất đối xứng, vd "160x192x224" (xem train.py)
parser.add_argument('--img-size', type=str, default='128',
                    help='"128", "256" or "160x192x224"; must match the trained checkpoint')
parser.add_argument('--latent-dim', type=int, default=1024)
parser.add_argument('--out-dir', required=True)
parser.add_argument('--seeds', type=str, required=True,
                    help='danh sách seed, vd "0,7,42" hoặc dải "100-104"; '
                         'số mẫu = số seed')
args = parser.parse_args()


def parse_seeds():
    """Trả về danh sách seed sẽ dùng."""
    out = []
    for part in args.seeds.split(','):
        part = part.strip()
        if '-' in part.lstrip('-'):                 # dải "100-104"
            lo, hi = part.split('-')
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out

# THAY ĐỔI: chọn model theo shape 3 chiều thay vì 1 số int
_dims = (tuple(int(v) for v in args.img_size.lower().split('x'))
         if 'x' in args.img_size.lower() else (int(args.img_size),) * 3)
if _dims == (256, 256, 256):
    from models.Model_HA_GAN_256 import Generator
elif _dims == (160, 192, 224):
    from models.Model_HA_GAN_160_192_224 import Generator
else:
    from models.Model_HA_GAN_128 import Generator

def main():
    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G = Generator(mode='eval', latent_dim=args.latent_dim).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(trim_state_dict_name(ckpt['model']))
    G.eval()

    seeds = parse_seeds()
    print(f'Sinh {len(seeds)} mẫu với seed: {seeds}')

    with torch.no_grad():
        for i, s in enumerate(seeds):
            g = torch.Generator(device=device).manual_seed(s)
            z = torch.randn(1, args.latent_dim, device=device, generator=g)
            vol = G(z).cpu().numpy()[0, 0]  # [-1, 1]
            vol01 = ((vol + 1) / 2).astype(np.float32)

            nib.save(nib.Nifti1Image(vol01, affine=np.eye(4)),
                      os.path.join(args.out_dir, f'sample_seed{s:05d}.nii.gz'))
            print(f'[{i+1}/{len(seeds)}] saved sample_seed{s:05d}.nii.gz')


if __name__ == '__main__':
    main()
