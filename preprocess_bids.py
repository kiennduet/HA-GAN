# Preprocess BIDS-style T1w .nii.gz (already dof6/dof12-registered to 160x192x224)
# into flat .npy volumes resized to a cube for HA-GAN training.
#
# Usage:
#   python preprocess_bids.py --input-glob "/path/dof6_080726/sub-*/ses-01/anat/*_T1w.nii.gz" \
#       --output-dir /path/to/npy_out/ --img-size 128

import argparse
import glob
import os

import nibabel as nib
import numpy as np
from skimage.transform import resize

parser = argparse.ArgumentParser()
parser.add_argument('--input-glob', required=True, help='glob pattern matching input .nii.gz files')
parser.add_argument('--output-dir', required=True)
parser.add_argument('--img-size', type=int, default=128, help='output cube size (128 or 256)')
parser.add_argument('--clip-percentile', type=float, default=99.5,
                     help='upper intensity percentile (computed on nonzero voxels) used for clipping before scaling to [-1,1]')
parser.add_argument('--limit', type=int, default=0, help='only process the first N files (0 = all), for smoke tests')
parser.add_argument('--already-normalized', action='store_true',
                     help='input is already intensity-normalized to [0,1]: skip percentile clipping, just map to [-1,1]')
args = parser.parse_args()


def preprocess_volume(img):
    img = img.astype(np.float32)
    if args.already_normalized:
        img = img * 2 - 1  # [0,1] -> [-1,1]
    else:
        nonzero = img[img > 0]
        hi = np.percentile(nonzero, args.clip_percentile) if nonzero.size else img.max()
        img = np.clip(img, 0, hi)
        img = img / hi  # [0,1], background stays 0
        img = img * 2 - 1  # [-1,1], background -> -1

    if img.shape != (args.img_size,) * 3:
        img = resize(img, (args.img_size,) * 3, mode='constant', cval=-1, anti_aliasing=True)
    return img.astype(np.float32)


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(args.input_glob, recursive=True))
    if args.limit:
        files = files[:args.limit]
    print(f'Found {len(files)} files')

    for i, f in enumerate(files):
        sid = os.path.basename(f).split('.')[0]
        out_path = os.path.join(args.output_dir, sid + '.npy')
        if os.path.exists(out_path):
            continue
        img = nib.load(f).get_fdata()
        img = preprocess_volume(img)
        np.save(out_path, img)
        print(f'[{i+1}/{len(files)}] {sid} -> {img.shape}')


if __name__ == '__main__':
    main()
