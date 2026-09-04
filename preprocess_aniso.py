#!/usr/bin/env python
# ============================================================================
# FILE MỚI - chuyển NIfTI ĐÃ TIỀN XỬ LÝ SẴN -> .npy cho HA-GAN anisotropic
# ============================================================================
# KHÁC preprocess.py gốc ở chỗ: KHÔNG resize, KHÔNG trim slice, KHÔNG đoán
# ngưỡng cường độ. Data của ta đã qua MRIPre/run_prepro_freesurfer.py:
#   - conform 1mm isotropic, skull-strip (SynthStrip), align MNI (FLIRT dof6)
#   - crop sẵn về đúng 160x192x224
#   - chuẩn hoá cường độ về [0,1] theo percentile 99
# Nên script này chỉ còn 2 việc:
#   1. lọc subject theo file CSV
#   2. đổi thang [0,1] -> [-1,1] (khớp tanh ở đầu ra Generator) và lưu .npy
#
# CẢNH BÁO: đừng dùng preprocess.py gốc cho data này - nó sẽ resize về khối
# lập phương IMG_SIZE^3 và phá mất shape 160x192x224.
#
# Cấu trúc input mong đợi:  <data-dir>/<subject_id>/anat/<subject_id>_T1w.nii.gz
# Output:                   <out-dir>/<subject_id>.npy   (float32, [-1,1])
#
# Ví dụ:
#   python preprocess_aniso.py \
#     --data-dir ../../data/brain_age/derivatives/dof6_SS_1mm \
#     --csv      ../../data/brain_age/participants_hc_qc_merged_vol1100.csv \
#     --out-dir  ../../data/brain_age/hagan_npy_160x192x224/ \
#     --workers 8
# ============================================================================

import argparse
import csv
import multiprocessing as mp
import os

import nibabel as nib
import numpy as np

EXPECTED_SHAPE = (160, 192, 224)

parser = argparse.ArgumentParser(
    description='Chuyển NIfTI đã tiền xử lý -> .npy [-1,1] cho HA-GAN')
parser.add_argument('--data-dir', required=True,
                    help='thư mục derivatives chứa <subject_id>/anat/')
parser.add_argument('--csv', required=True,
                    help='CSV danh sách subject cần chạy')
parser.add_argument('--out-dir', required=True,
                    help='thư mục lưu .npy (sẽ truyền vào train.py --data-dir)')
parser.add_argument('--id-col', default='subject_id',
                    help='tên cột chứa subject id trong CSV (mặc định: subject_id)')
parser.add_argument('--suffix', default='_T1w.nii.gz',
                    help='hậu tố file ảnh (mặc định: _T1w.nii.gz)')
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--overwrite', action='store_true',
                    help='ghi đè file .npy đã có (mặc định: bỏ qua để chạy tiếp được)')


def nii_path(data_dir, sid, suffix):
    return os.path.join(data_dir, sid, 'anat', sid + suffix)


def convert_one(args_tuple):
    """Trả về (sid, trạng thái, thông tin thêm)."""
    sid, a = args_tuple
    out_p = os.path.join(a.out_dir, sid + '.npy')
    if os.path.exists(out_p) and not a.overwrite:
        return sid, 'skip', 'đã có'

    in_p = nii_path(a.data_dir, sid, a.suffix)
    if not os.path.exists(in_p):
        return sid, 'missing', in_p

    try:
        img = nib.load(in_p)
        if img.shape != EXPECTED_SHAPE:
            # Không tự resize: shape sai nghĩa là subject này lệch pipeline,
            # cần xem lại tiền xử lý chứ không nên âm thầm nội suy.
            return sid, 'bad_shape', str(img.shape)

        vol = img.get_fdata(dtype=np.float32)

        if not np.isfinite(vol).all():
            return sid, 'nan', 'có NaN/Inf'

        lo, hi = float(vol.min()), float(vol.max())
        if lo < -1e-4 or hi > 1.0 + 1e-4:
            # Data đầu vào phải nằm trong [0,1] (run_prepro_freesurfer.py đã clip).
            return sid, 'bad_range', f'[{lo:.3f},{hi:.3f}]'

        # [0,1] -> [-1,1] để khớp tanh ở đầu ra Generator.
        # Nền (voxel 0 sau skull-strip) thành -1, đúng quy ước của HA-GAN.
        vol = np.clip(vol, 0.0, 1.0) * 2.0 - 1.0

        # Ghi ra file tạm rồi đổi tên: tránh để lại .npy hỏng nếu bị ngắt giữa chừng
        tmp_p = out_p + '.tmp.npy'
        np.save(tmp_p, vol.astype(np.float32))
        os.replace(tmp_p, out_p)
        return sid, 'ok', f'{hi:.3f}'

    except Exception as e:
        return sid, 'error', str(e)[:120]


def main():
    a = parser.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    with open(a.csv) as f:
        rows = list(csv.DictReader(f))
    if a.id_col not in rows[0]:
        raise SystemExit(f"CSV không có cột '{a.id_col}'. Các cột có sẵn: {list(rows[0])}")

    sids = []
    seen = set()
    for r in rows:
        sid = r[a.id_col].strip()
        if sid and sid not in seen:      # bỏ trùng lặp, giữ nguyên thứ tự
            seen.add(sid)
            sids.append(sid)

    print(f'CSV       : {a.csv}')
    print(f'Subjects  : {len(sids)}')
    print(f'Input     : {a.data_dir}/<sid>/anat/<sid>{a.suffix}')
    print(f'Output    : {a.out_dir}')
    print(f'Shape yêu cầu: {EXPECTED_SHAPE}, thang [0,1] -> [-1,1]\n')

    with mp.Pool(a.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(convert_one,
                                                    [(s, a) for s in sids],
                                                    chunksize=4), 1):
            results.append(res)
            if i % 100 == 0 or i == len(sids):
                print(f'  {i}/{len(sids)}', flush=True)

    # Tổng kết: liệt kê rõ subject nào hỏng để còn quay lại xem tiền xử lý
    counts = {}
    for _, status, _ in results:
        counts[status] = counts.get(status, 0) + 1

    print('\n=== TỔNG KẾT ===')
    for status in sorted(counts):
        print(f'  {status:<10} {counts[status]}')

    bad = [(s, st, info) for s, st, info in results if st not in ('ok', 'skip')]
    if bad:
        print(f'\n{len(bad)} subject KHÔNG chuyển được:')
        for s, st, info in bad[:30]:
            print(f'  {s:<24} {st:<10} {info}')
        if len(bad) > 30:
            print(f'  ... và {len(bad)-30} subject nữa')

    n_npy = len([f for f in os.listdir(a.out_dir) if f.endswith('.npy')])
    print(f'\nTổng số .npy trong {a.out_dir}: {n_npy}')
    print(f'Chạy train bằng:  python train.py --img-size 160x192x224 '
          f'--data-dir {os.path.join(a.out_dir, "")} --exp-name <ten_thi_nghiem>')


if __name__ == '__main__':
    main()
