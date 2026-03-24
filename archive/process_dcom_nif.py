#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DICOM 그레이스케일 폴더를 순회하며 dcm2niix로 NIfTI 변환 후,
0-1 정규화 및 정사각형 패딩을 적용하여 저장하는 스크립트
병렬 처리를 통해 속도 향상 
"""
import os
import glob
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import nibabel as nib

# —————————————————————————————————————————————————————
# 설정
# —————————————————————————————————————————————————————
BASE_DIRS = [
    "/public/sylee/Data/EUMC",
    "/public/sylee/Data/KNUCH",
    "/public/sylee/Data/KUAH",
    "/public/sylee/Data/SMC",
]
OUTPUT_ROOT = "/public/sylee/MDAISS_2/output/nifti"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

total_cores = os.cpu_count() or 1
MAX_WORKERS = max(1, int(total_cores * 0.75))

# —————————————————————————————————————————————————————
# 유틸리티 함수
# —————————————————————————————————————————————————————

def get_folder_components(folder_path):
    parts = folder_path.rstrip(os.sep).split(os.sep)
    return parts[-3], parts[-2], parts[-1]


def normalize_and_pad(nifti_path):
    img = nib.load(nifti_path)
    data = img.get_fdata().astype(np.float32)
    # 0-1 정규화
    vmin, vmax = data.min(), data.max()
    data = (data - vmin) / (vmax - vmin + 1e-8)
    # 정사각형 패딩 (첫 두 차원에만 적용)
    h, w = data.shape[0], data.shape[1]
    M = max(h, w)
    pad_h, pad_w = M - h, M - w
    pads = [
        (pad_h // 2, pad_h - pad_h // 2),
        (pad_w // 2, pad_w - pad_w // 2)
    ]
    for _ in data.shape[2:]:
        pads.append((0, 0))
    data_padded = np.pad(data, pads, mode="constant", constant_values=0.0)
    nib.save(nib.Nifti1Image(data_padded, affine=img.affine), nifti_path)


def process_grayscale_folder(folder):
    """
    개별 폴더를 처리:
    1) dcm2niix으로 변환 (JSON 원본 유지)
    2) NIfTI 정규화 및 패딩
    """
    dicom_files = glob.glob(os.path.join(folder, "*.dcm"))
    if not dicom_files:
        return f"[SKIP] No DICOM files in: {folder}"

    hosp, study, series = get_folder_components(folder)
    base = f"{hosp}_{study}_{series}"

    # 1) dcm2niix 호출
    cmd = [
        "dcm2niix", "-z", "y", "-f", base,
        "-o", OUTPUT_ROOT, folder
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return f"[ERROR] dcm2niix failed for {folder}: {e.stderr.strip()}"

    nii_path = os.path.join(OUTPUT_ROOT, base + ".nii.gz")

    # 2) 정규화 및 패딩
    try:
        normalize_and_pad(nii_path)
    except Exception as e:
        return f"[ERROR] normalize/pad failed for {nii_path}: {e}"

    return f"[OK] {base} → processed"


def collect_folders():
    folders = []
    for base_dir in BASE_DIRS:
        pattern = os.path.join(base_dir, "*", "*_grayscale")
        folders.extend(glob.glob(pattern))
    return folders


def main():
    folders = collect_folders()
    total = len(folders)
    print(f"총 {total}개 폴더 처리 시작: 병렬 {MAX_WORKERS} 워커 사용")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_folder = {executor.submit(process_grayscale_folder, folder): folder for folder in folders}
        for i, future in enumerate(as_completed(future_to_folder), 1):
            folder = future_to_folder[future]
            try:
                res = future.result()
            except Exception as exc:
                res = f"[ERROR] {folder} generated exception: {exc}"
            print(f"({i}/{total}) {res}")

if __name__ == "__main__":
    main()
