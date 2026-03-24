import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import logging
from sr3d_model import SuperResolution3D, SR3DVolumeDataset, device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"체크포인트 로드 완료: {checkpoint_path}")

def get_inset_position(zoom_center, img_shape, zoom_size):
    """상자 위치에 따라 inset 위치를 대각선으로 결정"""
    h, w = img_shape
    x, y = zoom_center
    center_x, center_y = w // 2, h // 2

    if x > center_x and y > center_y:      # 오른쪽 아래
        return [0.05, 0.05, 0.4, 0.4]       # 왼쪽 위
    elif x < center_x and y > center_y:    # 왼쪽 아래
        return [0.55, 0.05, 0.4, 0.4]       # 오른쪽 위
    elif x > center_x and y < center_y:    # 오른쪽 위
        return [0.05, 0.55, 0.4, 0.4]       # 왼쪽 아래
    else:                                  # 왼쪽 위
        return [0.55, 0.55, 0.4, 0.4]       # 오른쪽 아래

def plot_with_inset(ax, base_img, zoom_img, zoom_center, zoom_size):
    ax.imshow(base_img, cmap='gray', origin='lower')
    ax.axis('off')

    x, y = zoom_center
    hs = zoom_size

    inset_pos = get_inset_position(zoom_center, base_img.shape, zoom_size)
    inset_ax = ax.inset_axes(inset_pos)

    patch = zoom_img[y-hs:y+hs, x-hs:x+hs]
    inset_ax.imshow(patch, cmap='gray', origin='lower')
    inset_ax.set_title("Zoom", fontsize=8, color='white', pad=-8) 
    inset_ax.axis('off')

    bounds = [x-hs, y-hs, 2*hs, 2*hs]

    # 연결선과 박스 그리기
    rect, connectors = ax.indicate_inset(
        bounds, inset_ax,
        edgecolor='red',
        linewidth=1.5,
        alpha=0.8
    )
    
    for connector in connectors:
        if connector.get_visible():
            connector.set_color('gray')
            connector.set_alpha(0.9)
            connector.set_linewidth(1.0)

def main():
    checkpoint_path = '/public/sylee/MDAISS_2/models/sr3d/batch_2_lr_0.0001_epochs_100/checkpoints/checkpoint_epoch_100.pth'
    lr_csv_file    = '/public/sylee/MDAISS_2/output/test_dataset.csv'
    hr_csv_file    = '/public/sylee/MDAISS_2/output/test_dataset_2.csv'
    sample_dir     = Path('/public/sylee/MDAISS_2/output/sr3d_samples')
    os.makedirs(sample_dir, exist_ok=True)

    model = SuperResolution3D(
        in_channels=1, out_channels=1,
        base_channels=8, num_blocks=4,
        upscale_factor=2
    ).to(device)

    load_checkpoint(model, checkpoint_path)
    model.eval()

    dataset = SR3DVolumeDataset(
        hr_csv_file=hr_csv_file,
        lr_csv_file=lr_csv_file,
        lr_shape=(128,128,128),
        hr_shape=(256,256,256),
        lr_spacing=(1.5,1.5,1.5),
        hr_spacing=(0.75,0.75,0.75)
    )

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    sample_indices = [0,1,2,3]
    sample_rel_centers = {
        0: (0.25, 0.25),
        1: (0.75, 0.25),
        2: (0.25, 0.75),
        3: (0.75, 0.75),
    }
    zoom_frac = 0.125

    samples = {}
    with torch.no_grad():
        for i, (lr, hr) in enumerate(loader):
            if i in sample_indices:
                lr, hr = lr.to(device), hr.to(device)
                sr = torch.clamp(model(lr), 0, 1)
                samples[i] = {
                    'lr': lr.cpu().numpy()[0,0],
                    'hr': hr.cpu().numpy()[0,0],
                    'sr': sr.cpu().numpy()[0,0],
                }
            if len(samples) == len(sample_indices):
                break

    n = len(sample_indices)
    fig, axs = plt.subplots(n, 3, figsize=(12, 4*n))

    for row, idx in enumerate(sample_indices):
        vol_dict = samples[idx]
        relx, rely = sample_rel_centers[idx]

        for col, key in enumerate(['lr','hr','sr']):
            img3d = vol_dict[key]
            h, w, d = img3d.shape
            z = d // 2
            img2d = img3d[:, :, z]

            cx = int(w * relx)
            cy = int(h * rely)
            patch_size = int(min(h, w) * zoom_frac)

            ax = axs[row, col]
            plot_with_inset(ax, img2d, img2d, (cx, cy), patch_size)
            ax.set_title(f"Sample {idx} - {key.upper()}")

    plt.tight_layout()
    out = sample_dir / "samples.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"저장 완료: {out}")

if __name__ == '__main__':
    main()
