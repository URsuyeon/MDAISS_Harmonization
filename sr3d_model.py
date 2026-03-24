#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D 슈퍼레졸루션(SR) 모델 구현

이 모듈은 3D 의료 영상 데이터에 대한 슈퍼레졸루션 모델을 구현합니다.
저해상도(LR) 볼륨을 고해상도(HR) 볼륨으로 변환하는 것이 목표입니다.

주요 기능:
1. 3D 슈퍼레졸루션 모델 구현 (RRDB 기반)
2. 학습 파이프라인 구현
3. 저해상도 -> 고해상도 변환 및 평가
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms
import ants
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from collections import defaultdict
import torchvision.models as models

# matplotlib 백엔드 설정
import matplotlib
matplotlib.use('Agg')  # 'Agg' 백엔드는 GUI를 사용하지 않음

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class SR3DVolumeDataset(Dataset):
    """3D 의료 영상 데이터셋 클래스 (슈퍼레졸루션용)"""
    
    def __init__(self, hr_csv_file, lr_csv_file, transform=None, lr_shape=(128, 128, 128), hr_shape=(256, 256, 256),
                 lr_spacing=(1.5, 1.5, 1.5), hr_spacing=(0.75, 0.75, 0.75)):
        """
        초기화 함수
        
        Args:
            hr_csv_file (str): 고해상도 데이터셋 CSV 파일 경로
            lr_csv_file (str): 저해상도 데이터셋 CSV 파일 경로
            transform (callable, optional): 데이터 변환 함수
            lr_shape (tuple): 저해상도 볼륨 크기
            hr_shape (tuple): 고해상도 볼륨 크기
            lr_spacing (tuple): 저해상도 spacing
            hr_spacing (tuple): 고해상도 spacing
        """
        self.hr_info = pd.read_csv(hr_csv_file)
        self.lr_info = pd.read_csv(lr_csv_file)
        assert len(self.hr_info) == len(self.lr_info), "HR/LR csv 파일의 샘플 수가 다릅니다."
        self.transform = transform
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.lr_spacing = lr_spacing
        self.hr_spacing = hr_spacing
        
        logger.info(f"SR 데이터셋 로드 완료: {len(self.hr_info)} 샘플")
    
    def __len__(self):
        return len(self.hr_info)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 고해상도 이미지 로드
        hr_img_path = self.hr_info.iloc[idx]['file_path']
        hr_img = ants.image_read(hr_img_path)
        hr_img = self._resample_volume(hr_img, self.hr_shape, self.hr_spacing)
        hr_array = hr_img.numpy()
        hr_tensor = torch.from_numpy(hr_array).float().unsqueeze(0)  # [1, H, W, D]
        
        # 저해상도 이미지 로드
        lr_img_path = self.lr_info.iloc[idx]['file_path']
        lr_img = ants.image_read(lr_img_path)
        lr_img = self._resample_volume(lr_img, self.lr_shape, self.lr_spacing)
        lr_array = lr_img.numpy()
        lr_tensor = torch.from_numpy(lr_array).float().unsqueeze(0)  # [1, h, w, d]
        
        if self.transform:
            hr_tensor = self.transform(hr_tensor)
            lr_tensor = self.transform(lr_tensor)
        
        return lr_tensor, hr_tensor

    def _resample_volume(self, img, target_shape, target_spacing):
        """
        볼륨 리샘플링 (spacing, size 모두 지정)
        
        Args:
            img (ants.ANTsImage): ANTsPy 이미지 객체
            target_shape (tuple): 목표 크기
            target_spacing (tuple): 목표 spacing
            
        Returns:
            ants.ANTsImage: 리샘플링된 이미지
        """
        current_origin = img.origin
        current_direction = img.direction

        img = ants.resample_image(
            img,
            target_spacing,
            use_voxels=False,
            interp_type=2
        )
        # shape이 정확히 맞지 않을 수 있으므로 crop/pad
        if img.shape != target_shape:
            img = self._crop_or_pad(img, target_shape)
        img.set_spacing(target_spacing)
        img.set_origin(current_origin)
        img.set_direction(current_direction)
        return img
    
    def _crop_or_pad(self, img, target_shape):
        """
        이미지 크롭 또는 패딩
        
        Args:
            img (ants.ANTsImage): ANTsPy 이미지 객체
            target_shape (tuple): 목표 크기
            
        Returns:
            ants.ANTsImage: 크롭 또는 패딩된 이미지
        """
        current_spacing = img.spacing
        current_origin = img.origin
        current_direction = img.direction
        img_array = img.numpy()
        result = np.zeros(target_shape, dtype=img_array.dtype)
        slices_src = []
        slices_dst = []
        
        for i in range(len(target_shape)):
            src_size = img_array.shape[i]
            dst_size = target_shape[i]
            
            if src_size <= dst_size:  # 패딩 필요
                start_src = 0
                start_dst = (dst_size - src_size) // 2
                end_src = src_size
                end_dst = start_dst + src_size
            else:  # 크롭 필요
                start_src = (src_size - dst_size) // 2
                start_dst = 0
                end_src = start_src + dst_size
                end_dst = dst_size
            
            slices_src.append(slice(start_src, end_src))
            slices_dst.append(slice(start_dst, end_dst))
        
        result[tuple(slices_dst)] = img_array[tuple(slices_src)]
        result_img = ants.from_numpy(result)
        result_img.set_spacing(current_spacing)
        result_img.set_origin(current_origin)
        result_img.set_direction(current_direction)
        
        return result_img

class ResidualDenseBlock3D(nn.Module):
    """3D 잔차 밀집 블록"""
    
    def __init__(self, channels, growth_channels=8, res_scale=0.2):  
        """
        초기화 함수
        
        Args:
            channels (int): 입력 및 출력 채널 수
            growth_channels (int): 성장 채널 수
            res_scale (float): 잔차 스케일링 계수
        """
        super(ResidualDenseBlock3D, self).__init__()
        self.res_scale = res_scale
        
        self.conv1 = nn.Conv3d(channels, growth_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(channels + growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(channels + 2 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(channels + 3 * growth_channels, growth_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(channels + 4 * growth_channels, channels, kernel_size=3, padding=1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        """순전파"""
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        
        return x5 * self.res_scale + x

class RRDB3D(nn.Module):
    """3D 잔차 잔차 밀집 블록"""
    
    def __init__(self, channels, growth_channels=8, res_scale=0.2):  
        """
        초기화 함수
        
        Args:
            channels (int): 입력 및 출력 채널 수
            growth_channels (int): 성장 채널 수
            res_scale (float): 잔차 스케일링 계수
        """
        super(RRDB3D, self).__init__()
        self.res_scale = res_scale
        
        self.rdb1 = ResidualDenseBlock3D(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock3D(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock3D(channels, growth_channels)
    
    def forward(self, x):
        """순전파"""
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        
        return out * self.res_scale + x

class SuperResolution3D(nn.Module):
    """3D 슈퍼레졸루션 모델 (RRDB 기반)"""
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=8, num_blocks=4, upscale_factor=2):
        super(SuperResolution3D, self).__init__()
        
        # 초기 특징 추출
        self.conv_first = nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1)
        
        # RRDB 블록
        self.rrdb_blocks = nn.ModuleList([
            RRDB3D(base_channels) for _ in range(num_blocks)
        ])
        
        # 특징 추출 후 컨볼루션
        self.conv_after_blocks = nn.Conv3d(base_channels, base_channels, kernel_size=3, padding=1)
        
        # 업샘플링 블록
        if upscale_factor == 2:
            self.upsampling = nn.Sequential(
                nn.Conv3d(base_channels, base_channels * 8, kernel_size=3, padding=1),
                PixelShuffle3d(2),  
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        elif upscale_factor == 4:
            self.upsampling = nn.Sequential(
                nn.Conv3d(base_channels, base_channels * 8, kernel_size=3, padding=1),
                PixelShuffle3d(2),  
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv3d(base_channels, base_channels * 8, kernel_size=3, padding=1),
                PixelShuffle3d(2),  
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        else:
            raise ValueError(f"지원되지 않는 업스케일 계수: {upscale_factor}")
        
        # 최종 출력 컨볼루션
        self.conv_last = nn.Conv3d(base_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        """순전파"""
        # 초기 특징 추출
        feat = self.conv_first(x)
        
        # RRDB 블록
        rrdb_out = feat
        for rrdb in self.rrdb_blocks:
            rrdb_out = rrdb(rrdb_out)
        
        # 특징 추출 후 컨볼루션
        trunk = self.conv_after_blocks(rrdb_out)
        
        # 전역 잔차 학습
        feat = trunk + feat
        
        # 업샘플링
        feat = self.upsampling(feat)
        
        # 최종 출력
        out = self.conv_last(feat)
        
        return out

class PixelShuffle3d(nn.Module):
    """3D 픽셀 셔플 모듈"""
    
    def __init__(self, upscale_factor):
        """
        초기화 함수
        
        Args:
            upscale_factor (int): 업스케일 계수
        """
        super(PixelShuffle3d, self).__init__()
        self.upscale_factor = upscale_factor
    
    def forward(self, x):
        """순전파"""
        batch_size, channels, in_depth, in_height, in_width = x.size()
        
        channels //= self.upscale_factor ** 3
        out_depth = in_depth * self.upscale_factor
        out_height = in_height * self.upscale_factor
        out_width = in_width * self.upscale_factor
        
        input_view = x.contiguous().view(
            batch_size, channels, self.upscale_factor, self.upscale_factor, self.upscale_factor,
            in_depth, in_height, in_width
        )
        
        output = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        
        return output.view(batch_size, channels, out_depth, out_height, out_width)


class VGGPerceptualLoss(nn.Module):
    """VGG 기반 Perceptual Loss (2D slice)"""
    def __init__(self, layer='relu3_3'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval()
        self.vgg = vgg
        self.layer_map = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        self.target_layer = self.layer_map.get(layer, 15)
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, sr, hr):
        # sr, hr: [B, 1, H, W, D] → 2D slice [B, 1, H, W]
        # 가운데 z-slice 사용
        b, c, h, w, d = sr.shape
        z = d // 2
        sr2d = sr[..., z]
        hr2d = hr[..., z]
        # 1채널 → 3채널 복제
        sr2d = sr2d.repeat(1, 3, 1, 1)
        hr2d = hr2d.repeat(1, 3, 1, 1)
        # VGG 입력 크기 맞추기 (224x224)
        sr2d = F.interpolate(sr2d, size=(224, 224), mode='bilinear', align_corners=False)
        hr2d = F.interpolate(hr2d, size=(224, 224), mode='bilinear', align_corners=False)
        # VGG feature 추출
        sr_feat = self.vgg[:self.target_layer](sr2d)
        hr_feat = self.vgg[:self.target_layer](hr2d)
        loss = F.l1_loss(sr_feat, hr_feat)
        return loss

class GradientLoss(nn.Module):
    """Gradient Difference Loss (3D)"""
    def __init__(self, mode='l1'):
        super().__init__()
        self.mode = mode

    def forward(self, sr, hr):
        # sr, hr: [B, 1, H, W, D]
        def gradient(x):
            dx = x[..., 1:, :, :] - x[..., :-1, :, :]
            dy = x[..., :, 1:, :] - x[..., :, :-1, :]
            dz = x[..., :, :, 1:] - x[..., :, :, :-1]
            return dx, dy, dz
        sr_dx, sr_dy, sr_dz = gradient(sr)
        hr_dx, hr_dy, hr_dz = gradient(hr)
        if self.mode == 'l1':
            loss = (
                F.l1_loss(sr_dx, hr_dx) +
                F.l1_loss(sr_dy, hr_dy) +
                F.l1_loss(sr_dz, hr_dz)
            )
        else:
            loss = (
                F.mse_loss(sr_dx, hr_dx) +
                F.mse_loss(sr_dy, hr_dy) +
                F.mse_loss(sr_dz, hr_dz)
            )
        return loss

class SuperResolution3DModel:
    """3D 슈퍼레졸루션 모델 클래스"""
    
    def __init__(self, model_dir, lr_shape=(128, 128, 128), hr_shape=(256, 256, 256),
                 lr_spacing=(1.5, 1.5, 1.5), hr_spacing=(0.75, 0.75, 0.75), upscale_factor=2):
        """
        초기화 함수
        
        Args:
            model_dir (str): 모델 저장 디렉토리 경로
            lr_shape (tuple): 저해상도 볼륨 크기
            hr_shape (tuple): 고해상도 볼륨 크기
            lr_spacing (tuple): 저해상도 spacing
            hr_spacing (tuple): 고해상도 spacing
            upscale_factor (int): 업스케일 계수
        """
        self.model_dir = Path(model_dir)
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.lr_spacing = lr_spacing
        self.hr_spacing = hr_spacing
        self.upscale_factor = upscale_factor
        
        self.best_model_dir = self.model_dir / "best_model"
        self.checkpoint_dir = self.model_dir / "checkpoints"
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.model = SuperResolution3D(
            in_channels=1,
            out_channels=1,
            base_channels=8,   
            num_blocks=4,      
            upscale_factor=upscale_factor
        )
        
        self.model = self.model.to(device)
        
        self.criterion_pixel = nn.L1Loss()
        self.criterion_perceptual = VGGPerceptualLoss().to(device)
        self.criterion_gradient = GradientLoss(mode='l1').to(device)
        
        logger.info("3D 슈퍼레졸루션 모델 초기화 완료")
    
    def save_checkpoint(self, epoch, optimizer, scheduler):
        """
        체크포인트 저장
        
        Args:
            epoch (int): 현재 에폭
            optimizer (torch.optim.Optimizer): 옵티마이저
            scheduler (torch.optim.lr_scheduler._LRScheduler): 학습률 스케줄러
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None
        }
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
        logger.info(f"체크포인트 저장 완료: 에폭 {epoch}")
    
    def save_best_model(self):
        """베스트 모델 저장"""
        torch.save(self.model.state_dict(), self.best_model_dir / "sr_best.pth")
        logger.info("베스트 모델 저장 완료")
    
    def train(self, data_dir, batch_size=1, num_epochs=100, lr=1e-4, beta1=0.9, beta2=0.999,
              save_interval=5, grad_accum_steps=1, use_mixed_precision=True):
        """
        3D 슈퍼레졸루션 모델 학습
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            batch_size (int): 배치 크기
            num_epochs (int): 에폭 수
            lr (float): 학습률
            beta1 (float): Adam 옵티마이저의 beta1 파라미터
            beta2 (float): Adam 옵티마이저의 beta2 파라미터
            save_interval (int): 체크포인트 저장 간격
            grad_accum_steps (int): 그래디언트 누적 스텝 수
            use_mixed_precision (bool): 혼합 정밀도 사용 여부
            
        Returns:
            dict: 학습 결과 (손실 등)
        """
        # 학습 폴더 생성
        training_folder_name = f"batch_{batch_size}_lr_{lr}_epochs_{num_epochs}"
        self.model_dir = self.model_dir / training_folder_name
        self.best_model_dir = self.model_dir / "best_model"
        self.checkpoint_dir = self.model_dir / "checkpoints"
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 데이터 로더 생성
        dataset = SR3DVolumeDataset(
            hr_csv_file=os.path.join(data_dir, 'train_dataset_2.csv'),
            lr_csv_file=os.path.join(data_dir, 'train_dataset.csv'),
            lr_shape=self.lr_shape,
            hr_shape=self.hr_shape,
            lr_spacing=self.lr_spacing,
            hr_spacing=self.hr_spacing
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,      
            pin_memory=False    
        )
        
        # 검증 데이터 로더 생성
        val_dataset = SR3DVolumeDataset(
            hr_csv_file=os.path.join(data_dir, 'val_dataset_2.csv'),
            lr_csv_file=os.path.join(data_dir, 'val_dataset.csv'),
            lr_shape=self.lr_shape,
            hr_shape=self.hr_shape,
            lr_spacing=self.lr_spacing,
            hr_spacing=self.hr_spacing
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,      
            pin_memory=False    
        )
        
        # Optimizer 및 Scheduler 설정
        optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2))
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        scaler = torch.amp.GradScaler('cuda', ) if use_mixed_precision else None
        losses = defaultdict(list)
        
        best_epoch = 0
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            # 학습 모드
            self.model.train()
            epoch_loss = 0.0
            
            with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for i, (lr_imgs, hr_imgs) in enumerate(data_loader):
                    lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                    
                    optimizer.zero_grad(set_to_none=True)
                    
                    if use_mixed_precision:
                        with torch.amp.autocast('cuda',):
                            sr_imgs = self.model(lr_imgs)
                            pixel_loss = self.criterion_pixel(sr_imgs, hr_imgs)
                            perceptual_loss = self.criterion_perceptual(sr_imgs, hr_imgs)
                            gradient_loss = self.criterion_gradient(sr_imgs, hr_imgs)
                            # 가중치 조정 (예시: pixel 1.0, perceptual 0.1, gradient 0.1)
                            loss = pixel_loss + 0.1 * perceptual_loss + 0.1 * gradient_loss
                        scaler.scale(loss).backward()
                        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
                            scaler.step(optimizer)
                            scaler.update()
                    else:
                        sr_imgs = self.model(lr_imgs)
                        pixel_loss = self.criterion_pixel(sr_imgs, hr_imgs)
                        perceptual_loss = self.criterion_perceptual(sr_imgs, hr_imgs)
                        gradient_loss = self.criterion_gradient(sr_imgs, hr_imgs)
                        loss = pixel_loss + 0.1 * perceptual_loss + 0.1 * gradient_loss
                        loss.backward()
                        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
                            optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.update(1)
                    pbar.set_postfix(loss=loss.item())
                    # 메모리 누수 방지
                    torch.cuda.empty_cache()
            
            # 에폭 평균 손실 계산
            avg_loss = epoch_loss / len(data_loader)
            losses['train'].append(avg_loss)
            
            # 검증
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for lr_imgs, hr_imgs in val_loader:
                    lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                    
                    sr_imgs = self.model(lr_imgs)
                    pixel_loss = self.criterion_pixel(sr_imgs, hr_imgs)
                    perceptual_loss = self.criterion_perceptual(sr_imgs, hr_imgs)
                    gradient_loss = self.criterion_gradient(sr_imgs, hr_imgs)
                    loss = pixel_loss + 0.1 * perceptual_loss + 0.1 * gradient_loss
                    val_loss += loss.item()
                    torch.cuda.empty_cache()
            
            # 검증 평균 손실 계산
            avg_val_loss = val_loss / len(val_loader)
            losses['val'].append(avg_val_loss)
            
            # 학습률 스케줄러 업데이트
            scheduler.step()
            
            # 결과 출력
            logger.info(f"Epoch {epoch + 1}/{num_epochs}: "
                       f"Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # 베스트 에포크 갱신
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_epoch = epoch + 1
                self.save_best_model()  # 베스트 모델 저장
            
            # 손실 그래프 저장
            self._plot_loss_curve(losses, epoch + 1)
            
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, optimizer, scheduler)
            
            # 5 에포크마다 샘플 생성
            if (epoch + 1) % 5 == 0:
                self._generate_and_save_samples(val_loader, epoch + 1)
        
        logger.info(f"베스트 에포크: {best_epoch}, 손실: {best_loss:.4f}")
        print(f"베스트 에포크: {best_epoch}, 손실: {best_loss:.4f}")
        return losses
    
    def _plot_loss_curve(self, losses, epoch):
        """
        손실 그래프 저장
        
        Args:
            losses (dict): 손실 딕셔너리
            epoch (int): 현재 에폭
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(losses['train']) + 1), losses['train'], label='Train Loss', color='blue')
        plt.plot(range(1, len(losses['val']) + 1), losses['val'], label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.model_dir / 'loss_curve.png')
        plt.close()

    @staticmethod
    def get_inset_position(zoom_center, img_shape, zoom_size):
        h, w = img_shape
        x, y = zoom_center
        center_x, center_y = w // 2, h // 2
        if x > center_x and y > center_y:
            return [0.05, 0.05, 0.4, 0.4]
        elif x < center_x and y > center_y:
            return [0.55, 0.05, 0.4, 0.4]
        elif x > center_x and y < center_y:
            return [0.05, 0.55, 0.4, 0.4]
        else:
            return [0.55, 0.55, 0.4, 0.4]

    @staticmethod
    def plot_with_inset(ax, base_img, zoom_img, zoom_center, zoom_size):
        ax.imshow(base_img, cmap='gray', origin='lower')
        ax.axis('off')
        x, y = zoom_center
        hs = zoom_size
        inset_pos = SuperResolution3DModel.get_inset_position(zoom_center, base_img.shape, zoom_size)
        inset_ax = ax.inset_axes(inset_pos)
        patch = zoom_img[y-hs:y+hs, x-hs:x+hs]
        inset_ax.imshow(patch, cmap='gray', origin='lower')
        inset_ax.set_title("Zoom", fontsize=8, color='white', pad=-8)
        inset_ax.axis('off')
        bounds = [x-hs, y-hs, 2*hs, 2*hs]
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

    def _generate_and_save_samples(self, val_loader, epoch, max_samples=4):
        """
        샘플 이미지 생성 및 저장 (inset plot 방식)
        """
        sample_dir = self.model_dir / 'samples'
        os.makedirs(sample_dir, exist_ok=True)
        self.model.eval()

        sample_indices = list(range(max_samples))
        sample_rel_centers = {
            0: (0.25, 0.25),
            1: (0.75, 0.25),
            2: (0.25, 0.75),
            3: (0.75, 0.75),
        }
        zoom_frac = 0.125
        samples = {}

        with torch.no_grad():
            for i, (lr_imgs, hr_imgs) in enumerate(val_loader):
                if i in sample_indices:
                    lr_imgs, hr_imgs = lr_imgs.to(device), hr_imgs.to(device)
                    sr_imgs = torch.clamp(self.model(lr_imgs), 0, 1)
                    samples[i] = {
                        'lr': lr_imgs.cpu().numpy()[0,0],
                        'hr': hr_imgs.cpu().numpy()[0,0],
                        'sr': sr_imgs.cpu().numpy()[0,0],
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
                self.plot_with_inset(ax, img2d, img2d, (cx, cy), patch_size)
                ax.set_title(f"Sample {idx} - {key.upper()}")

        plt.tight_layout()
        out = sample_dir / f"samples_epoch_{epoch}.png"
        plt.savefig(out, dpi=150)
        plt.close(fig)
        logger.info(f"샘플 저장 완료: {out}")
    
    def transform_dataset(self, data_dir, output_dir, split='test'):
        """
        데이터셋 변환 (저해상도 -> 고해상도)
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            output_dir (str): 출력 디렉토리 경로
            split (str): 데이터셋 분할 ('train', 'val', 'test')
            
        Returns:
            list: 변환된 이미지 정보
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 베스트 모델 로드
        self.model.load_state_dict(torch.load(self.best_model_dir / "sr_best.pth", map_location=device))
        logger.info("베스트 모델 로드 완료")
        
        # 데이터셋 생성 (저해상도 이미지만 필요)
        dataset = SR3DVolumeDataset(
            hr_csv_file=os.path.join(data_dir, f'{split}_transformed.csv'),
            lr_csv_file=os.path.join(data_dir, f'{split}_transformed.csv'),
            # hr_csv_file=os.path.join(data_dir, f'/public/sylee/MDAISS_2/output/{split}_dataset_2.csv'),
            # lr_csv_file=os.path.join(data_dir, f'{split}_dataset.csv'),
            lr_shape=self.lr_shape,
            hr_shape=self.hr_shape,
            lr_spacing=self.lr_spacing,
            hr_spacing=self.hr_spacing
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0      
        )
        
        transformed_info = []
        
        self.model.eval()
        
        with torch.no_grad():
            for i, (lr_imgs, _) in enumerate(tqdm(data_loader, desc=f"변환 중")):
                lr_imgs = lr_imgs.to(device)
                
                # 슈퍼레졸루션 적용
                sr_imgs = self.model(lr_imgs)
                
                # 값 범위 조정
                sr_imgs = torch.clamp(sr_imgs, 0, 1)
                
                # NumPy 배열로 변환
                sr_np = sr_imgs.cpu().numpy()[0, 0]  # (1, 1, H, W, D) -> (H, W, D)
                
                # 원본 파일 경로 및 이름 가져오기
                orig_row = dataset.lr_info.iloc[i]
                orig_path = orig_row['file_path']
                orig_filename = Path(orig_path).name
                if orig_filename.endswith('.nii.gz'):
                    orig_name = orig_filename[:-7]
                elif orig_filename.endswith('.nii'):
                    orig_name = orig_filename[:-4]
                else:
                    orig_name = Path(orig_filename).stem
                
                # 파일 저장
                out_name = f"sr_{orig_name}.nii.gz"
                out_path = output_dir / out_name
                
                # ANTs 이미지로 변환 후 저장
                sr_img = ants.from_numpy(sr_np)
                sr_img.set_spacing(self.hr_spacing)  
                ants.image_write(sr_img, str(out_path))
                
                # 기존 메타데이터 유지 및 file_path/ original_path 추가
                info = {
                    'file_path': str(out_path),
                    'original_path': orig_path
                }
                # 기존 컬럼 추가
                for col in ['hospital_id', 'patient_id', 'series_id']:
                    if col in orig_row:
                        info[col] = orig_row[col]
                transformed_info.append(info)
        
        # 변환 정보 저장 (컬럼 순서 지정)
        transformed_df = pd.DataFrame(transformed_info)
        columns = ['file_path', 'original_path', 'hospital_id', 'patient_id', 'series_id']
        # 없는 컬럼은 자동으로 제외
        columns = [col for col in columns if col in transformed_df.columns]
        transformed_df = transformed_df[columns]
        transformed_df.to_csv(output_dir / f"{split}_sr_transformed.csv", index=False)
        
        logger.info(f"데이터셋 변환 완료: {len(transformed_info)}개 이미지")
        return transformed_info

def main():
    """메인 함수"""
    # ─── 학습 모드 설정 ───
    # data_dir = '/public/sylee/MDAISS_2/output'
    # model_dir = '/public/sylee/MDAISS_2/models/sr3d/'

    # ─── 변환 모드 설정 ───
    data_dir = '/public/sylee/MDAISS_2/output/transformed/stargan3d_v2/'
    model_dir = '/public/sylee/MDAISS_2/models/sr3d/batch_2_lr_0.001_epochs_101'
    
    output_dir = '/public/sylee/MDAISS_2/output/transformed/sr3d'
    
    # 3D 슈퍼레졸루션 모델 생성
    sr_model = SuperResolution3DModel(
        model_dir=model_dir,
        lr_shape=(128, 128, 128),
        hr_shape=(256, 256, 256),
        lr_spacing=(1.5, 1.5, 1.5),
        hr_spacing=(0.75, 0.75, 0.75),
        upscale_factor=2
    )
    
    # ─── 학습 실행 ───
    # 아래 섹션의 주석을 해제하고, 변환 섹션은 주석 처리하세요.
    '''
    losses = sr_model.train(
        data_dir=data_dir,
        batch_size=2,
        num_epochs=101,
        lr=1e-4
    )
    '''

    # ─── 변환 실행 ───
    # 학습을 이미 완료한 뒤, 이 부분만 활성화하여 사용하세요.
    sr_model.transform_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        split='test' # 'train', 'val', 'test' 중 선택 가능
    )
    
if __name__ == '__main__':
    main()
