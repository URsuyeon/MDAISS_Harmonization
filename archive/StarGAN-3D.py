#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StarGAN-3D 모델 구현

이 모듈은 3D 의료 영상 데이터에 대한 StarGAN-3D 모델을 구현합니다.
병원 간 이미지 스타일 변환을 통해 병원 식별 정확도를 낮추는 것이 목표입니다.

주요 기능:
1. 3D StarGAN-3D Generator 및 Discriminator 모델 구현
2. 학습 파이프라인 구현
3. 이미지 변환 및 평가
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
import random
from collections import defaultdict
import json
import matplotlib

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

matplotlib.use('Agg') 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class MedicalVolumeDataset(Dataset):
    """3D 의료 영상 데이터셋 클래스 (StarGAN-3D용)"""
    
    def __init__(self, csv_file, hospital_id=None, transform=None, target_shape=(128, 128, 128)):
        """
        초기화 함수
        """
        self.data_info = pd.read_csv(csv_file)
        self.hospital_id = hospital_id
        self.transform = transform
        self.target_shape = target_shape
        
        if hospital_id is not None:
            self.data_info = self.data_info[self.data_info['hospital_id'] == hospital_id]
        
        if len(self.data_info) == 0:
            raise ValueError(f"병원 ID {hospital_id}에 해당하는 데이터가 없습니다.")
        
        logger.info(f"데이터셋 로드 완료: {len(self.data_info)} 샘플")
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.data_info.iloc[idx]['file_path']
        img = ants.image_read(img_path)
        img = self._resample_volume(img)
        img_array = img.numpy()
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor
    
    def _resample_volume(self, img):
        """
        볼륨 리샘플링
        """
        current_shape = img.shape
        current_spacing = img.spacing
        current_origin = img.origin
        current_direction = img.direction
        
        if current_shape != self.target_shape:
            physical_size = [s * sp for s, sp in zip(current_shape, current_spacing)]
            target_spacing = [ps / ts for ps, ts in zip(physical_size, self.target_shape)]
            interp_type = 2 
            img = ants.resample_image(
                img,
                target_spacing,
                use_voxels=False,
                interp_type=interp_type,
            )
            if img.shape != self.target_shape:
                img = self._crop_or_pad(img, self.target_shape)

        img.set_spacing(current_spacing)
        img.set_origin(current_origin)
        img.set_direction(current_direction)
        
        return img
    
    def _crop_or_pad(self, img, target_shape):
        """
        이미지 크롭 또는 패딩
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
            
            if src_size <= dst_size:  
                start_src = 0
                start_dst = (dst_size - src_size) // 2
                end_src = src_size
                end_dst = start_dst + src_size
            else:  
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

class Conv3DBlock(nn.Module):
    """3D 컨볼루션 블록"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batch_norm=True, use_leaky=False):
        """
        초기화 함수
        """
        super(Conv3DBlock, self).__init__()
        
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=not use_batch_norm
        )
        
        self.bn = nn.BatchNorm3d(out_channels) if use_batch_norm else None
        
        if use_leaky:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """순전파"""
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.activation(x)
        return x

class TransConv3DBlock(nn.Module):
    """3D 전치 컨볼루션 블록"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, use_batch_norm=True):
        """
        초기화 함수
        """
        super(TransConv3DBlock, self).__init__()
        
        self.transconv = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=not use_batch_norm
        )
        
        self.bn = nn.BatchNorm3d(out_channels) if use_batch_norm else None
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """순전파"""
        x = self.transconv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock3D(nn.Module):
    """3D 잔차 블록"""
    
    def __init__(self, channels):
        """
        초기화 함수
        """
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
    
    def forward(self, x):
        """순전파"""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out

class StarGAN3DGenerator(nn.Module):
    """StarGAN-3D Generator (U-Net3D 기반)"""
    
    def __init__(self, input_channels=1, output_channels=1, base_filters=16, n_residual_blocks=2, domain_dim=4):
        super(StarGAN3DGenerator, self).__init__()
        
        self.enc1 = Conv3DBlock(input_channels + domain_dim, base_filters)
        self.enc2 = Conv3DBlock(base_filters, base_filters * 2, stride=2)
        self.enc3 = Conv3DBlock(base_filters * 2, base_filters * 4, stride=2)

        res_blocks = [ResidualBlock3D(base_filters * 4) for _ in range(n_residual_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)
        
        self.dec3 = TransConv3DBlock(base_filters * 4, base_filters * 2)
        self.dec3_conv = Conv3DBlock(base_filters * 2 + base_filters * 2, base_filters * 2)
        
        self.dec2 = TransConv3DBlock(base_filters * 2, base_filters)
        self.dec2_conv = Conv3DBlock(base_filters + base_filters, base_filters)
        
        self.final = nn.Sequential(
            nn.Conv3d(base_filters, output_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x, domain_code):
        """순전파"""
        domain_code = domain_code.view(domain_code.size(0), domain_code.size(1), 1, 1, 1)
        domain_code = domain_code.expand(-1, -1, x.size(2), x.size(3), x.size(4))
        x = torch.cat([x, domain_code], dim=1)
        
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        r = self.res_blocks(e3)
        
        d3 = self.dec3(r)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3_conv(d3)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2_conv(d2)
        
        out = self.final(d2)
        return out

class StarGAN3DDiscriminator(nn.Module):
    """StarGAN-3D Discriminator (PatchGAN + Domain Classification)"""
    
    def __init__(self, input_channels=1, base_filters=32, domain_dim=4):
        super(StarGAN3DDiscriminator, self).__init__()
        
        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.layers = nn.ModuleList()
        in_channels = base_filters
        for i in range(3):
            out_channels = min(base_filters * (2 ** (i + 1)), 512)
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        
        self.gan_head = nn.Conv3d(in_channels, 1, kernel_size=4, stride=1, padding=1)
        self.domain_head = nn.Conv3d(in_channels, domain_dim, kernel_size=4, stride=1, padding=1)
    def forward(self, x):
        """순전파"""
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        gan_out = self.gan_head(x)
        domain_out = self.domain_head(x).view(x.size(0), -1)
        return gan_out, domain_out

class StarGAN3D:
    """StarGAN-3D 모델 클래스"""
    
    def __init__(self, model_dir, input_shape=(128, 128, 128), domain_dim=4):
        self.model_dir = Path(model_dir)
        self.input_shape = input_shape
        self.domain_dim = domain_dim
        
        self.best_model_dir = self.model_dir / "best_model"
        self.checkpoint_dir = self.model_dir / "checkpoints"
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.G = StarGAN3DGenerator(input_channels=1, output_channels=1, domain_dim=domain_dim)
        self.D = StarGAN3DDiscriminator(input_channels=1, domain_dim=domain_dim)
        
        self.G = self.G.to(device)
        self.D = self.D.to(device)
        
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        self.criterion_domain = nn.CrossEntropyLoss()
        
        logger.info("StarGAN-3D 모델 초기화 완료")

    def save_checkpoint(self, epoch, optimizer_G, optimizer_D, scheduler_G, scheduler_D):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'G_state_dict': self.G.state_dict(),
            'D_state_dict': self.D.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
        logger.info(f"체크포인트 저장 완료: 에폭 {epoch}")

    def save_best_model(self):
        """베스트 모델 저장"""
        torch.save(self.G.state_dict(), self.best_model_dir / "G_best.pth")
        torch.save(self.D.state_dict(), self.best_model_dir / "D_best.pth")
        logger.info("베스트 모델 저장 완료")

    def train(self, data_dir, batch_size=1, num_epochs=10, lr=1e-5, beta1=0.5, beta2=0.999,
              lambda_identity=5.0, lambda_cycle=10.0, save_interval=5, grad_accum_steps=4, 
              use_mixed_precision=False, num_workers=4, pin_memory=True):  
        """
        StarGAN-3D 학습
        """
        # 학습 폴더 생성
        training_folder_name = f"batch_{batch_size}_lr_{lr}_epochs_{num_epochs}"
        self.model_dir = self.model_dir / training_folder_name
        self.best_model_dir = self.model_dir / "best_model"
        self.checkpoint_dir = self.model_dir / "checkpoints"
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # 데이터 로더 생성
        dataset = MedicalVolumeDataset(
            csv_file=os.path.join(data_dir, 'train_dataset.csv'),
            hospital_id=None,
            target_shape=self.input_shape
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,  
            pin_memory=pin_memory     
        )

        # Optimizer 및 Scheduler 설정
        optimizer_G = optim.Adam(self.G.parameters(), lr=lr, betas=(beta1, beta2))
        optimizer_D = optim.Adam(self.D.parameters(), lr=lr, betas=(beta1, beta2))

        scheduler_G = optim.lr_scheduler.OneCycleLR(
            optimizer_G, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(data_loader)
        )
        scheduler_D = optim.lr_scheduler.OneCycleLR(
            optimizer_D, max_lr=lr, epochs=num_epochs, steps_per_epoch=len(data_loader)
        )

        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        losses = defaultdict(list)

        best_epoch = 0
        best_loss = float('inf')

        for epoch in range(num_epochs):
            epoch_losses = defaultdict(list)

            with tqdm(total=len(data_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for i, real_X in enumerate(data_loader):
                    real_X = real_X.to(device)

                    domain_code = torch.randint(0, self.domain_dim, (real_X.size(0),), device=device)
                    domain_code_onehot = F.one_hot(domain_code, num_classes=self.domain_dim).float()

                    # Generator 학습
                    optimizer_G.zero_grad(set_to_none=True)
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            fake_Y = self.G(real_X, domain_code_onehot)
                            pred_fake_Y, pred_domain_Y = self.D(fake_Y)

                            # 손실 계산
                            loss_GAN = self.criterion_GAN(pred_fake_Y, torch.ones_like(pred_fake_Y))
                            loss_identity = self.criterion_identity(fake_Y, real_X) * lambda_identity
                            loss_domain = self.criterion_domain(pred_domain_Y, domain_code)
                            loss_G = loss_GAN + loss_identity + loss_domain

                        scaler.scale(loss_G).backward()
                        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
                            scaler.step(optimizer_G)
                            scaler.update()
                            scheduler_G.step()
                    else:
                        fake_Y = self.G(real_X, domain_code_onehot)
                        pred_fake_Y, pred_domain_Y = self.D(fake_Y)

                        # 손실 계산
                        loss_GAN = self.criterion_GAN(pred_fake_Y, torch.ones_like(pred_fake_Y))
                        loss_identity = self.criterion_identity(fake_Y, real_X) * lambda_identity
                        loss_domain = self.criterion_domain(pred_domain_Y, domain_code)
                        loss_G = loss_GAN + loss_identity + loss_domain

                        loss_G.backward()
                        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
                            optimizer_G.step()
                            scheduler_G.step()

                    # Discriminator 학습
                    optimizer_D.zero_grad(set_to_none=True)
                    if use_mixed_precision:
                        with torch.cuda.amp.autocast():
                            pred_real_X, pred_domain_X = self.D(real_X)
                            pred_fake_Y, _ = self.D(fake_Y.detach())

                            # 손실 계산
                            loss_real = self.criterion_GAN(pred_real_X, torch.ones_like(pred_real_X))
                            loss_fake = self.criterion_GAN(pred_fake_Y, torch.zeros_like(pred_fake_Y))
                            loss_domain = self.criterion_domain(pred_domain_X, domain_code)
                            loss_D = (loss_real + loss_fake) * 0.5 + loss_domain

                        scaler.scale(loss_D).backward()
                        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
                            scaler.step(optimizer_D)
                            scaler.update()
                            scheduler_D.step()
                    else:
                        pred_real_X, pred_domain_X = self.D(real_X)
                        pred_fake_Y, _ = self.D(fake_Y.detach())

                        # 손실 계산
                        loss_real = self.criterion_GAN(pred_real_X, torch.ones_like(pred_real_X))
                        loss_fake = self.criterion_GAN(pred_fake_Y, torch.zeros_like(pred_fake_Y))
                        loss_domain = self.criterion_domain(pred_domain_X, domain_code)
                        loss_D = (loss_real + loss_fake) * 0.5 + loss_domain

                        loss_D.backward()
                        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(data_loader):
                            optimizer_D.step()
                            scheduler_D.step()

                    # 손실 기록
                    epoch_losses['G'].append(loss_G.item())
                    epoch_losses['D'].append(loss_D.item())
                    pbar.update(1)

            # 에포크 손실 기록
            avg_loss_G = np.mean(epoch_losses['G'])
            avg_loss_D = np.mean(epoch_losses['D'])
            losses['G'].append(avg_loss_G)
            losses['D'].append(avg_loss_D)

            # 베스트 에포크 갱신
            total_loss = avg_loss_G + avg_loss_D
            if total_loss < best_loss:
                best_loss = total_loss
                best_epoch = epoch + 1
                self.save_best_model()  # 베스트 모델 저장

            # 손실 그래프 저장
            self._plot_loss_curve(losses, epoch + 1)
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch + 1, optimizer_G, optimizer_D, scheduler_G, scheduler_D)

            # 10 에포크마다 샘플 생성
            if (epoch + 1) % 10 == 0:
                self._generate_and_save_samples(data_dir, epoch + 1)

        logger.info(f"베스트 에포크: {best_epoch}, 손실: {best_loss:.4f}")
        print(f"베스트 에포크: {best_epoch}, 손실: {best_loss:.4f}")
        return losses

    def _plot_loss_curve(self, losses, epoch):
        """
        손실 그래프 저장
        """
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, len(losses['G']) + 1), losses['G'], label='Generator Loss', color='blue')
        plt.plot(range(1, len(losses['D']) + 1), losses['D'], label='Discriminator Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(self.model_dir / 'loss_curve.png')  
        plt.close()

    def _generate_and_save_samples(self, data_dir, epoch):
        """
        샘플 이미지 생성 및 저장
        """
        dataset = MedicalVolumeDataset(
            csv_file=os.path.join(data_dir, 'val_dataset.csv'),
            hospital_id=None, 
            target_shape=self.input_shape
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        sample_dir = self.model_dir / 'samples'
        os.makedirs(sample_dir, exist_ok=True)

        self.G.eval()

        with torch.no_grad():
            for i, real_X in enumerate(data_loader):
                if i >= 4:  
                    break

                real_X = real_X.to(device)

                domain_codes = torch.arange(self.domain_dim, device=device)
                domain_codes_onehot = F.one_hot(domain_codes, num_classes=self.domain_dim).float()

                # Forward pass
                fake_Ys = [self.G(real_X, domain_code.unsqueeze(0)) for domain_code in domain_codes_onehot]

                self._save_combined_sample_slices(real_X, fake_Ys, sample_dir, epoch, i)

    def _save_combined_sample_slices(self, real_X, fake_Ys, sample_dir, epoch, sample_idx):
        """
        샘플 슬라이스 저장 
        """
        real_X_np = real_X.cpu().numpy()[0, 0]
        fake_Ys_np = [fake_Y.cpu().numpy()[0, 0] for fake_Y in fake_Ys]

        z_idx = real_X_np.shape[2] // 2

        real_X_slice = real_X_np[:, :, z_idx]
        fake_Y_slices = [fake_Y_np[:, :, z_idx] for fake_Y_np in fake_Ys_np]

        fig, axes = plt.subplots(4, 4, figsize=(15, 15))

        # 원본 이미지 (첫 번째 열)
        for j in range(4):
            axes[j, 0].imshow(real_X_slice, cmap='gray')
            axes[j, 0].set_title(f'Real X (Sample {j})')
            axes[j, 0].axis('off')

        # 변환된 이미지 (나머지 열)
        for j, fake_Y_slice in enumerate(fake_Y_slices):
            for k in range(4):
                axes[k, j + 1].imshow(fake_Y_slice, cmap='gray')
                axes[k, j + 1].set_title(f'Fake Y (Domain {j})')
                axes[k, j + 1].axis('off')

        plt.tight_layout()
        plt.savefig(sample_dir / f"combined_sample_epoch_{epoch}_sample_{sample_idx}.png")
        plt.close()

    def save_models(self, epoch):
        """
        모델 저장
        """
        torch.save(self.G.state_dict(), self.model_dir / f"G_{epoch}.pth")
        torch.save(self.D.state_dict(), self.model_dir / f"D_{epoch}.pth")
        
        logger.info(f"모델 저장 완료: 에폭 {epoch}")
    
    def load_models(self, epoch):
        """
        모델 로드
        """
        self.G.load_state_dict(torch.load(self.model_dir / f"G_{epoch}.pth"))
        self.D.load_state_dict(torch.load(self.model_dir / f"D_{epoch}.pth"))
        
        logger.info(f"모델 로드 완료: 에폭 {epoch}")
    
    def transform_dataset(self, data_dir, output_dir, split='test'):
        """
        데이터셋 변환
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        dataset = MedicalVolumeDataset(
            csv_file=os.path.join(data_dir, f'{split}_dataset.csv'),
            hospital_id=None, 
            target_shape=self.input_shape
        )

        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        transformed_info = []

        self.G.eval()

        with torch.no_grad():
            for i, real_X in enumerate(tqdm(data_loader, desc=f"변환 중")):
                real_X = real_X.to(device)

                domain_code = torch.randint(0, self.domain_dim, (real_X.size(0),), device=device)
                domain_code_onehot = F.one_hot(domain_code, num_classes=self.domain_dim).float()

                fake_Y = self.G(real_X, domain_code_onehot)

                fake_Y = (fake_Y + 1) / 2  # [-1, 1] -> [0, 1]

                fake_Y_np = fake_Y.cpu().numpy()[0, 0]  # (B, C, H, W, D) -> (H, W, D)

                # 파일 저장
                filename = f"transformed_{i}.nii.gz"
                output_path = output_dir / filename
                ants.image_write(ants.from_numpy(fake_Y_np), str(output_path))

                transformed_info.append({'index': i, 'output_path': str(output_path)})

        transformed_df = pd.DataFrame(transformed_info)
        transformed_df.to_csv(output_dir / f"{split}_transformed.csv", index=False)

        logger.info(f"데이터셋 변환 완료: {len(transformed_info)}개 이미지")
        return transformed_info
    
    def _generate_and_save_samples(self, data_dir, epoch):
        """
        샘플 이미지 생성 및 저장
        """
        dataset = MedicalVolumeDataset(
            csv_file=os.path.join(data_dir, 'val_dataset.csv'),
            hospital_id=None, 
            target_shape=self.input_shape
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )

        sample_dir = self.model_dir / 'samples'
        os.makedirs(sample_dir, exist_ok=True)

        self.G.eval()

        with torch.no_grad():
            for i, real_X in enumerate(data_loader):
                if i >= 5:
                    break

                real_X = real_X.to(device)

                domain_code = torch.randint(0, self.domain_dim, (real_X.size(0),), device=device)
                domain_code_onehot = F.one_hot(domain_code, num_classes=self.domain_dim).float()

                # Forward pass
                fake_Y = self.G(real_X, domain_code_onehot)
                fake_X = self.G(fake_Y, domain_code_onehot)  
                real_Y = real_X  

                # 샘플 저장
                self._save_sample_slices(real_X, fake_Y, fake_X, real_Y, sample_dir, epoch, i)

    def _save_sample_slices(self, real_X, fake_Y, fake_X, real_Y, sample_dir, epoch, sample_idx):
        """
        샘플 슬라이스 저장
        """
        real_X_np = real_X.cpu().numpy()[0, 0]
        fake_Y_np = fake_Y.cpu().numpy()[0, 0]
        fake_X_np = fake_X.cpu().numpy()[0, 0]
        real_Y_np = real_Y.cpu().numpy()[0, 0]

        z_idx = real_X_np.shape[2] // 2

        real_X_slice = real_X_np[:, :, z_idx]
        fake_Y_slice = fake_Y_np[:, :, z_idx]
        fake_X_slice = fake_X_np[:, :, z_idx]
        real_Y_slice = real_Y_np[:, :, z_idx]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].imshow(real_X_slice, cmap='gray')
        axes[0, 0].set_title('Real X')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(fake_Y_slice, cmap='gray')
        axes[0, 1].set_title('Fake Y')
        axes[0, 1].axis('off')
        
        axes[1, 0].imshow(fake_X_slice, cmap='gray')
        axes[1, 0].set_title('Fake X')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(real_Y_slice, cmap='gray')
        axes[1, 1].set_title('Real Y')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(sample_dir / f"sample_{epoch}_{sample_idx}.png")
        plt.close()
    
    def _plot_training_curves(self, losses):
        """
        학습 곡선 시각화
        """
        batch_per_epoch = len(losses['G']) // len(losses['G_cycle'])
        epoch_losses = defaultdict(list)
        
        for key in losses:
            for i in range(0, len(losses[key]), batch_per_epoch):
                epoch_losses[key].append(np.mean(losses[key][i:i+batch_per_epoch]))
        
        epochs = range(1, len(epoch_losses['G']) + 1)

        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 1, 1)
        plt.plot(epochs, epoch_losses['G'], 'b-', label='Generator Loss')
        plt.plot(epochs, epoch_losses['G_identity'], 'r-', label='Identity Loss')
        plt.plot(epochs, epoch_losses['G_GAN'], 'g-', label='GAN Loss')
        plt.plot(epochs, epoch_losses['G_cycle'], 'y-', label='Cycle Loss')
        plt.title('Generator Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(epochs, epoch_losses['D'], 'b-', label='Discriminator Loss')
        plt.title('Discriminator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_curves.png')
        plt.close()

def main():
    """메인 함수"""
    data_dir = '/public/sylee/MDAISS/output'
    model_dir = '/public/sylee/MDAISS/models/stargan3d'
    output_dir = '/public/sylee/MDAISS/output/transformed/stargan3d'

    stargan = StarGAN3D(model_dir=model_dir, input_shape=(128, 128, 128), domain_dim=4)

    # 학습
    losses = stargan.train(
        data_dir=data_dir,
        batch_size=1,
        num_epochs=200,
        lr=1e-4,
        num_workers=8,  
        pin_memory=True
    )

    # 베스트 에포크 출력
    logger.info(f"최종 베스트 에포크: {max(range(len(losses['G'])), key=lambda i: losses['G'][i]) + 1}")
    print(f"최종 베스트 에포크: {max(range(len(losses['G'])), key=lambda i: losses['G'][i]) + 1}")

    # 데이터셋 변환
    stargan.transform_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        split='test'
    )

if __name__ == '__main__':
    main()
