#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CycleGAN 모델 구현

이 모듈은 3D 의료 영상 데이터에 대한 CycleGAN 모델을 구현합니다.
병원 간 이미지 스타일 변환을 통해 병원 식별 정확도를 낮추는 것이 목표입니다.

주요 기능:
1. 3D CycleGAN Generator 및 Discriminator 모델 구현
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

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class MedicalVolumeDataset(Dataset):
    """3D 의료 영상 데이터셋 클래스 (CycleGAN용)"""
    
    def __init__(self, csv_file, hospital_id, transform=None, target_shape=(128, 128, 128)):
        """
        초기화 함수
        
        Args:
            csv_file (str): 데이터셋 CSV 파일 경로
            hospital_id (str): 대상 병원 ID
            transform (callable, optional): 데이터 변환 함수
            target_shape (tuple, optional): 입력 볼륨의 목표 크기
        """
        self.data_info = pd.read_csv(csv_file)
        self.hospital_id = hospital_id
        self.transform = transform
        self.target_shape = target_shape
        
        # 해당 병원 데이터만 필터링
        self.data_info = self.data_info[self.data_info['hospital_id'] == hospital_id]
        
        if len(self.data_info) == 0:
            raise ValueError(f"병원 ID {hospital_id}에 해당하는 데이터가 없습니다.")
        
        logger.info(f"병원 {hospital_id} 데이터셋 로드 완료: {len(self.data_info)} 샘플")
    
    def __len__(self):
        """데이터셋 길이 반환"""
        return len(self.data_info)
    
    def __getitem__(self, idx):
        """데이터셋 아이템 반환"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 이미지 파일 경로
        img_path = self.data_info.iloc[idx]['file_path']
        
        # ANTsPy로 이미지 로드
        img = ants.image_read(img_path)
        
        # 이미지 리샘플링 (목표 크기로)
        img = self._resample_volume(img)
        
        # NumPy 배열로 변환
        img_array = img.numpy()
        
        # 차원 추가 (채널 차원)
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)
        
        # 변환 적용 (있는 경우)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor
    
    def _resample_volume(self, img):
        """
        볼륨 리샘플링
        
        Args:
            img (ants.ANTsImage): ANTsPy 이미지 객체
            
        Returns:
            ants.ANTsImage: 리샘플링된 이미지
        """
        # 현재 이미지 크기와 물리적 정보 저장
        current_shape = img.shape
        current_spacing = img.spacing
        current_origin = img.origin
        current_direction = img.direction
        
        # 목표 크기와 다른 경우에만 리샘플링
        if current_shape != self.target_shape:
            # 물리적 크기 계산 (mm 단위)
            physical_size = [s * sp for s, sp in zip(current_shape, current_spacing)]
            
            # 목표 스페이싱 계산 (물리적 크기 유지)
            target_spacing = [ps / ts for ps, ts in zip(physical_size, self.target_shape)]
            
            # 고품질 보간을 위한 파라미터 설정
            # 1: 선형 보간 (기본값)
            # 2: B-스플라인 보간 (더 부드러운 결과)
            # 3: 가우시안 보간 (가장 부드러운 결과)
            interp_type = 2  # B-스플라인 보간 사용
            
            # 리샘플링 수행 (물리적 크기 유지)
            img = ants.resample_image(
                img,
                target_spacing,
                use_voxels=False,
                interp_type=interp_type,
            )
            
            # 크기 확인 및 조정
            if img.shape != self.target_shape:
                # 크롭 또는 패딩 (물리적 정보 보존)
                img = self._crop_or_pad(img, self.target_shape)
        
        # 물리적 정보 복원
        img.set_spacing(current_spacing)
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
        # 현재 이미지의 물리적 정보 저장
        current_spacing = img.spacing
        current_origin = img.origin
        current_direction = img.direction
        
        # NumPy 배열로 변환
        img_array = img.numpy()
        
        # 결과 배열 초기화 (0으로 채움)
        result = np.zeros(target_shape, dtype=img_array.dtype)
        
        # 각 차원에 대해 크롭 또는 패딩
        slices_src = []
        slices_dst = []
        
        for i in range(len(target_shape)):
            src_size = img_array.shape[i]
            dst_size = target_shape[i]
            
            if src_size <= dst_size:  # 패딩 필요
                # 중앙 정렬을 위한 시작 위치 계산
                start_src = 0
                start_dst = (dst_size - src_size) // 2
                end_src = src_size
                end_dst = start_dst + src_size
            else:  # 크롭 필요
                # 중앙 정렬을 위한 시작 위치 계산
                start_src = (src_size - dst_size) // 2
                start_dst = 0
                end_src = start_src + dst_size
                end_dst = dst_size
            
            slices_src.append(slice(start_src, end_src))
            slices_dst.append(slice(start_dst, end_dst))
        
        # 크롭 또는 패딩 적용
        result[tuple(slices_dst)] = img_array[tuple(slices_src)]
        
        # ANTsImage로 변환
        result_img = ants.from_numpy(result)
        
        # 물리적 정보 복원
        result_img.set_spacing(current_spacing)
        result_img.set_origin(current_origin)
        result_img.set_direction(current_direction)
        
        return result_img

class CycleGANDataset:
    """CycleGAN 학습을 위한 데이터셋 클래스"""
    
    def __init__(self, data_dir, hospital_ids, split='train', target_shape=(128, 128, 128)):
        """
        초기화 함수
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            hospital_ids (list): 병원 ID 목록
            split (str): 데이터셋 분할 ('train', 'val', 'test')
            target_shape (tuple): 입력 볼륨의 목표 크기
        """
        self.data_dir = Path(data_dir)
        self.hospital_ids = hospital_ids
        self.split = split
        self.target_shape = target_shape
        
        # 병원별 데이터셋 생성
        self.datasets = {}
        for hospital_id in hospital_ids:
            try:
                self.datasets[hospital_id] = MedicalVolumeDataset(
                    csv_file=str(self.data_dir / f'{split}_dataset.csv'),
                    hospital_id=hospital_id,
                    target_shape=target_shape
                )
            except ValueError as e:
                logger.warning(f"병원 {hospital_id} 데이터셋 로드 실패: {str(e)}")
        
        # 유효한 병원 ID만 유지
        self.hospital_ids = [h_id for h_id in hospital_ids if h_id in self.datasets]
        
        if len(self.hospital_ids) < 2:
            raise ValueError("CycleGAN 학습을 위해서는 최소 2개 이상의 병원 데이터가 필요합니다.")
        
        # 각 병원별 데이터 길이
        self.dataset_lengths = {h_id: len(self.datasets[h_id]) for h_id in self.hospital_ids}
        
        logger.info(f"CycleGAN 데이터셋 초기화 완료: {self.hospital_ids}")
        for h_id in self.hospital_ids:
            logger.info(f"  - 병원 {h_id}: {self.dataset_lengths[h_id]} 샘플")
    
    def get_hospital_pair_loader(self, source_id, target_id, batch_size=1, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True):
        """
        병원 쌍에 대한 데이터 로더 생성
        
        Args:
            source_id (str): 소스 병원 ID
            target_id (str): 타겟 병원 ID
            batch_size (int): 배치 크기
            shuffle (bool): 셔플 여부
            num_workers (int): 워커 수
            pin_memory (bool): GPU 메모리로 빠르게 전송
            persistent_workers (bool): 워커 재사용
            
        Returns:
            tuple: (source_loader, target_loader)
        """
        if source_id not in self.datasets or target_id not in self.datasets:
            raise ValueError(f"병원 ID {source_id} 또는 {target_id}에 해당하는 데이터셋이 없습니다.")
        
        source_dataset = self.datasets[source_id]
        target_dataset = self.datasets[target_id]
        
        source_loader = DataLoader(
            source_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        
        target_loader = DataLoader(
            target_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers
        )
        
        return source_loader, target_loader

# 3D 컨볼루션 블록
class Conv3DBlock(nn.Module):
    """3D 컨볼루션 블록"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batch_norm=True, use_leaky=False):
        """
        초기화 함수
        
        Args:
            in_channels (int): 입력 채널 수
            out_channels (int): 출력 채널 수
            kernel_size (int): 커널 크기
            stride (int): 스트라이드
            padding (int): 패딩
            use_batch_norm (bool): 배치 정규화 사용 여부
            use_leaky (bool): LeakyReLU 사용 여부
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

# 3D 전치 컨볼루션 블록
class TransConv3DBlock(nn.Module):
    """3D 전치 컨볼루션 블록"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, use_batch_norm=True):
        """
        초기화 함수
        
        Args:
            in_channels (int): 입력 채널 수
            out_channels (int): 출력 채널 수
            kernel_size (int): 커널 크기
            stride (int): 스트라이드
            padding (int): 패딩
            output_padding (int): 출력 패딩
            use_batch_norm (bool): 배치 정규화 사용 여부
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

# 3D 잔차 블록
class ResidualBlock3D(nn.Module):
    """3D 잔차 블록"""
    
    def __init__(self, channels):
        """
        초기화 함수
        
        Args:
            channels (int): 채널 수
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

# 3D Generator 모델
class Unet3D(nn.Module):
    """3D Unet 모델"""
    
    def __init__(self, input_channels=1, output_channels=1, base_filters=16, n_residual_blocks=2):
        """
        초기화 함수
        
        Args:
            input_channels (int): 입력 채널 수
            output_channels (int): 출력 채널 수
            base_filters (int): 기본 필터 수 (16)
            n_residual_blocks (int): residual 블록 수 (2)
        """
        super(Unet3D, self).__init__()
        
        # 인코더 부분 
        self.enc1 = Conv3DBlock(input_channels, base_filters)
        self.enc2 = Conv3DBlock(base_filters, base_filters*2, stride=2)
        self.enc3 = Conv3DBlock(base_filters*2, base_filters*4, stride=2)
        
        # 잔차 블록 
        res_blocks = []
        for _ in range(n_residual_blocks):
            res_blocks.append(ResidualBlock3D(base_filters*4))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # 디코더 부분 
        self.dec3 = TransConv3DBlock(base_filters*4, base_filters*2)
        self.dec3_conv = Conv3DBlock(base_filters*2 + base_filters*2, base_filters*2)  
        
        self.dec2 = TransConv3DBlock(base_filters*2, base_filters)
        self.dec2_conv = Conv3DBlock(base_filters + base_filters, base_filters) 
        
        # 최종 컨볼루션 레이어
        self.final = nn.Sequential(
            nn.Conv3d(base_filters, output_channels, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        """순전파"""
        # 인코더
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        # 잔차 블록
        r = self.res_blocks(e3)
        
        # 디코더 (스킵 커넥션 포함)
        d3 = self.dec3(r)
        d3 = torch.cat([d3, e2], dim=1)
        d3 = self.dec3_conv(d3)
        
        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2_conv(d2)
        
        # 최종 출력
        out = self.final(d2)
        
        return out

# 3D Generator 모델 (기존 클래스는 Unet3D로 대체)
Generator3D = Unet3D

# 3D Discriminator 모델
class Discriminator3D(nn.Module):
    """3D Discriminator 모델 (PatchGAN)"""
    
    def __init__(self, input_channels=1, base_filters=32, n_layers=2):
        """
        초기화 함수
        
        Args:
            input_channels (int): 입력 채널 수
            base_filters (int): 기본 필터 수 (32)
            n_layers (int): 레이어 수 (2)
        """
        super(Discriminator3D, self).__init__()
        
        # 초기 컨볼루션 레이어 (배치 정규화 없음)
        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 중간 레이어
        self.layers = nn.ModuleList()
        in_channels = base_filters
        for i in range(1, n_layers):
            out_channels = min(base_filters * (2 ** i), 512)
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        
        # 최종 레이어
        self.final = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=4, stride=1, padding=1)
        )
    
    def forward(self, x):
        """순전파"""
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final(x)
        return x

# CycleGAN 모델
class CycleGAN3D:
    """3D CycleGAN 모델 클래스"""
    
    def __init__(self, source_id, target_id, model_dir, input_shape=(128, 128, 128)):
        """
        초기화 함수
        
        Args:
            source_id (str): 소스 병원 ID
            target_id (str): 타겟 병원 ID
            model_dir (str): 모델 저장 디렉토리 경로
            input_shape (tuple): 입력 이미지 크기 (기본값: (128, 128, 128))
        """
        self.source_id = source_id
        self.target_id = target_id
        self.model_dir = Path(model_dir) / f"{source_id}_to_{target_id}"
        self.input_shape = input_shape
        
        # 모델 디렉토리 생성
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 체크포인트 디렉토리 생성
        self.checkpoint_dir = self.model_dir / 'checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Generator 모델 (G: source -> target, F: target -> source)
        self.G = Unet3D(input_channels=1, output_channels=1)
        self.F = Unet3D(input_channels=1, output_channels=1)
        
        # Discriminator 모델 (D_X: source 판별, D_Y: target 판별)
        self.D_X = Discriminator3D(input_channels=1)
        self.D_Y = Discriminator3D(input_channels=1)
        
        # 모델을 디바이스로 이동
        self.G = self.G.to(device)
        self.F = self.F.to(device)
        self.D_X = self.D_X.to(device)
        self.D_Y = self.D_Y.to(device)
        
        # 손실 함수
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        logger.info(f"CycleGAN 모델 초기화 완료: {source_id} -> {target_id}")
    
    def save_checkpoint(self, epoch, optimizer_G, optimizer_D, scheduler_G, scheduler_D):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'G_state_dict': self.G.state_dict(),
            'F_state_dict': self.F.state_dict(),
            'D_X_state_dict': self.D_X.state_dict(),
            'D_Y_state_dict': self.D_Y.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'scheduler_G_state_dict': scheduler_G.state_dict(),
            'scheduler_D_state_dict': scheduler_D.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth')
    
    def load_checkpoint(self, epoch):
        """체크포인트 로드"""
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.G.load_state_dict(checkpoint['G_state_dict'])
            self.F.load_state_dict(checkpoint['F_state_dict'])
            self.D_X.load_state_dict(checkpoint['D_X_state_dict'])
            self.D_Y.load_state_dict(checkpoint['D_Y_state_dict'])
            return checkpoint
        return None
    
    def train(self, data_dir, batch_size=1, num_epochs=10, lr=1e-5, beta1=0.5, beta2=0.999,
              lambda_cycle=10.0, lambda_identity=5.0, save_interval=5, grad_accum_steps=4,
              use_mixed_precision=False, clear_cache_freq=10, start_epoch=0):
        """
        모델 학습
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            batch_size (int): 배치 크기
            num_epochs (int): 에폭 수
            lr (float): 학습률
            beta1 (float): Adam 옵티마이저의 beta1 파라미터
            beta2 (float): Adam 옵티마이저의 beta2 파라미터
            lambda_cycle (float): 사이클 손실 가중치
            lambda_identity (float): 아이덴티티 손실 가중치
            save_interval (int): 모델 저장 간격 (에폭)
            grad_accum_steps (int): 그라디언트 누적 단계 수
            use_mixed_precision (bool): 혼합 정밀도 사용 여부 (메모리 절약)
            clear_cache_freq (int): 메모리 캐시 정리 빈도 (배치 단위)
            start_epoch (int): 시작할 에폭
        """
        # 데이터셋 로드
        dataset = CycleGANDataset(
            data_dir=data_dir,
            hospital_ids=[self.source_id, self.target_id],
            split='train',
            target_shape=self.input_shape
        )
        
        # 데이터 로더 생성 (최적화된 설정)
        source_loader, target_loader = dataset.get_hospital_pair_loader(
            source_id=self.source_id,
            target_id=self.target_id,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # CPU 코어 수에 따라 조정
            pin_memory=True,  # GPU 메모리로 빠르게 전송
            persistent_workers=True  # 워커 재사용
        )
        
        # 옵티마이저 설정
        optimizer_G = optim.Adam(
            list(self.G.parameters()) + list(self.F.parameters()),
            lr=lr, betas=(beta1, beta2)
        )
        optimizer_D = optim.Adam(
            list(self.D_X.parameters()) + list(self.D_Y.parameters()),
            lr=lr, betas=(beta1, beta2)
        )
        
        # 학습률 스케줄러 (동적 학습률 조정)
        scheduler_G = optim.lr_scheduler.OneCycleLR(
            optimizer_G,
            max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=len(source_loader),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        scheduler_D = optim.lr_scheduler.OneCycleLR(
            optimizer_D,
            max_lr=lr,
            epochs=num_epochs,
            steps_per_epoch=len(source_loader),
            pct_start=0.3,
            div_factor=25.0,
            final_div_factor=10000.0
        )
        
        # 혼합 정밀도 스케일러
        scaler = torch.cuda.amp.GradScaler() if use_mixed_precision else None
        
        # 손실 기록
        losses = defaultdict(list)
        
        # 학습 루프
        for epoch in range(start_epoch, num_epochs):
            # 배치 반복
            for i, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
                # 실제 배치 크기 (마지막 배치는 크기가 다를 수 있음)
                real_batch_size = min(source_batch.size(0), target_batch.size(0))
                
                # 배치 크기가 다른 경우 작은 쪽에 맞춤
                if source_batch.size(0) != target_batch.size(0):
                    source_batch = source_batch[:real_batch_size]
                    target_batch = target_batch[:real_batch_size]
                
                # 데이터를 디바이스로 이동
                real_X = source_batch.to(device, non_blocking=True)  # 비동기 전송
                real_Y = target_batch.to(device, non_blocking=True)
                
                #--------------------
                # Generator 학습
                #--------------------
                optimizer_G.zero_grad(set_to_none=True)  # 메모리 최적화
                
                # 혼합 정밀도 사용 여부에 따라 다른 방식으로 학습
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        # identity 손실
                        identity_Y = self.G(real_Y)
                        identity_X = self.F(real_X)
                        loss_identity_Y = self.criterion_identity(identity_Y, real_Y) * lambda_identity
                        loss_identity_X = self.criterion_identity(identity_X, real_X) * lambda_identity
                        
                        # GAN 손실
                        fake_Y = self.G(real_X)
                        pred_fake_Y = self.D_Y(fake_Y)
                        
                        fake_X = self.F(real_Y)
                        pred_fake_X = self.D_X(fake_X)
                        
                        # 실제/가짜 레이블 생성
                        real_label = torch.ones_like(pred_fake_Y, device=device) 
                        fake_label = torch.zeros_like(pred_fake_Y, device=device)
                        
                        loss_GAN_G = self.criterion_GAN(pred_fake_Y, real_label)
                        loss_GAN_F = self.criterion_GAN(pred_fake_X, real_label)
                        
                        # 사이클 손실
                        recovered_X = self.F(fake_Y)
                        loss_cycle_X = self.criterion_cycle(recovered_X, real_X) * lambda_cycle
                        
                        recovered_Y = self.G(fake_X)
                        loss_cycle_Y = self.criterion_cycle(recovered_Y, real_Y) * lambda_cycle
                        
                        # 전체 Generator 손실
                        loss_G = loss_identity_Y + loss_identity_X + loss_GAN_G + loss_GAN_F + loss_cycle_X + loss_cycle_Y
                    
                    # 역전파 및 최적화 (그라디언트 누적)
                    scaler.scale(loss_G).backward()
                    if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(source_loader):
                        scaler.step(optimizer_G)
                        scaler.update()
                        scheduler_G.step()
                else:
                    # 아이덴티티 손실
                    identity_Y = self.G(real_Y)
                    identity_X = self.F(real_X)
                    loss_identity_Y = self.criterion_identity(identity_Y, real_Y) * lambda_identity
                    loss_identity_X = self.criterion_identity(identity_X, real_X) * lambda_identity
                    
                    # GAN 손실
                    fake_Y = self.G(real_X)
                    pred_fake_Y = self.D_Y(fake_Y)
                    
                    fake_X = self.F(real_Y)
                    pred_fake_X = self.D_X(fake_X)
                    
                    # 실제/가짜 레이블 생성
                    real_label = torch.ones_like(pred_fake_Y, device=device) 
                    fake_label = torch.zeros_like(pred_fake_Y, device=device)
                    
                    loss_GAN_G = self.criterion_GAN(pred_fake_Y, real_label)
                    loss_GAN_F = self.criterion_GAN(pred_fake_X, real_label)
                    
                    # 사이클 손실
                    recovered_X = self.F(fake_Y)
                    loss_cycle_X = self.criterion_cycle(recovered_X, real_X) * lambda_cycle
                    
                    recovered_Y = self.G(fake_X)
                    loss_cycle_Y = self.criterion_cycle(recovered_Y, real_Y) * lambda_cycle
                    
                    # 전체 Generator 손실
                    loss_G = loss_identity_Y + loss_identity_X + loss_GAN_G + loss_GAN_F + loss_cycle_X + loss_cycle_Y
                    
                    # 역전파 및 최적화 (그라디언트 누적)
                    loss_G.backward()
                    if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(source_loader):
                        optimizer_G.step()
                        scheduler_G.step()
                
                #--------------------
                # Discriminator 학습
                #--------------------
                optimizer_D.zero_grad(set_to_none=True)  # 메모리 최적화
                
                # 혼합 정밀도 사용 여부에 따라 다른 방식으로 학습
                if use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        # D_X 손실
                        pred_real_X = self.D_X(real_X)
                        loss_D_real_X = self.criterion_GAN(pred_real_X, real_label)
                        
                        # 이전에 생성된 가짜 샘플 사용
                        pred_fake_X = self.D_X(fake_X.detach())
                        loss_D_fake_X = self.criterion_GAN(pred_fake_X, fake_label)
                        
                        loss_D_X = (loss_D_real_X + loss_D_fake_X) * 0.5
                        
                        # D_Y 손실
                        pred_real_Y = self.D_Y(real_Y)
                        loss_D_real_Y = self.criterion_GAN(pred_real_Y, real_label)
                        
                        # 이전에 생성된 가짜 샘플 사용
                        pred_fake_Y = self.D_Y(fake_Y.detach())
                        loss_D_fake_Y = self.criterion_GAN(pred_fake_Y, fake_label)
                        
                        loss_D_Y = (loss_D_real_Y + loss_D_fake_Y) * 0.5
                        
                        # 전체 Discriminator 손실
                        loss_D = loss_D_X + loss_D_Y
                    
                    # 역전파 및 최적화 (그라디언트 누적)
                    scaler.scale(loss_D).backward()
                    if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(source_loader):
                        scaler.step(optimizer_D)
                        scaler.update()
                        scheduler_D.step()
                else:
                    # D_X 손실
                    pred_real_X = self.D_X(real_X)
                    loss_D_real_X = self.criterion_GAN(pred_real_X, real_label)
                    
                    # 이전에 생성된 가짜 샘플 사용
                    pred_fake_X = self.D_X(fake_X.detach())
                    loss_D_fake_X = self.criterion_GAN(pred_fake_X, fake_label)
                    
                    loss_D_X = (loss_D_real_X + loss_D_fake_X) * 0.5
                    
                    # D_Y 손실
                    pred_real_Y = self.D_Y(real_Y)
                    loss_D_real_Y = self.criterion_GAN(pred_real_Y, real_label)
                    
                    # 이전에 생성된 가짜 샘플 사용
                    pred_fake_Y = self.D_Y(fake_Y.detach())
                    loss_D_fake_Y = self.criterion_GAN(pred_fake_Y, fake_label)
                    
                    loss_D_Y = (loss_D_real_Y + loss_D_fake_Y) * 0.5
                    
                    # 전체 Discriminator 손실
                    loss_D = loss_D_X + loss_D_Y
                    
                    # 역전파 및 최적화 (그라디언트 누적)
                    loss_D.backward()
                    if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(source_loader):
                        optimizer_D.step()
                        scheduler_D.step()
                
                # 메모리 캐시 관리 (일정 빈도로)
                if (i + 1) % clear_cache_freq == 0:
                    del real_X, real_Y, fake_X, fake_Y, recovered_X, recovered_Y
                    torch.cuda.empty_cache()
                
                # 손실 기록
                losses['G'].append(loss_G.item())
                losses['D'].append(loss_D.item())
                losses['G_identity'].append((loss_identity_X.item() + loss_identity_Y.item()))
                losses['G_GAN'].append((loss_GAN_G.item() + loss_GAN_F.item()))
                losses['G_cycle'].append((loss_cycle_X.item() + loss_cycle_Y.item()))
                
                # 진행 상황 출력 (10배치마다)
                if (i + 1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{min(len(source_loader), len(target_loader))}], "
                               f"Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}, "
                               f"Loss Cycle: {(loss_cycle_X.item() + loss_cycle_Y.item()):.4f}")
            
            # 체크포인트 저장 (일정 간격마다)
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == num_epochs:
                self.save_checkpoint(epoch + 1, optimizer_G, optimizer_D, scheduler_G, scheduler_D)
                self.save_models(epoch + 1)
                
                # 샘플 이미지 생성 및 저장
                self._generate_and_save_samples(data_dir, epoch + 1)
        
        # 학습 곡선 시각화
        self._plot_training_curves(losses)
        
        return losses
    
    def save_models(self, epoch):
        """
        모델 저장
        
        Args:
            epoch (int): 현재 에폭
        """
        torch.save(self.G.state_dict(), self.model_dir / f"G_{epoch}.pth")
        torch.save(self.F.state_dict(), self.model_dir / f"F_{epoch}.pth")
        torch.save(self.D_X.state_dict(), self.model_dir / f"D_X_{epoch}.pth")
        torch.save(self.D_Y.state_dict(), self.model_dir / f"D_Y_{epoch}.pth")
        
        logger.info(f"모델 저장 완료: 에폭 {epoch}")
    
    def load_models(self, epoch):
        """
        모델 로드
        
        Args:
            epoch (int): 로드할 에폭
        """
        self.G.load_state_dict(torch.load(self.model_dir / f"G_{epoch}.pth"))
        self.F.load_state_dict(torch.load(self.model_dir / f"F_{epoch}.pth"))
        self.D_X.load_state_dict(torch.load(self.model_dir / f"D_X_{epoch}.pth"))
        self.D_Y.load_state_dict(torch.load(self.model_dir / f"D_Y_{epoch}.pth"))
        
        logger.info(f"모델 로드 완료: 에폭 {epoch}")
    
    def transform_dataset(self, data_dir, output_dir, split='test'):
        """
        데이터셋 변환
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            output_dir (str): 출력 디렉토리 경로
            split (str): 데이터셋 분할 ('train', 'val', 'test')
            
        Returns:
            dict: 변환 결과 정보
        """
        # 출력 디렉토리 생성
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # 데이터셋 로드
        dataset = CycleGANDataset(
            data_dir=data_dir,
            hospital_ids=[self.source_id, self.target_id],
            split=split,
            target_shape=self.input_shape
        )
        
        # 데이터 로더 생성
        source_loader, _ = dataset.get_hospital_pair_loader(
            source_id=self.source_id,
            target_id=self.target_id,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        
        # 원본 데이터 정보 로드
        data_info = pd.read_csv(Path(data_dir) / f'{split}_dataset.csv')
        source_data = data_info[data_info['hospital_id'] == self.source_id]
        
        # 변환 결과 정보
        transformed_info = []
        
        # 평가 모드
        self.G.eval()
        
        # 변환 수행
        with torch.no_grad():
            for i, source_batch in enumerate(tqdm(source_loader, desc=f"변환 중 ({self.source_id} -> {self.target_id})")):
                # 데이터를 디바이스로 이동
                real_X = source_batch.to(device)
                
                # 변환
                fake_Y = self.G(real_X)
                
                # Tanh 출력값을 [0, 1] 범위로 변환
                fake_Y = (fake_Y + 1) / 2
                
                # 원본 이미지의 밝기 범위로 스케일링
                # 원본 이미지의 최소/최대값 계산
                original_min = real_X.min().item()
                original_max = real_X.max().item()
                
                # 스케일링 적용
                fake_Y = fake_Y * (original_max - original_min) + original_min
                
                # CPU로 이동 및 NumPy 배열로 변환
                fake_Y_np = fake_Y.cpu().numpy()[0, 0]  # (B, C, H, W, D) -> (H, W, D)
                
                # 원본 파일 정보
                original_path = source_data.iloc[i]['file_path']
                original_file = Path(original_path)
                
                # 원본 이미지 로드 (메타데이터 유지를 위해)
                original_img = ants.image_read(original_path)
                
                # 변환된 이미지 생성
                transformed_img = ants.from_numpy(fake_Y_np)
                transformed_img.set_spacing(original_img.spacing)
                transformed_img.set_origin(original_img.origin)
                transformed_img.set_direction(original_img.direction)
                
                # 파일명 생성
                filename = f"{self.source_id}_to_{self.target_id}_{original_file.stem}.nii.gz"
                output_path = output_dir / filename
                
                # 변환된 이미지 저장
                ants.image_write(transformed_img, str(output_path))
                
                # 결과 정보 저장
                info = source_data.iloc[i].to_dict()
                info['original_path'] = original_path
                info['transformed_path'] = str(output_path)
                info['source_id'] = self.source_id
                info['target_id'] = self.target_id
                transformed_info.append(info)
        
        # 결과 정보 저장
        transformed_df = pd.DataFrame(transformed_info)
        transformed_df.to_csv(output_dir / f"{self.source_id}_to_{self.target_id}_transformed.csv", index=False)
        
        logger.info(f"데이터셋 변환 완료: {len(transformed_info)}개 이미지")
        
        return {
            'count': len(transformed_info),
            'info': transformed_df
        }
    
    def _generate_and_save_samples(self, data_dir, epoch):
        """
        샘플 이미지 생성 및 저장
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            epoch (int): 현재 에폭
        """
        # 데이터셋 로드
        dataset = CycleGANDataset(
            data_dir=data_dir,
            hospital_ids=[self.source_id, self.target_id],
            split='val',
            target_shape=self.input_shape
        )
        
        # 데이터 로더 생성
        source_loader, target_loader = dataset.get_hospital_pair_loader(
            source_id=self.source_id,
            target_id=self.target_id,
            batch_size=1,
            shuffle=False,
            num_workers=1
        )
        
        # 샘플 디렉토리 생성
        sample_dir = self.model_dir / 'samples'
        os.makedirs(sample_dir, exist_ok=True)
        
        # 평가 모드
        self.G.eval()
        self.F.eval()
        
        # 샘플 생성
        with torch.no_grad():
            # 최대 5개 샘플만 생성
            for i, (source_batch, target_batch) in enumerate(zip(source_loader, target_loader)):
                if i >= 5:
                    break
                
                # 데이터를 디바이스로 이동
                real_X = source_batch.to(device)
                real_Y = target_batch.to(device)
                
                # 변환
                fake_Y = self.G(real_X)
                fake_X = self.F(real_Y)
                
                # 사이클 변환
                recovered_X = self.F(fake_Y)
                recovered_Y = self.G(fake_X)
                
                # 중앙 슬라이스 추출 및 시각화
                self._save_sample_slices(real_X, fake_Y, recovered_X, real_Y, fake_X, recovered_Y, sample_dir, epoch, i)
    
    def _save_sample_slices(self, real_X, fake_Y, recovered_X, real_Y, fake_X, recovered_Y, sample_dir, epoch, sample_idx):
        """
        샘플 슬라이스 저장
        
        Args:
            real_X (torch.Tensor): 실제 X 이미지
            fake_Y (torch.Tensor): 가짜 Y 이미지
            recovered_X (torch.Tensor): 복원된 X 이미지
            real_Y (torch.Tensor): 실제 Y 이미지
            fake_X (torch.Tensor): 가짜 X 이미지
            recovered_Y (torch.Tensor): 복원된 Y 이미지
            sample_dir (Path): 샘플 디렉토리 경로
            epoch (int): 현재 에폭
            sample_idx (int): 샘플 인덱스
        """
        # CPU로 이동 및 NumPy 배열로 변환
        real_X_np = real_X.cpu().numpy()[0, 0]
        fake_Y_np = fake_Y.cpu().numpy()[0, 0]
        recovered_X_np = recovered_X.cpu().numpy()[0, 0]
        real_Y_np = real_Y.cpu().numpy()[0, 0]
        fake_X_np = fake_X.cpu().numpy()[0, 0]
        recovered_Y_np = recovered_Y.cpu().numpy()[0, 0]
        
        # 중앙 슬라이스 인덱스
        z_idx = real_X_np.shape[2] // 2
        
        # 슬라이스 추출
        real_X_slice = real_X_np[:, :, z_idx]
        fake_Y_slice = fake_Y_np[:, :, z_idx]
        recovered_X_slice = recovered_X_np[:, :, z_idx]
        real_Y_slice = real_Y_np[:, :, z_idx]
        fake_X_slice = fake_X_np[:, :, z_idx]
        recovered_Y_slice = recovered_Y_np[:, :, z_idx]
        
        # 시각화
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 첫 번째 행: X -> Y -> X
        axes[0, 0].imshow(real_X_slice, cmap='gray')
        axes[0, 0].set_title(f'Real {self.source_id}')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(fake_Y_slice, cmap='gray')
        axes[0, 1].set_title(f'Fake {self.target_id}')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(recovered_X_slice, cmap='gray')
        axes[0, 2].set_title(f'Recovered {self.source_id}')
        axes[0, 2].axis('off')
        
        # 두 번째 행: Y -> X -> Y
        axes[1, 0].imshow(real_Y_slice, cmap='gray')
        axes[1, 0].set_title(f'Real {self.target_id}')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(fake_X_slice, cmap='gray')
        axes[1, 1].set_title(f'Fake {self.source_id}')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(recovered_Y_slice, cmap='gray')
        axes[1, 2].set_title(f'Recovered {self.target_id}')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(sample_dir / f"sample_{epoch}_{sample_idx}.png")
        plt.close()
    
    def _plot_training_curves(self, losses):
        """
        학습 곡선 시각화
        
        Args:
            losses (dict): 손실 기록
        """
        # 손실 평균 계산 (배치 단위 -> 에폭 단위)
        batch_per_epoch = len(losses['G']) // len(losses['G_cycle'])
        epoch_losses = defaultdict(list)
        
        for key in losses:
            for i in range(0, len(losses[key]), batch_per_epoch):
                epoch_losses[key].append(np.mean(losses[key][i:i+batch_per_epoch]))
        
        epochs = range(1, len(epoch_losses['G']) + 1)
        
        # 손실 곡선
        plt.figure(figsize=(15, 10))
        
        # Generator 손실
        plt.subplot(2, 1, 1)
        plt.plot(epochs, epoch_losses['G'], 'b-', label='Generator Loss')
        plt.plot(epochs, epoch_losses['G_identity'], 'r-', label='Identity Loss')
        plt.plot(epochs, epoch_losses['G_GAN'], 'g-', label='GAN Loss')
        plt.plot(epochs, epoch_losses['G_cycle'], 'y-', label='Cycle Loss')
        plt.title('Generator Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Discriminator 손실
        plt.subplot(2, 1, 2)
        plt.plot(epochs, epoch_losses['D'], 'b-', label='Discriminator Loss')
        plt.title('Discriminator Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_curves.png')
        plt.close()

class TrainingStatus:
    """학습 상태 관리 클래스"""
    
    def __init__(self, status_file):
        """
        초기화 함수
        
        Args:
            status_file (str): 상태 파일 경로
        """
        self.status_file = Path(status_file)
        self.status = self._load_status()
    
    def _load_status(self):
        """상태 파일 로드"""
        if self.status_file.exists():
            with open(self.status_file, 'r') as f:
                return json.load(f)
        else:
            # 기본 상태 설정
            return {
                'current_source': 'EUMC',
                'current_target': 'KNUCH',
                'current_epoch': 0,
                'is_completed': False
            }
    
    def save_status(self):
        """상태 파일 저장"""
        with open(self.status_file, 'w') as f:
            json.dump(self.status, f, indent=4)
    
    def update_status(self, source_id, target_id, epoch, is_completed=False):
        """상태 업데이트"""
        self.status['current_source'] = source_id
        self.status['current_target'] = target_id
        self.status['current_epoch'] = epoch
        self.status['is_completed'] = is_completed
        self.save_status()
    
    def get_current_task(self):
        """현재 작업 정보 반환"""
        return (
            self.status['current_source'],
            self.status['current_target'],
            self.status['current_epoch']
        )
    
    def is_training_completed(self):
        """학습 완료 여부 확인"""
        return self.status['is_completed']

def train_all_cyclegan_pairs(data_dir, model_dir, hospital_ids, num_epochs=50, batch_size=1):
    """
    모든 병원 쌍에 대해 CycleGAN 학습
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        model_dir (str): 모델 저장 디렉토리 경로
        hospital_ids (list): 병원 ID 목록
        num_epochs (int): 에폭 수
        batch_size (int): 배치 크기
        
    Returns:
        dict: 학습 결과
    """
    results = {}
    
    status_manager = TrainingStatus(Path(model_dir) / 'train_status.json')
    current_source, current_target, current_epoch = status_manager.get_current_task()
    
    # 모든 병원 쌍에 대해 반복
    for i, source_id in enumerate(hospital_ids):
        for j, target_id in enumerate(hospital_ids):
            if i == j:  
                continue
            
            # 현재 작업보다 이전 작업은 건너뛰기
            if (source_id, target_id) < (current_source, current_target):
                continue
            
            logger.info(f"CycleGAN 학습 시작: {source_id} -> {target_id}")
            
            # CycleGAN 모델 생성
            cyclegan = CycleGAN3D(source_id, target_id, model_dir)
            
            # 모델 학습
            losses = cyclegan.train(
                data_dir=data_dir,
                batch_size=batch_size,
                num_epochs=num_epochs,
                start_epoch=current_epoch if (source_id, target_id) == (current_source, current_target) else 0
            )
            
            # 결과 저장
            results[f"{source_id}_to_{target_id}"] = {
                'losses': losses
            }
            
            # 다음 병원 쌍으로 상태 업데이트
            next_source, next_target = _get_next_pair(hospital_ids, source_id, target_id)
            if next_source and next_target:
                status_manager.update_status(next_source, next_target, 0)
            else:
                # 모든 학습 완료
                status_manager.update_status(source_id, target_id, num_epochs, is_completed=True)
    
    return results

def _get_next_pair(hospital_ids, current_source, current_target):
    """다음 병원 쌍 반환"""
    for i, source_id in enumerate(hospital_ids):
        for j, target_id in enumerate(hospital_ids):
            if i == j:
                continue
            if (source_id, target_id) > (current_source, current_target):
                return source_id, target_id
    return None, None

def transform_all_datasets(data_dir, model_dir, output_dir, hospital_ids, epoch=50):
    """
    모든 병원 쌍에 대해 데이터셋 변환
    
    Args:
        data_dir (str): 데이터 디렉토리 경로
        model_dir (str): 모델 디렉토리 경로
        output_dir (str): 출력 디렉토리 경로
        hospital_ids (list): 병원 ID 목록
        epoch (int): 로드할 에폭
        
    Returns:
        dict: 변환 결과
    """
    results = {}
    
    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)
    
    # 모든 병원 쌍에 대해 반복
    for i, source_id in enumerate(hospital_ids):
        for j, target_id in enumerate(hospital_ids):
            if i == j:  # 같은 병원은 건너뜀
                continue
            
            logger.info(f"데이터셋 변환 시작: {source_id} -> {target_id}")
            
            # CycleGAN 모델 생성
            cyclegan = CycleGAN3D(source_id, target_id, model_dir)
            
            try:
                # 모델 로드
                cyclegan.load_models(epoch)
                
                # 데이터셋 변환
                transform_result = cyclegan.transform_dataset(
                    data_dir=data_dir,
                    output_dir=output_dir
                )
                
                # 결과 저장
                results[f"{source_id}_to_{target_id}"] = transform_result
            except Exception as e:
                logger.error(f"데이터셋 변환 실패: {source_id} -> {target_id}, 오류: {str(e)}")
    
    return results

def main():
    """메인 함수"""
    data_dir = '/public/sylee/MDAISS/output'
    model_dir = '/public/sylee/MDAISS/models/cyclegan'
    output_dir = '/public/sylee/MDAISS/output/transformed'
    
    # 병원 ID 설정 
    hospital_ids = ['EUMC', 'KNUCH', 'KUAH', 'SMC']

    # 모든 병원 쌍에 대해 CycleGAN 학습
    train_results = train_all_cyclegan_pairs(
        data_dir=data_dir,
        model_dir=model_dir,
        hospital_ids=hospital_ids,
        num_epochs=100,
        batch_size=1
    )

    # 모든 병원 쌍에 대해 데이터셋 변환
    transform_results = transform_all_datasets(
        data_dir=data_dir,
        model_dir=model_dir,
        output_dir=output_dir,
        hospital_ids=hospital_ids,
        epoch=90
    )

    logger.info("CycleGAN 학습 및 데이터셋 변환 완료")

if __name__ == '__main__':
    main()
