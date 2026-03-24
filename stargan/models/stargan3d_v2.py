#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
StarGAN-3D 모델 구현 (v2 - Content/Style 분리)

이 모듈은 3D 의료 영상 데이터에 대한 StarGAN-3D 모델의 개선된 버전을 구현합니다.
Content Encoder, Style Encoder, 학습 가능한 Neutral Style Vector 등을 통합하여
병원 간 이미지 스타일 변환 및 도메인 정규화를 수행합니다.

주요 기능:
1. Content Encoder, Style Encoder, Generator, Discriminator 구현
2. 학습 가능한 Neutral Style Vector 정의
3. AdaIN 기반 스타일 주입
4. MMD/CORAL, GRL 등 추가 손실 함수 통합
5. 개선된 학습 파이프라인 구현
6. 이미지 변환 및 평가
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
from torch.autograd import Function
import shutil
import wandb

# matplotlib 백엔드 설정 추가
import matplotlib
matplotlib.use("Agg")  # "Agg" 백엔드는 GUI를 사용하지 않음

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- 보조 함수 및 모듈 ---

def calc_mean_std(feat, eps=1e-5):
    """AdaIN을 위한 평균과 표준편차 계산 함수.
    인자:
        feat (Tensor): [N, C, D, H, W] 형태의 텐서.
        eps (float): 0으로 나누는 것을 방지하기 위한 작은 값.
    반환:
        tuple: (평균, 표준편차) [N, C, 1, 1, 1] 형태.
    """
    size = feat.size()
    assert len(size) == 5
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1, 1)
    return feat_mean, feat_std

class AdaIN3d(nn.Module):
    """3D 적응 인스턴스 정규화 레이어."""
    def __init__(self):
        super().__init__()

    def forward(self, content_feat, style_feat):
        """순전파.
        인자:
            content_feat (Tensor): [N, C, D, H, W] 형태의 콘텐츠 특징.
            style_feat (Tensor): [N, C, D, H, W] 형태의 스타일 특징.
        반환:
            Tensor: 스타일이 적용된 정규화된 콘텐츠 특징.
        """
        assert content_feat.size()[:2] == style_feat.size()[:2]
        size = content_feat.size()
        style_mean, style_std = calc_mean_std(style_feat)
        content_mean, content_std = calc_mean_std(content_feat)

        normalized_feat = (content_feat - content_mean) / content_std
        return normalized_feat * style_std + style_mean

class GradientReversalFunction(Function):
    """Gradient Reversal Layer 함수.
    출처: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py
    """
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_val
        return output, None

class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer 모듈."""
    def __init__(self, lambda_val=1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_val = lambda_val

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_val)
    
'''
def MMDLoss(x, y, kernel="rbf", sigma=1.0):
    """최대 평균 차이(MMD) 손실 함수.
    인자:
        x (Tensor): 첫 번째 샘플 배치.
        y (Tensor): 두 번째 샘플 배치.
        kernel (str): 커널 타입 ("rbf" 또는 "linear").
        sigma (float): RBF 커널을 위한 시그마.
    반환:
        Tensor: MMD 손실.
    """
    x = x.view(x.size(0), -1)
    y = y.view(y.size(0), -1)
    
    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    
    if kernel == "linear":
        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))
        K = xx - 0.5 * (rx.t() + rx)  # K(x, x)
        L = yy - 0.5 * (ry.t() + ry)  # K(y, y)
        P = xy - 0.5 * (rx.t() + ry)  # K(x, y)
    elif kernel == "rbf":
        # 가우시안 커널
        gamma = 1.0 / (2 * sigma**2)
        XX = torch.sum(x*x, 1, keepdim=True)
        YY = torch.sum(y*y, 1, keepdim=True)
        XY = torch.mm(x, y.t())
        
        dist_xx = XX + XX.t() - 2*xx
        dist_yy = YY + YY.t() - 2*yy
        dist_xy = XX + YY.t() - 2*xy
        
        K = torch.exp(-gamma * dist_xx)
        L = torch.exp(-gamma * dist_yy)
        P = torch.exp(-gamma * dist_xy)
    else:
        raise ValueError("지원하지 않는 커널 타입")
        
    beta = (1. / (x.size(0) * (x.size(0) - 1)))
    gamma = (2. / (x.size(0) * y.size(0)))
    delta = (1. / (y.size(0) * (y.size(0) - 1)))
    
    # MMD^2 추정
    loss = beta * (torch.sum(K) - K.diag().sum()) + delta * (torch.sum(L) - L.diag().sum()) - gamma * torch.sum(P)
    
    return loss
'''

# --- 데이터셋  ---
class MedicalVolumeDataset(Dataset):
    """3D 의료 영상 데이터셋 클래스 (StarGAN-3D용)"""
    
    def __init__(self, csv_file, hospital_id=None, transform=None, target_shape=(128, 128, 128)):
        """
        초기화 함수.
        """
        self.data_info = pd.read_csv(csv_file)
        self.hospital_id = hospital_id
        self.transform = transform
        self.target_shape = target_shape
        
        # 병원 ID를 숫자 레이블로 변환 
        self.unique_hospital_ids = sorted(self.data_info["hospital_id"].unique())
        self.num_domains = len(self.unique_hospital_ids)  
        self.hospital_to_label = {h_id: i for i, h_id in enumerate(self.unique_hospital_ids)}
        self.label_to_hospital = {i: h_id for i, h_id in enumerate(self.unique_hospital_ids)}
        self.hospital_domain_label = 1  
        # logger.info(f"데이터셋 로드: {len(self.data_info)} 샘플, {self.num_domains-1} 병원")
        # logger.info(f"병원 ID 매핑: {self.hospital_to_label}")
        
        if hospital_id is not None:
            self.data_info = self.data_info[self.data_info["hospital_id"] == hospital_id]
        
        if len(self.data_info) == 0 and hospital_id is not None:
            raise ValueError(f"병원 ID {hospital_id}에 해당하는 데이터가 없습니다.")

    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.data_info.iloc[idx]["file_path"]
        hospital_id = self.data_info.iloc[idx]["hospital_id"]
        label = self.hospital_to_label[hospital_id]
        
        img = ants.image_read(img_path)
        img = self._resample_volume(img)
        img_array = img.numpy()
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)

        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, torch.tensor(label, dtype=torch.long)
    
    def _resample_volume(self, img):
        """
        볼륨 리샘플링.
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
        이미지 크롭 또는 패딩.
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

# --- 네트워크 블록 ---
class Conv3DBlock(nn.Module):
    """3D 컨볼루션 블록."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batch_norm=True, use_leaky=False, activation=True):
        """
        초기화 함수.
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
        
        if activation:
            if use_leaky:
                self.activation = nn.LeakyReLU(0.2, inplace=True)
            else:
                self.activation = nn.ReLU(inplace=True)
        else:
            self.activation = None
    
    def forward(self, x):
        """순전파."""
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class TransConv3DBlock(nn.Module):
    """3D 전치 컨볼루션 블록."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1, use_batch_norm=True):
        """
        초기화 함수.
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
        """순전파."""
        x = self.transconv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock3D(nn.Module):
    """3D 잔차 블록."""
    
    def __init__(self, channels):
        """
        초기화 함수.
        """
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm3d(channels, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.InstanceNorm3d(channels, affine=True)
    
    def forward(self, x):
        """순전파."""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        # 최종 ReLU 적용 없음 (GAN의 ResNet 블록에서 일반적)
        return out

# --- 네트워크 구조 ---
class UNet3DEncoder(nn.Module):
    """3D U-Net 인코더 (다운샘플링)"""
    def __init__(self, in_channels=1, base_filters=16, num_levels=4):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.out_channels = []
        channels = in_channels
        for i in range(num_levels):
            out_channels = base_filters * (2 ** i)
            self.enc_blocks.append(
                nn.Sequential(
                    Conv3DBlock(channels, out_channels, kernel_size=3, padding=1),
                    Conv3DBlock(out_channels, out_channels, kernel_size=3, padding=1)
                )
            )
            self.pools.append(nn.MaxPool3d(2))
            self.out_channels.append(out_channels)
            channels = out_channels
        self.bottleneck_channels = channels

    def forward(self, x):
        features = []
        for enc, pool in zip(self.enc_blocks, self.pools):
            x = enc(x)
            features.append(x)
            x = pool(x)
        return x, features

class UNet3DDecoder(nn.Module):
    """3D U-Net 디코더 (업샘플링)"""
    def __init__(self, out_channels=1, base_filters=16, num_levels=4, style_dim=64):
        super().__init__()
        self.num_levels = num_levels
        # Encoder에서 사용된 채널 정보 필요
        self.enc_channels = [base_filters * (2 ** i) for i in range(num_levels)]
        self.bottleneck_channels = self.enc_channels[-1]
        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.adain_layers = nn.ModuleList()
        self.style_to_adain = nn.ModuleList()
        for i in reversed(range(num_levels)):
            skip_channels = self.enc_channels[i]
            in_ch = self.bottleneck_channels if i == num_levels - 1 else out_ch
            upconv = nn.ConvTranspose3d(in_ch, skip_channels, kernel_size=2, stride=2)
            self.upconvs.append(upconv)
            dec_in_ch = skip_channels * 2  
            out_ch = skip_channels
            self.dec_blocks.append(
                nn.Sequential(
                    Conv3DBlock(dec_in_ch, out_ch, kernel_size=3, padding=1),
                    Conv3DBlock(out_ch, out_ch, kernel_size=3, padding=1)
                )
            )
            self.adain_layers.append(AdaIN3d())
            self.style_to_adain.append(nn.Linear(style_dim, out_ch * 2))
        self.final_conv = nn.Conv3d(base_filters, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x, encoder_features, style_code):
        for i, (upconv, dec_block, adain, style_fc) in enumerate(zip(
            self.upconvs, self.dec_blocks, self.adain_layers, self.style_to_adain
        )):
            x = upconv(x)
            skip = encoder_features[-(i + 1)]
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)  # conv로 채널 수를 out_ch로 맞춤
            # AdaIN 파라미터 생성
            adain_params = style_fc(style_code)
            gamma, beta = torch.chunk(adain_params, 2, dim=1)
            gamma = gamma.view(x.size(0), -1, 1, 1, 1)
            beta = beta.view(x.size(0), -1, 1, 1, 1)
            mean, std = calc_mean_std(x)
            x = (x - mean) / std
            x = gamma * x + beta
        x = self.final_conv(x)
        x = self.tanh(x)
        return x

# ContentEncoder를 U-Net encoder로 대체
class ContentEncoder(nn.Module):
    """U-Net 기반 Content Encoder"""
    def __init__(self, input_channels=1, base_filters=16, num_levels=4):
        super().__init__()
        self.encoder = UNet3DEncoder(input_channels, base_filters, num_levels)
    def forward(self, x):
        z, features = self.encoder(x)
        return z, features

# Generator를 U-Net decoder로 대체
class Generator(nn.Module):
    """U-Net 기반 Generator (AdaIN 포함)"""
    def __init__(self, content_dim=256, style_dim=64, output_channels=1, base_filters=16, num_levels=4):
        super().__init__()
        self.decoder = UNet3DDecoder(output_channels, base_filters, num_levels, style_dim)
    def forward(self, content_z_and_features, style_code):
        z, features = content_z_and_features
        out = self.decoder(z, features, style_code)
        return out

class Discriminator(nn.Module):
    """Discriminator (D) - PatchGAN 및 도메인 분류 """
    
    def __init__(self, input_channels=1, base_filters=32, domain_dim=4):
        super().__init__()
        self.domain_dim = domain_dim
        
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
                    # GAN에서는 BatchNorm 대신 InstanceNorm을 사용하는 것이 더 좋을 수 있음
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        
        # 컨볼루션 레이어에서 출력된 특징 맵
        self.features = nn.Sequential(*self.layers)
        feature_dim = in_channels # 헤드 전의 특징 맵 차원
        
        # GAN 헤드 (실제/가짜 판별)
        self.gan_head = nn.Conv3d(feature_dim, 1, kernel_size=4, stride=1, padding=1)
        
        # 도메인 분류 헤드
        # FC 레이어 전에 다양한 공간 크기를 처리하기 위해 AdaptiveAvgPool 사용
        self.domain_pool = nn.AdaptiveAvgPool3d(1)
        self.domain_fc = nn.Linear(feature_dim, domain_dim)
        # self.domain_head = nn.Conv3d(feature_dim, domain_dim, kernel_size=4, stride=1, padding=1)
    
    def forward(self, x):
        """순전파"""
        x = self.initial(x)
        features = self.features(x)
        
        # GAN 출력
        gan_out = self.gan_head(features)
        
        # 도메인 분류 출력
        domain_pooled = self.domain_pool(features).view(x.size(0), -1)
        domain_out = self.domain_fc(domain_pooled)
        # domain_out = self.domain_head(features).view(x.size(0), -1) 
        
        return gan_out, domain_out

class StyleEncoder(nn.Module):
    """Style Encoder (E_s)."""
    def __init__(self, input_channels=1, base_filters=16, style_dim=64):
        super().__init__()
        layers = []
        layers.append(Conv3DBlock(input_channels, base_filters, kernel_size=7, padding=3, use_leaky=True)) # 128
        layers.append(Conv3DBlock(base_filters, base_filters * 2, stride=2, use_leaky=True)) # 64
        layers.append(Conv3DBlock(base_filters * 2, base_filters * 4, stride=2, use_leaky=True)) # 32
        layers.append(Conv3DBlock(base_filters * 4, base_filters * 8, stride=2, use_leaky=True)) # 16
        layers.append(Conv3DBlock(base_filters * 8, base_filters * 8, stride=2, use_leaky=True)) # 8
        layers.append(nn.AdaptiveAvgPool3d(1))
        layers.append(nn.Conv3d(base_filters * 8, style_dim, kernel_size=1))
        self.encoder = nn.Sequential(*layers)
        
    def forward(self, x):
        style_code = self.encoder(x)
        return style_code.view(x.size(0), -1)  # [N, style_dim] 형태로 평탄화

# --- 메인 StarGAN-3D 클래스 ---
class StarGAN3Dv2(nn.Module):
    """StarGAN-3D  모델 클래스."""
    
    def __init__(self, model_dir, input_shape=(128, 128, 128), num_domains=4, style_dim=64, content_dim=256):
        super().__init__()
        self.model_dir = Path(model_dir)
        self.input_shape = input_shape
        self.num_domains = num_domains 
        self.style_dim = style_dim
        self.content_dim = content_dim 
        
        self.best_model_dir = self.model_dir / "best_model"
        self.checkpoint_dir = self.model_dir / "checkpoints"
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 네트워크 정의
        self.E_c = ContentEncoder(input_channels=1, base_filters=16, num_levels=4).to(device)
        self.E_s = StyleEncoder(style_dim=style_dim).to(device)
        self.G = Generator(content_dim=content_dim, style_dim=style_dim, output_channels=1, base_filters=16, num_levels=4).to(device)
        self.D = Discriminator(domain_dim=num_domains).to(device)  # Discriminator for real/fake and domain
        
        # ContentEncoder의 bottleneck 채널 수를 정확히 가져옴
        self.content_bottleneck_channels = self.E_c.encoder.bottleneck_channels
        
        # 학습 가능한 중립 스타일 벡터
        self.style_embeddings = nn.Embedding(self.num_domains, self.style_dim).to(device)
        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(lambda_val=1.0)
        # content_classifier 입력 채널을 bottleneck 채널로 맞춤
        self.content_classifier = nn.Sequential(
            nn.Conv3d(self.content_bottleneck_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(32, self.num_domains)
        ).to(device)
        
        # 손실 함수 정의
        self.criterion_GAN = nn.MSELoss() # LSGAN loss
        self.criterion_cyc = nn.L1Loss() # Cycle consistency
        self.criterion_sty = nn.L1Loss() # Style reconstruction
        self.criterion_con = nn.L1Loss() # Content reconstruction
        # self.criterion_idt = nn.L1Loss() # Identity mapping
        self.criterion_cls = nn.CrossEntropyLoss() # Domain classification
        # self.criterion_mmd = MMDLoss # MMD loss function
        
        logger.info("StarGAN-3D 모델 초기화 완료")

    def save_checkpoint(self, epoch, optimizers, schedulers):
        """체크포인트 저장."""
        checkpoint = {
            "epoch": epoch,
            "E_c_state_dict": self.E_c.state_dict(),
            "E_s_state_dict": self.E_s.state_dict(),
            "G_state_dict": self.G.state_dict(),
            "D_state_dict": self.D.state_dict(),
            "style_embeddings_state_dict": self.style_embeddings.state_dict(),
            "optimizer_G_state_dict": optimizers["G"].state_dict(),
            "optimizer_D_state_dict": optimizers["D"].state_dict(),
            "scheduler_G_state_dict": schedulers["G"].state_dict() if schedulers["G"] else None,
            "scheduler_D_state_dict": schedulers["D"].state_dict() if schedulers["D"] else None,
        }
        torch.save(checkpoint, self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth")
        # config_stargan.yaml도 같이 저장
        config_dst = self.model_dir / "config_stargan.yaml"
        config_src = Path("/public/sylee/stargan/scripts/config_stargan.yaml")
        shutil.copy(config_src, config_dst)
        # logger.info(f"체크포인트 저장 완료: 에폭 {epoch}")

    def save_best_model(self):
        """베스트 모델 저장."""
        torch.save(self.E_c.state_dict(), self.best_model_dir / "E_c_best.pth")
        torch.save(self.E_s.state_dict(), self.best_model_dir / "E_s_best.pth")
        torch.save(self.G.state_dict(), self.best_model_dir / "G_best.pth")
        torch.save(self.D.state_dict(), self.best_model_dir / "D_best.pth")
        torch.save(self.style_embeddings.state_dict(), self.best_model_dir / "style_embeddings_best.pth")
        # config_stargan.yaml도 같이 저장
        config_dst = self.best_model_dir / "config_stargan.yaml"
        config_src = Path("/public/sylee/stargan/scripts/config_stargan.yaml")
        shutil.copy(config_src, config_dst)
        logger.info("베스트 모델 저장 완료")

    def _get_lambda(self, epoch, max_val, warmup_epoch):
        """선형 warm-up 스케줄링 함수"""
        if epoch < warmup_epoch:
            return max_val * (epoch / warmup_epoch)
        return max_val

    def train(self, data_dir, batch_size=1, num_epochs=100,
              generator_lr=1e-4, discriminator_lr=4e-4,
              beta1=0.5, beta2=0.999,
              lambda_cyc_max=1.0, lambda_sty_max=5.0, lambda_con_max=5.0, lambda_cls=1.0, lambda_grl_max=0.5,
              warmup_cyc_epochs=10, warmup_sty_epochs=5, warmup_con_epochs=20, warmup_grl_epochs=10,
              two_stage_training=True,
              save_interval=5, grad_accum_steps=1, use_mixed_precision=False,
              use_warmup=True,
              use_wandb=False):  
        """
        StarGAN-3D 학습.
        """
        # 주요 설정값을 폴더명에 포함
        training_folder_name = (
            f"batch_{batch_size}_glr_{generator_lr}_dlr_{discriminator_lr}_epochs_{num_epochs}"
            f"_cyc{lambda_cyc_max}_sty{lambda_sty_max}_con{lambda_con_max}_cls{lambda_cls}_grl{lambda_grl_max}"
            f"_warmup{int(use_warmup)}_twostage{int(two_stage_training)}"
        )
        self.model_dir = self.model_dir / training_folder_name
        self.best_model_dir = self.model_dir / "best_model"
        self.checkpoint_dir = self.model_dir / "checkpoints"
        os.makedirs(self.best_model_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # --- WandB 연동 ---
        wandb_run = None
        if use_wandb:
            try:
                wandb_run = wandb.init(
                    project="stargan3d_v2",
                    name=str(self.model_dir.name),
                    config={
                        "batch_size": batch_size,
                        "generator_lr": generator_lr,
                        "discriminator_lr": discriminator_lr,
                        "epochs": num_epochs,
                        "lambda_cyc_max": lambda_cyc_max,
                        "lambda_sty_max": lambda_sty_max,
                        "lambda_con_max": lambda_con_max,
                        "lambda_cls": lambda_cls,
                        "lambda_grl_max": lambda_grl_max,
                        "warmup": use_warmup,
                        "two_stage_training": two_stage_training,
                        "use_mixed_precision": use_mixed_precision,
                    }
                )
            except ImportError:
                logger.warning("wandb 패키지가 설치되어 있지 않습니다. WandB 로깅이 비활성화됩니다.")
                use_wandb = False

        # 데이터 로더 생성
        dataset = MedicalVolumeDataset(
            csv_file=os.path.join(data_dir, "train_dataset.csv"),
            target_shape=self.input_shape
        )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        self.num_domains = dataset.num_domains 

        # TTUR: G와 D의 학습률을 config에서 받아서 사용
        optimizer_G = optim.Adam(
            list(self.E_c.parameters()) + list(self.E_s.parameters()) + list(self.G.parameters()) + list(self.style_embeddings.parameters()),
            lr=generator_lr, betas=(beta1, beta2)
        )
        optimizer_D = optim.Adam(self.D.parameters(), lr=discriminator_lr, betas=(beta1, beta2))
        optimizers = {"G": optimizer_G, "D": optimizer_D}

        # 스케줄러 설정 (옵션)
        scheduler_G = optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
        scheduler_D = optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)
        schedulers = {"G": scheduler_G, "D": scheduler_D}

        scaler = torch.amp.GradScaler('cuda', ) if use_mixed_precision else None
        losses_log = defaultdict(list)
        best_val_loss = float("inf")
        best_epoch = -1

        # neutral_label = torch.tensor([dataset.neutral_domain_label] * batch_size, dtype=torch.long, device=device)

        for epoch in range(num_epochs):
            epoch_losses = defaultdict(float)
            num_batches = len(data_loader)

            # config에서 warmup 단계/최대값을 받아서 스케줄링
            if use_warmup:
                lambda_con_now = self._get_lambda(epoch, lambda_con_max, warmup_con_epochs)
                lambda_sty_now = self._get_lambda(epoch, lambda_sty_max, warmup_sty_epochs)
                lambda_cyc_now = self._get_lambda(epoch, lambda_cyc_max, warmup_cyc_epochs)
                lambda_grl_now = self._get_lambda(epoch, lambda_grl_max, warmup_grl_epochs)
            else:
                lambda_con_now = lambda_con_max
                lambda_sty_now = lambda_sty_max
                lambda_cyc_now = lambda_cyc_max
                lambda_grl_now = lambda_grl_max
            # lambda_cls는 고정

            self.grl.lambda_val = lambda_grl_now

            with tqdm(total=num_batches, desc=f"에폭 {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for i, (real_x, real_labels) in enumerate(data_loader):
                    real_x = real_x.to(device)
                    real_labels = real_labels.to(device)
                    rand_idx = torch.randperm(real_x.size(0))
                    target_labels = real_labels[rand_idx]
                    real_style_codes = self.style_embeddings(real_labels)
                    target_style_codes = self.style_embeddings(target_labels)

                    # -----------------
                    # Stage A: GRL만 업데이트 (two_stage_training 옵션 적용)
                    # -----------------
                    if two_stage_training:
                        self.E_c.train()
                        self.content_classifier.train()
                        optimizer_G.zero_grad()
                        with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                            content_z_and_features = self.E_c(real_x)
                            pred_content_cls = self.content_classifier(self.grl(content_z_and_features[0]))
                            loss_grl = self.criterion_cls(pred_content_cls, real_labels)
                            loss_stageA = lambda_grl_now * loss_grl
                        if use_mixed_precision:
                            scaler.scale(loss_stageA).backward()
                            scaler.step(optimizer_G)
                            scaler.update()
                        else:
                            loss_stageA.backward()
                            optimizer_G.step()
                        epoch_losses["StageA_grl"] += loss_grl.item()
                        epoch_losses["StageA_total"] += loss_stageA.item()
                    # -----------------
                    # Stage B: Generator, Discriminator 등 업데이트
                    # -----------------
                    # 1. Discriminator 업데이트
                    optimizer_D.zero_grad()
                    with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                        pred_real_gan, pred_real_cls = self.D(real_x)
                        loss_D_real_gan = self.criterion_GAN(pred_real_gan, torch.ones_like(pred_real_gan))
                        loss_D_real_cls = self.criterion_cls(pred_real_cls, real_labels)
                        with torch.no_grad():
                            content_z_and_features = self.E_c(real_x)
                            fake_x = self.G(content_z_and_features, target_style_codes)
                        pred_fake_gan, _ = self.D(fake_x.detach())
                        loss_D_fake_gan = self.criterion_GAN(pred_fake_gan, torch.zeros_like(pred_fake_gan))
                        loss_D = loss_D_real_gan + loss_D_fake_gan + lambda_cls * loss_D_real_cls
                    if use_mixed_precision:
                        scaler.scale(loss_D).backward()
                        scaler.step(optimizer_D)
                        scaler.update()
                    else:
                        loss_D.backward()
                        optimizer_D.step()
                    epoch_losses["D_real_gan"] += loss_D_real_gan.item()
                    epoch_losses["D_fake_gan"] += loss_D_fake_gan.item()
                    epoch_losses["D_real_cls"] += loss_D_real_cls.item()
                    epoch_losses["D_total"] += loss_D.item()

                    # 2. Generator 등 나머지 업데이트
                    optimizer_G.zero_grad()
                    with torch.amp.autocast('cuda', enabled=use_mixed_precision):
                        content_z_and_features = self.E_c(real_x)
                        real_style_feat = self.E_s(real_x)
                        fake_x = self.G(content_z_and_features, target_style_codes)
                        pred_fake_gan, pred_fake_cls = self.D(fake_x)
                        loss_G_adv = self.criterion_GAN(pred_fake_gan, torch.ones_like(pred_fake_gan))
                        loss_G_cls = self.criterion_cls(pred_fake_cls, target_labels)
                        pred_style_feat = self.E_s(fake_x)
                        loss_sty_rec = self.criterion_sty(pred_style_feat, target_style_codes)
                        rec_content_z_and_features = self.E_c(fake_x)
                        loss_con_rec = self.criterion_con(rec_content_z_and_features[0], content_z_and_features[0].detach())
                        rec_x = self.G(rec_content_z_and_features, real_style_codes)
                        loss_cyc = self.criterion_cyc(rec_x, real_x)
                        # loss_idt, loss_mmd 등 생략
                        loss_G = loss_G_adv + lambda_cls * loss_G_cls + \
                                 lambda_sty_now * loss_sty_rec + lambda_con_now * loss_con_rec + \
                                 lambda_cyc_now * loss_cyc
                    if use_mixed_precision:
                        scaler.scale(loss_G).backward()
                        scaler.step(optimizer_G)
                        scaler.update()
                    else:
                        loss_G.backward()
                        optimizer_G.step()
                    epoch_losses["G_adv"] += loss_G_adv.item()
                    epoch_losses["G_cls"] += loss_G_cls.item()
                    epoch_losses["G_sty_rec"] += loss_sty_rec.item()
                    epoch_losses["G_con_rec"] += loss_con_rec.item()
                    epoch_losses["G_cyc"] += loss_cyc.item()
                    epoch_losses["G_total"] += loss_G.item()

                    pbar.set_postfix({k: f"{v / (i + 1):.4f}" for k, v in epoch_losses.items() if "total" in k})
                    pbar.update(1)

            # 에폭 종료 후 손실 평균 계산 및 로그 출력
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            # logger.info(f"Epoch {epoch + 1} Avg Losses: G={avg_losses.get('G_total', 0):.4f}, D={avg_losses.get('D_total', 0):.4f}")
            for k, v in avg_losses.items():
                losses_log[k].append(v)

            # --- WandB 로깅 ---
            if use_wandb and wandb_run is not None:
                wandb_run.log({f"loss/{k}": v for k, v in avg_losses.items()}, step=epoch+1)
                if "G_total" in avg_losses:
                    wandb_run.log({"best/best_loss": best_val_loss, "best/best_epoch": best_epoch}, step=epoch+1)

            # 스케줄러 업데이트
            if scheduler_G: scheduler_G.step()
            if scheduler_D: scheduler_D.step()

            # 체크포인트 저장
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(epoch + 1, optimizers, schedulers)
                self._generate_and_save_samples(
                    data_dir, epoch + 1, hospital_label_idx=3,
                    wandb_run=wandb_run if use_wandb else None  
                )

            # 베스트 모델 저장 (현재는 G_total 기준)
            if avg_losses.get("G_total", float("inf")) < best_val_loss:
                 best_val_loss = avg_losses.get("G_total", float("inf"))
                 best_epoch = epoch + 1
                 self.save_best_model()

            # 손실 그래프를 매 에포크마다 저장
            self._plot_loss_curve(losses_log, best_epoch, best_val_loss)
            # 손실 로그를 JSON 파일로 저장 (베스트 정보 포함)
            log_with_best = dict(losses_log)
            log_with_best["best_epoch"] = best_epoch
            log_with_best["best_loss"] = best_val_loss
            # 베스트 정보가 첫 entry로 오도록 OrderedDict 사용
            from collections import OrderedDict
            ordered_log = OrderedDict()
            ordered_log["best_epoch"] = best_epoch
            ordered_log["best_loss"] = best_val_loss
            for k, v in losses_log.items():
                ordered_log[k] = v
            with open(self.model_dir / "losses_log.json", "w") as f:
                json.dump(ordered_log, f, indent=4)

        # --- WandB 종료 ---
        if use_wandb and wandb_run is not None:
            wandb_run.finish()

        logger.info(f"학습 종료. 추정 베스트 G 손실: {best_val_loss:.4f}")
        return losses_log

    def _plot_loss_curve(self, losses, best_epoch=None, best_loss=None):
        """여러 손실을 subplot으로 저장. 베스트 에포크 표시."""
        keys_g = [k for k in losses if k.startswith("G_")]
        keys_d = [k for k in losses if k.startswith("D_")]
        keys_other = [k for k in losses if k not in keys_g + keys_d]

        n_rows = 1
        n_cols = 3 if keys_other else 2
        plt.figure(figsize=(6 * n_cols, 5))

        # Generator losses
        plt.subplot(1, n_cols, 1)
        for k in keys_g:
            plt.plot(range(1, len(losses[k]) + 1), losses[k], label=k)
        if best_epoch is not None and best_loss is not None and "G_total" in losses:
            plt.scatter([best_epoch], [best_loss], color='red', marker='*', s=200, label=f'Best (epoch {best_epoch})')
            plt.annotate(f'Best: {best_loss:.4f}\n(Epoch {best_epoch})', 
                         xy=(best_epoch, best_loss), 
                         xytext=(best_epoch, best_loss + 0.05),
                         fontsize=12, color='red')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Generator Losses")
        plt.legend()
        plt.grid()

        # Discriminator losses
        plt.subplot(1, n_cols, 2)
        for k in keys_d:
            plt.plot(range(1, len(losses[k]) + 1), losses[k], label=k)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Discriminator Losses")
        plt.legend()
        plt.grid()

        # 기타 손실
        if keys_other:
            plt.subplot(1, n_cols, 3)
            for k in keys_other:
                plt.plot(range(1, len(losses[k]) + 1), losses[k], label=k)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Other Losses")
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.savefig(self.model_dir / f"loss_curve.png")
        plt.close()

    def _generate_and_save_samples(self, data_dir, epoch, hospital_label_idx, max_samples=5, wandb_run=None):
        """샘플 이미지 생성 및 저장 (4개 병원 원본/변환을 한 PNG에 저장) + wandb 업로드"""
        sample_dir = self.model_dir / "samples"
        os.makedirs(sample_dir, exist_ok=True)
        
        val_dataset = MedicalVolumeDataset(
            csv_file=os.path.join(data_dir, "val_dataset.csv"),
            target_shape=self.input_shape
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
        
        self.E_c.eval()
        self.G.eval()
        self.style_embeddings.eval()
        
        # 병원별 샘플 저장용
        hospital_samples = {}
        with torch.no_grad():
            for i, (real_x, real_label) in enumerate(val_loader):
                hospital_id = real_label.item()
                if hospital_id in hospital_samples:
                    continue

                hospital_style_code = self.style_embeddings(torch.tensor([hospital_label_idx], dtype=torch.long, device=device))
                content_z_and_features = self.E_c(real_x.to(device))
                fake_hospital = self.G(content_z_and_features, hospital_style_code)
                # 저장용 numpy 변환
                real_np = real_x.detach().cpu().numpy()[0, 0]
                fake_np = fake_hospital.detach().cpu().numpy()[0, 0]
                hospital_samples[hospital_id] = (real_np, fake_np)
                if len(hospital_samples) >= val_dataset.num_domains:
                    break

        # 4개 병원만 사용 (병원 ID 오름차순)
        hospital_ids = sorted(list(hospital_samples.keys()))[:4]
        if len(hospital_ids) < 4:
            logger.warning("병원 수가 4개 미만입니다. 일부만 시각화됩니다.")
        real_list = []
        fake_list = []
        for hid in hospital_ids:
            real_list.append(hospital_samples[hid][0])
            fake_list.append(hospital_samples[hid][1])
        self._save_sample_slices_multi(real_list, fake_list, sample_dir, epoch, hospital_ids, hospital_label_idx)

        # wandb 이미지 업로드
        if wandb_run is not None:
            import numpy as np
            import PIL.Image
            img_path = sample_dir / f"samples_epoch_{epoch}.png"
            if img_path.exists():
                img = np.array(PIL.Image.open(img_path))
                wandb_run.log({"samples/samples_epoch": wandb.Image(img, caption=f"samples_epoch_{epoch}")}, step=epoch)

        self.E_c.train()
        self.G.train()
        self.style_embeddings.train()

    def _save_sample_slices_multi(self, real_np_list, fake_np_list, sample_dir, epoch, hospital_ids, hospital_label):
        """여러 병원 원본/변환 슬라이스를 하나의 PNG로 저장"""
        num_hospitals = len(real_np_list)
        z_idx = real_np_list[0].shape[2] // 2

        fig, axes = plt.subplots(2, num_hospitals, figsize=(5 * num_hospitals, 10))

        for i in range(num_hospitals):
            # 원본
            slice_real = real_np_list[i][:, :, z_idx]
            axes[0, i].imshow(slice_real, cmap="gray")
            axes[0, i].set_title(f"Real (Hospital {hospital_ids[i]})")
            axes[0, i].axis("off")
            # 변환
            slice_fake = fake_np_list[i][:, :, z_idx]
            axes[1, i].imshow(slice_fake, cmap="gray")
            axes[1, i].set_title(f"Transformed to Hospital {hospital_label}")
            axes[1, i].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(sample_dir / f"samples_epoch_{epoch}.png")
        plt.close()

    def transform_dataset(self, data_dir, output_dir, split="test"):
        """
        데이터셋 변환 
        """
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # 베스트 모델 로드
        try:
            self.E_c.load_state_dict(torch.load(self.best_model_dir / "E_c_best.pth", map_location=device))
            self.G.load_state_dict(torch.load(self.best_model_dir / "G_best.pth", map_location=device))
            self.style_embeddings.load_state_dict(torch.load(self.best_model_dir / "style_embeddings_best.pth", map_location=device))
            logger.info("베스트 모델 로드 완료")
        except FileNotFoundError as e:
            logger.error(f"베스트 모델 파일을 찾을 수 없습니다: {e}. 먼저 학습을 실행하세요.")
            return []

        dataset = MedicalVolumeDataset(
            csv_file=os.path.join(data_dir, f"{split}_dataset.csv"),
            target_shape=self.input_shape
        )
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

        transformed_info = []

        self.E_c.eval()
        self.G.eval()
        self.style_embeddings.eval()
        
        # 변환 기준이 될 병원
        hospital_label_idx = 3  # (예: 3번 병원으로 변환(0~3))
        hospital_style_code = self.style_embeddings(torch.tensor([hospital_label_idx], dtype=torch.long, device=device))

        with torch.no_grad():
            for i, (real_x, _) in enumerate(tqdm(data_loader, desc=f"변환 중")):
                real_x = real_x.to(device)

                # 콘텐츠 추출 및 스타일 변환
                content_z_and_features = self.E_c(real_x)
                fake_hospital = self.G(content_z_and_features, hospital_style_code)

                fake_hospital = torch.clamp(fake_hospital, -1, 1) # Tanh 출력 가정

                fake_np = fake_hospital.cpu().numpy()[0, 0]
                
                orig_path = dataset.data_info.iloc[i]["file_path"]
                orig_filename = Path(orig_path).name
                if orig_filename.endswith(".nii.gz"):
                    orig_name = orig_filename[:-7]
                elif orig_filename.endswith(".nii"):
                    orig_name = orig_filename[:-4]
                else:
                    orig_name = Path(orig_filename).stem

                # hospital_id, patient_id, series_id 추출
                parts = orig_name.split("_")
                hospital_id = parts[0]
                patient_id = parts[1]
                series_id = "_".join(parts[2:])

                out_name = f"transformed_{orig_name}.nii.gz"
                out_path = output_dir / out_name
                
                # --- 메타데이터 복사: spacing, origin, direction ---
                # 원본 이미지 로드
                orig_img = ants.image_read(orig_path)
                fake_img = ants.from_numpy(fake_np)
                fake_img.set_spacing(orig_img.spacing)
                fake_img.set_origin(orig_img.origin)
                fake_img.set_direction(orig_img.direction)
                ants.image_write(fake_img, str(out_path))
                # -------------------------------------------------

                transformed_info.append({
                    "file_path": str(out_path),
                    "hospital_id": hospital_id,
                    "patient_id": patient_id,
                    "series_id": series_id,
                })

        transformed_df = pd.DataFrame(transformed_info)
        transformed_df.to_csv(output_dir / f"{split}_transformed.csv", index=False)

        logger.info(f"데이터셋 변환 완료: {len(transformed_info)}개 이미지")
        return transformed_info

def main():
    """메인 함수"""

    data_dir = '/public/sylee/MDAISS_2/output'
    model_dir = '/public/sylee/MDAISS_2/models/stargan3d_V2'
    output_dir = '/public/sylee/MDAISS_2/output/transformed/stargan3d_V2'
    
    # 데이터로부터 num_domains 결정
    temp_dataset = MedicalVolumeDataset(csv_file=os.path.join(data_dir, "train_dataset.csv"))
    num_domains = temp_dataset.num_domains
    del temp_dataset

    stargan = StarGAN3Dv2(
        model_dir=model_dir,
        input_shape=(128, 128, 128),
        num_domains=num_domains, 
        style_dim=64,
        content_dim=64 
    )

    # 학습 실행
    losses = stargan.train(
        data_dir=data_dir,
        batch_size=6,
        num_epochs=150, 
        lr=1e-4,
        lambda_cyc=0.1, # 기본값 10.0
        lambda_sty=2.0, # 기본값 1.0
        lambda_con=20.0, # 기본값 10.0
        lambda_cls=1.0, # 기본값 1.0
        lambda_grl=0.1, # 기본값 1.0   
        save_interval=1
    )
    # 데이터셋 변환 실행
    stargan.transform_dataset(
        data_dir=data_dir,
        output_dir=output_dir,
        split="test"
    )

if __name__ == "__main__":
    main()