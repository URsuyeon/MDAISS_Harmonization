#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D-CNN 기반 병원 분류 모델

이 모듈은 3D ResNet을 사용하여 DICOM 이미지로부터 병원을 분류하는 모델을 구현합니다.

주요 기능:
1. 3D ResNet 모델 구현
2. 데이터 로더 및 학습 파이프라인
3. 모델 평가 및 시각화
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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
from pathlib import Path
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CUDA 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"GPU가 사용됩니다: {torch.cuda.get_device_name(0)}")
else:
    logger.info("GPU 사용 불가, CPU로 실행됩니다.")
logger.info(f"Using device: {device}")        

class MedicalVolumeDataset(Dataset):
    """3D 의료 영상 데이터셋 클래스"""
    
    def __init__(self, csv_file, transform=None, target_shape=(64, 64, 64)):
        """
        초기화 함수
        
        Args:
            csv_file (str): 데이터셋 CSV 파일 경로
            transform (callable, optional): 데이터 변환 함수
            target_shape (tuple, optional): 입력 볼륨의 목표 크기
        """
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform
        self.target_shape = target_shape
        
        # 병원 ID를 숫자 레이블로 변환
        self.hospital_ids = self.data_info['hospital_id'].unique()
        self.hospital_to_label = {hospital: i for i, hospital in enumerate(self.hospital_ids)}
        self.label_to_hospital = {i: hospital for i, hospital in enumerate(self.hospital_ids)}
        
        logger.info(f"데이터셋 로드 완료: {len(self.data_info)} 샘플, {len(self.hospital_ids)} 병원")
        logger.info(f"병원 ID 매핑: {self.hospital_to_label}")
    
    def __len__(self):
        """데이터셋 길이 반환"""
        return len(self.data_info)
    
    def __getitem__(self, idx):
        """데이터셋 아이템 반환"""
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # 이미지 파일 경로
        img_path = self.data_info.iloc[idx]['file_path']
        
        # 이미지 로드 및 전처리
        img_tensor = self._load_and_preprocess_image(img_path)
        
        # 병원 레이블
        hospital_id = self.data_info.iloc[idx]['hospital_id']
        label = self.hospital_to_label[hospital_id]
        
        # 변환 적용 (있는 경우)
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return img_tensor, label
    
    def _load_and_preprocess_image(self, img_path):
        """
        주어진 이미지 경로에서 이미지를 로드하고 전처리합니다.
        
        Args:
            img_path (str): 이미지 파일 경로
            
        Returns:
            torch.Tensor: 전처리된 이미지 텐서
        """
        # ANTsPy로 이미지 로드
        img = ants.image_read(img_path)
        
        # 이미지 리샘플링 (목표 크기로)
        img = self._resample_volume(img)
        
        # NumPy 배열로 변환
        img_array = img.numpy()
        
        # 차원 추가 (채널 차원)
        img_tensor = torch.from_numpy(img_array).float().unsqueeze(0)
        
        return img_tensor
    
    def _resample_volume(self, img):
        """
        볼륨 리샘플링
        
        Args:
            img (ants.ANTsImage): ANTsPy 이미지 객체
            
        Returns:
            ants.ANTsImage: 리샘플링된 이미지
        """
        # 현재 이미지 크기
        current_shape = img.shape
        
        # 목표 크기와 다른 경우에만 리샘플링
        if current_shape != self.target_shape:
            # 목표 간격 계산
            current_spacing = img.spacing
            physical_size = [s * sp for s, sp in zip(current_shape, current_spacing)]
            target_spacing = [ps / ts for ps, ts in zip(physical_size, self.target_shape)]
            
            # 리샘플링
            img = ants.resample_image(img, target_spacing, use_voxels=False, interp_type=1)
            
            # 크기 확인 및 조정 (리샘플링 후 크기가 정확히 목표 크기와 일치하지 않을 수 있음)
            if img.shape != self.target_shape:
                # 크롭 또는 패딩
                img = self._crop_or_pad(img, self.target_shape)
        
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
        
        # 크롭 또는 패딩 적용
        result[tuple(slices_dst)] = img_array[tuple(slices_src)]
        
        # ANTsImage로 변환
        result_img = ants.from_numpy(result)
        result_img.set_spacing(img.spacing)
        result_img.set_origin(img.origin)
        result_img.set_direction(img.direction)
        
        return result_img

class Conv3DBlock(nn.Module):
    """3D 컨볼루션 블록"""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_batch_norm=True):
        """
        초기화 함수
        
        Args:
            in_channels (int): 입력 채널 수
            out_channels (int): 출력 채널 수
            kernel_size (int): 커널 크기
            stride (int): 스트라이드
            padding (int): 패딩
            use_batch_norm (bool): 배치 정규화 사용 여부
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
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """순전파"""
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.relu(x)
        return x

class ResidualBlock3D(nn.Module):
    """3D 잔차 블록"""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        초기화 함수
        
        Args:
            in_channels (int): 입력 채널 수
            out_channels (int): 출력 채널 수
            stride (int): 스트라이드
            downsample (nn.Module): 다운샘플링 모듈
        """
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        """순전파"""
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out

class ResNet3D(nn.Module):
    """3D ResNet 모델"""
    
    def __init__(self, block, layers, num_classes=4, input_channels=1):
        """
        초기화 함수
        
        Args:
            block (nn.Module): 블록 타입 (ResidualBlock3D)
            layers (list): 각 레이어의 블록 수
            num_classes (int): 클래스 수 (병원 수)
            input_channels (int): 입력 채널 수
        """
        super(ResNet3D, self).__init__()
        
        # self.in_channels = 64
        self.in_channels = 128
        
        # 초기 컨볼루션 레이어
        # self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv3d(input_channels, 128, kernel_size=7, stride=2, padding=3, bias=False)
        # self.bn1 = nn.BatchNorm3d(64)
        self.bn1 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        # 잔차 레이어
        # self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer1 = self._make_layer(block, 128, layers[0])
        self.layer2 = self._make_layer(block, 256, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 512, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 1024, layers[3], stride=2)
        
        # 글로벌 평균 풀링 및 분류기
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # self.fc = nn.Linear(512, num_classes)
        self.fc = nn.Linear(1024, num_classes)
        
        # 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        레이어 생성
        
        Args:
            block (nn.Module): 블록 타입
            out_channels (int): 출력 채널 수
            blocks (int): 블록 수
            stride (int): 스트라이드
            
        Returns:
            nn.Sequential: 레이어
        """
        downsample = None
        
        # 다운샘플링이 필요한 경우 (채널 수 변경 또는 스트라이드 > 1)
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        
        layers = []
        # 첫 번째 블록 (다운샘플링 포함)
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        
        self.in_channels = out_channels
        
        # 나머지 블록
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """순전파"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def resnet18_3d(num_classes=4, input_channels=1):
    """
    3D ResNet-18 모델 생성
    
    Args:
        num_classes (int): 클래스 수 (병원 수)
        input_channels (int): 입력 채널 수
        
    Returns:
        ResNet3D: 3D ResNet-18 모델
    """
    return ResNet3D(ResidualBlock3D, [2, 2, 2, 2], num_classes, input_channels)

def resnet34_3d(num_classes=4, input_channels=1):
    """
    3D ResNet-34 모델 생성
    
    Args:
        num_classes (int): 클래스 수 (병원 수)
        input_channels (int): 입력 채널 수
        
    Returns:
        ResNet3D: 3D ResNet-34 모델
    """
    return ResNet3D(ResidualBlock3D, [3, 4, 6, 3], num_classes, input_channels)

class HospitalClassifier:
    """병원 분류 모델 학습 및 평가 클래스"""
    
    def __init__(self, data_dir, model_dir, num_classes=4, model_type='resnet18'):
        """
        초기화 함수
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            model_dir (str): 모델 저장 디렉토리 경로
            num_classes (int): 클래스 수 (병원 수)
            model_type (str): 모델 타입 ('resnet18' 또는 'resnet34')
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.num_classes = num_classes
        self.model_type = model_type
        
        # 모델 디렉토리 생성
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 모델 생성
        if model_type == 'resnet18':
            self.model = resnet18_3d(num_classes=num_classes)
        elif model_type == 'resnet34':
            self.model = resnet34_3d(num_classes=num_classes)
        else:
            raise ValueError(f"지원되지 않는 모델 타입: {model_type}")
        
        self.model = self.model.to(device)
        
        logger.info(f"병원 분류 모델 초기화 완료: {model_type}, 클래스 수: {num_classes}")
    
    def train(self, batch_size=8, num_epochs=50, learning_rate=0.001, weight_decay=1e-4):
        """
        모델 학습
        
        Args:
            batch_size (int): 배치 크기
            num_epochs (int): 에폭 수
            learning_rate (float): 학습률
            weight_decay (float): 가중치 감쇠
            
        Returns:
            dict: 학습 결과 (손실, 정확도 등)
        """
        # 데이터 로더 생성
        train_dataset = MedicalVolumeDataset(
            csv_file=str(self.data_dir / 'train_dataset.csv'),
            target_shape=(128, 128, 128)
        )
        
        val_dataset = MedicalVolumeDataset(
            csv_file=str(self.data_dir / 'val_dataset.csv'),
            target_shape=(128, 128, 128)
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        # 손실 함수 및 옵티마이저 설정
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
        
        # 학습 결과 저장
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        # 최고 검증 정확도 및 해당 모델 저장
        best_val_acc = 0.0
        
        # 학습 루프
        for epoch in range(num_epochs):
            # 학습 모드
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            # 학습 데이터 반복
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Train)"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 그래디언트 초기화
                optimizer.zero_grad()
                
                # 순전파
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # 역전파 및 최적화
                loss.backward()
                optimizer.step()
                
                # 통계 업데이트
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # 에폭 평균 손실 및 정확도 계산
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # 검증 모드
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            # 검증 데이터 반복 (그래디언트 계산 없음)
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    # 순전파
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    # 통계 업데이트
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # 에폭 평균 손실 및 정확도 계산
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # 학습률 스케줄러 업데이트
            scheduler.step(val_loss)
            
            # 결과 출력
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 최고 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_dir / f"best_model_{self.model_type}.pth")
                logger.info(f"새로운 최고 모델 저장: 검증 정확도 {val_acc:.4f}")
        
        # 최종 모델 저장
        torch.save(self.model.state_dict(), self.model_dir / f"final_model_{self.model_type}.pth")
        
        # 학습 곡선 시각화
        self._plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        
        # 학습 결과 반환
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accs': train_accs,
            'val_accs': val_accs,
            'best_val_acc': best_val_acc
        }
    
    def evaluate(self, model_path=None):
        """
        모델 평가
        
        Args:
            model_path (str, optional): 평가할 모델 경로. 기본값은 None으로, 최고 모델 사용
            
        Returns:
            dict: 평가 결과 (정확도, 혼동 행렬 등)
        """
        # 모델 로드
        if model_path is None:
            model_path = self.model_dir / f"best_model_{self.model_type}.pth"
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 테스트 데이터셋 로드
        test_dataset = MedicalVolumeDataset(
            csv_file=str(self.data_dir / 'test_dataset.csv'),
            target_shape=(128, 128, 128)
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        # 병원 ID 매핑
        label_to_hospital = test_dataset.label_to_hospital
        
        # 예측 및 실제 레이블 저장
        all_preds = []
        all_labels = []
        all_probs = []
        
        # 테스트 데이터 반복 (그래디언트 계산 없음)
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="테스트 중"):
                inputs, labels = inputs.to(device), labels.to(device)
                
                # 순전파
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                
                # 예측
                _, preds = torch.max(outputs, 1)
                
                # 결과 저장
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 정확도 계산
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        
        # 혼동 행렬 계산
        cm = confusion_matrix(all_labels, all_preds)
        
        # 분류 보고서 생성
        class_names = [label_to_hospital[i] for i in range(self.num_classes)]
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
        
        # ROC 곡선 및 AUC 계산 (one-vs-rest)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # 결과 시각화
        self._plot_confusion_matrix(cm, class_names)
        self._plot_roc_curves(fpr, tpr, roc_auc, class_names)
        
        # 결과 출력
        logger.info(f"테스트 정확도: {accuracy:.4f}")
        logger.info(f"분류 보고서:\n{classification_report(all_labels, all_preds, target_names=class_names)}")
        
        # 결과 반환
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_auc': roc_auc
        }
    
    def _plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """
        학습 곡선 시각화
        
        Args:
            train_losses (list): 훈련 손실 리스트
            val_losses (list): 검증 손실 리스트
            train_accs (list): 훈련 정확도 리스트
            val_accs (list): 검증 정확도 리스트
        """
        epochs = range(1, len(train_losses) + 1)
        
        # 손실 곡선
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # 정확도 곡선
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_curves.png')
        plt.close()
    
    def _plot_confusion_matrix(self, cm, class_names):
        """
        혼동 행렬 시각화
        
        Args:
            cm (numpy.ndarray): 혼동 행렬
            class_names (list): 클래스 이름 리스트
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(self.model_dir / 'confusion_matrix.png')
        plt.close()
    
    def _plot_roc_curves(self, fpr, tpr, roc_auc, class_names):
        """
        ROC 곡선 시각화
        
        Args:
            fpr (dict): 각 클래스의 False Positive Rate
            tpr (dict): 각 클래스의 True Positive Rate
            roc_auc (dict): 각 클래스의 AUC 값
            class_names (list): 클래스 이름 리스트
        """
        plt.figure(figsize=(10, 8))
        
        for i, class_name in enumerate(class_names):
            plt.plot(fpr[i], tpr[i], label=f'{class_name} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(self.model_dir / 'roc_curves.png')
        plt.close()

def main():
    """메인 함수"""
    # 경로 설정 
    data_dir = '/public/sylee/MDAISS_2/output'
    model_dir = '/public/sylee/MDAISS_2/models/classifier/original'
    
    # 병원 수 설정 
    num_classes = 4
    
    # 병원 분류 모델 생성
    classifier = HospitalClassifier(
        data_dir=data_dir,
        model_dir=model_dir,
        num_classes=num_classes,
        model_type='resnet18' 
    )
    
    # 모델 학습
    train_results = classifier.train(
        batch_size=1,
        num_epochs=100,
        learning_rate=1e-6,
        weight_decay=1e-8
    )
    
    # 모델 평가
    eval_results = classifier.evaluate()
    
    logger.info("병원 분류 모델 학습 및 평가 완료")
    logger.info(f"최고 검증 정확도: {train_results['best_val_acc']:.4f}")
    logger.info(f"테스트 정확도: {eval_results['accuracy']:.4f}")

if __name__ == '__main__':
    main()
