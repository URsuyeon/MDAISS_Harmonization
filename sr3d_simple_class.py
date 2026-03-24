#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
3D-CNN 기반 병원 분류 모델

이 모듈은 GAN 모델에서 사용하던 간단한 3D CNN을 사용하여 DICOM 이미지로부터 병원을 분류하는 모델을 구현합니다.
주요 기능:
1. 3D CNN 기반 분류 모델 구현 (GAN의 Discriminator 구조 사용)
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
import random

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

# 랜덤 시드 설정
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class MedicalVolumeDataset(Dataset):
    """3D 의료 영상 데이터셋 클래스"""
    
    def __init__(self, csv_file, transform=None, target_shape=(64, 64, 64), hospital_to_label=None, label_to_hospital=None):
        """
        초기화 함수
        
        Args:
            csv_file (str): 데이터셋 CSV 파일 경로
            transform (callable, optional): 데이터 변환 함수
            target_shape (tuple, optional): 입력 볼륨의 목표 크기
            hospital_to_label (dict, optional): 병원명→레이블 매핑
            label_to_hospital (dict, optional): 레이블→병원명 매핑
        """
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform
        self.target_shape = target_shape
        
        if hospital_to_label is not None and label_to_hospital is not None:
            self.hospital_to_label = hospital_to_label
            self.label_to_hospital = label_to_hospital
            self.hospital_ids = list(hospital_to_label.keys())
        else:
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

class Simple3DClassifier(nn.Module):
    """GAN의 Discriminator 구조를 기반으로 한 3D 분류기"""
    def __init__(self, input_channels=1, base_filters=32, num_classes=4):
        super(Simple3DClassifier, self).__init__()
        
        # 초기 레이어: Conv3d -> LeakyReLU
        self.initial = nn.Sequential(
            nn.Conv3d(input_channels, base_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 이후 레이어: 3개의 Conv3d 블록 (InstanceNorm3d + LeakyReLU)
        self.layers = nn.ModuleList()
        in_channels = base_filters
        for i in range(3):
            out_channels = min(base_filters * (2 ** (i + 1)), 512)
            self.layers.append(
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                    nn.InstanceNorm3d(out_channels, affine=True),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )
            in_channels = out_channels
        
        # Global Pooling 및 Fully Connected
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(in_channels, num_classes)
    
    def forward(self, x):
        x = self.initial(x)
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

class HospitalClassifier:
    """병원 분류 모델 학습 및 평가 클래스"""
    
    def __init__(self, data_dir, model_dir, num_classes=4):
        """
        초기화 함수
        
        Args:
            data_dir (str): 데이터 디렉토리 경로
            model_dir (str): 모델 저장 디렉토리 경로
            num_classes (int): 클래스 수 (병원 수)
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.num_classes = num_classes
        
        # 모델 디렉토리 생성
        os.makedirs(self.model_dir, exist_ok=True)
        
        # 모델 생성 (GAN에서 사용한 Discriminator 기반)
        self.model = Simple3DClassifier(input_channels=1, base_filters=32, num_classes=num_classes)
        self.model = self.model.to(device)
        
        logger.info(f"병원 분류 모델 초기화 완료: Simple3DClassifier 구조, 클래스 수: {num_classes}")
    
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
        # 1. train 데이터셋 생성 및 병원 매핑 추출
        train_dataset = MedicalVolumeDataset(
            # 파일 경로: 변환 데이터를 분류할 때
            csv_file=str(self.data_dir / 'transformed/sr3d/train_sr_transformed.csv'),
            
            # 파일 경로: 원본 데이터를 분류할 때
            # csv_file=str(self.data_dir / 'train_dataset_2.csv'),
            target_shape=(256, 256, 256)
        )
        # train 기준 병원 매핑 고정
        hospital_to_label = train_dataset.hospital_to_label
        label_to_hospital = train_dataset.label_to_hospital

        # 2. val 데이터셋에도 동일 매핑 전달
        val_dataset = MedicalVolumeDataset(
            # 파일 경로: 변환 데이터를 분류할 때
            csv_file=str(self.data_dir / 'transformed/sr3d/val_sr_transformed.csv'),
            
            # 파일 경로: 원본 데이터를 분류할 때
            # csv_file=str(self.data_dir / 'val_dataset_2.csv'),
            target_shape=(256, 256, 256),
            hospital_to_label=hospital_to_label,
            label_to_hospital=label_to_hospital
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
            batch_size=int(batch_size/4),
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        # 손실 함수 및 옵티마이저 설정
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
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
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # 검증 모드
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Val)"):
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            scheduler.step(val_loss)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # 최고 모델 저장
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), self.model_dir / "best_model_simple.pth")
                logger.info(f"새로운 최고 모델 저장: 검증 정확도 {val_acc:.4f}")
        
        torch.save(self.model.state_dict(), self.model_dir / "final_model_simple.pth")

        self.hospital_to_label = hospital_to_label
        self.label_to_hospital = label_to_hospital
        self._plot_training_curves(train_losses, val_losses, train_accs, val_accs)
        
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
        if model_path is None:
            model_path = self.model_dir / "best_model_simple.pth"
        
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # train에서 저장한 매핑 사용
        hospital_to_label = getattr(self, 'hospital_to_label', None)
        label_to_hospital = getattr(self, 'label_to_hospital', None)
        if hospital_to_label is None or label_to_hospital is None:
            raise RuntimeError("train()을 먼저 실행하여 병원 매핑을 생성해야 합니다.")

        test_dataset = MedicalVolumeDataset(
            # 파일 경로: 변환 데이터를 분류할 때
            csv_file=str(self.data_dir / 'transformed/sr3d/test_sr_transformed.csv'),
            
            # 파일 경로: 원본 데이터를 분류할 때
            # csv_file=str(self.data_dir / 'test_dataset_2.csv'),
            
            target_shape=(256, 256, 256),
            hospital_to_label=hospital_to_label,
            label_to_hospital=label_to_hospital
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=1,
            pin_memory=True
        )
        
        label_to_hospital = test_dataset.label_to_hospital
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc="테스트 중"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        cm = confusion_matrix(all_labels, all_preds)
        class_names = [label_to_hospital[i] for i in range(self.num_classes)]
        report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
        
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(self.num_classes):
            fpr[i], tpr[i], _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        self._plot_confusion_matrix(cm, class_names)
        self._plot_roc_curves(fpr, tpr, roc_auc, class_names)
        
        logger.info(f"테스트 정확도: {accuracy:.4f}")
        logger.info(f"분류 보고서:\n{classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)}")
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'roc_auc': roc_auc
        }
    
    def _plot_training_curves(self, train_losses, val_losses, train_accs, val_accs):
        """
        학습 곡선 시각화
        """
        epochs = range(1, len(train_losses) + 1)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.model_dir / 'training_curves_simple.png')
        plt.close()
    
    def _plot_confusion_matrix(self, cm, class_names):
        """
        혼동 행렬 시각화
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig(self.model_dir / 'confusion_matrix_simple.png')
        plt.close()
    
    def _plot_roc_curves(self, fpr, tpr, roc_auc, class_names):
        """
        ROC 곡선 시각화
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
        plt.savefig(self.model_dir / 'roc_curves_simple.png')
        plt.close()

if __name__ == "__main__":
    data_dir = '/public/sylee/MDAISS_2/output'
    
    # 모델 저장 경로: 변환 데이터를 분류할 때
    model_dir = '/public/sylee/MDAISS_2/models/classifier/simple_transformed'
    
    # 모델 저장 경로: 원본 데이터를 분류할 때
    # model_dir = '/public/sylee/MDAISS_2/models/classifier/simple_original'
    
    classifier = HospitalClassifier(data_dir, model_dir, num_classes=4)
    classifier.train(batch_size=8, num_epochs=50, learning_rate=0.0001)
    results = classifier.evaluate()
    print(results)
