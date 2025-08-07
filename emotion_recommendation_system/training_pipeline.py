import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import wandb
from tqdm import tqdm
import os
import json
from datetime import datetime

from data_processor import DataProcessor, LyricsDataset
from emotion_model import EmotionClassificationModel, EmotionTrainer
from context_encoder import ContextEncoder, SimilarityCalculator, RecommendationEngine

class MetricsCalculator:
    """평가 메트릭 계산 클래스"""
    
    @staticmethod
    def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """회귀 메트릭 계산"""
        metrics = {}
        
        # MSE, MAE
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # 감정별 MSE
        emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        for i, emotion in enumerate(emotion_labels):
            metrics[f'{emotion}_mse'] = mean_squared_error(y_true[:, i], y_pred[:, i])
        
        # 상관계수
        correlations = []
        for i in range(y_true.shape[1]):
            if np.std(y_true[:, i]) > 0 and np.std(y_pred[:, i]) > 0:
                corr, _ = pearsonr(y_true[:, i], y_pred[:, i])
                correlations.append(corr)
            else:
                correlations.append(0.0)
        
        metrics['mean_correlation'] = np.mean(correlations)
        for i, emotion in enumerate(emotion_labels):
            metrics[f'{emotion}_correlation'] = correlations[i]
        
        return metrics
    
    @staticmethod
    def calculate_distribution_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """분포 관련 메트릭 계산"""
        metrics = {}
        
        # KL Divergence
        eps = 1e-8
        y_true_safe = y_true + eps
        y_pred_safe = y_pred + eps
        
        kl_divs = []
        for i in range(len(y_true)):
            kl_div = np.sum(y_true_safe[i] * np.log(y_true_safe[i] / y_pred_safe[i]))
            kl_divs.append(kl_div)
        
        metrics['mean_kl_divergence'] = np.mean(kl_divs)
        metrics['std_kl_divergence'] = np.std(kl_divs)
        
        # JS Divergence
        js_divs = []
        for i in range(len(y_true)):
            m = 0.5 * (y_true_safe[i] + y_pred_safe[i])
            js_div = 0.5 * np.sum(y_true_safe[i] * np.log(y_true_safe[i] / m)) + \
                     0.5 * np.sum(y_pred_safe[i] * np.log(y_pred_safe[i] / m))
            js_divs.append(js_div)
        
        metrics['mean_js_divergence'] = np.mean(js_divs)
        metrics['std_js_divergence'] = np.std(js_divs)
        
        # Earth Mover's Distance
        emd_distances = []
        for i in range(len(y_true)):
            # 누적 분포 계산
            cdf_true = np.cumsum(y_true[i])
            cdf_pred = np.cumsum(y_pred[i])
            emd = np.sum(np.abs(cdf_true - cdf_pred))
            emd_distances.append(emd)
        
        metrics['mean_emd'] = np.mean(emd_distances)
        metrics['std_emd'] = np.std(emd_distances)
        
        return metrics

class TrainingPipeline:
    """전체 훈련 파이프라인"""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: 훈련 설정 딕셔너리
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델 초기화
        self.model = EmotionClassificationModel(
            model_name=config.get('model_name', 'klue/roberta-base'),
            num_emotions=config.get('num_emotions', 6),
            dropout_rate=config.get('dropout_rate', 0.1),
            hidden_dim=config.get('hidden_dim', None)
        ).to(self.device)
        
        # 트레이너 초기화
        self.trainer = EmotionTrainer(
            model=self.model,
            loss_type=config.get('loss_type', 'kl_divergence'),
            loss_kwargs=config.get('loss_kwargs', {})
        )
        
        # 옵티마이저
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 2e-5),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 스케줄러
        self.scheduler = None
        if config.get('use_scheduler', True):
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
        
        # 메트릭 계산기
        self.metrics_calculator = MetricsCalculator()
        
        # 훈련 기록
        self.train_history = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
        
        # Context encoder (추천 시스템용)
        self.context_encoder = None
        if config.get('use_context_encoder', True):
            self.context_encoder = ContextEncoder(
                model_name=config.get('context_model_name', 'klue/bert-base')
            )
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """한 에포크 훈련"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            # 배치를 디바이스로 이동
            batch = {k: v.to(self.device) for k, v in batch.items() if k != 'original_text'}
            
            # 훈련 스텝
            loss = self.trainer.train_step(batch, self.optimizer)
            
            total_loss += loss
            num_batches += 1
            
            # 프로그레스 바 업데이트
            progress_bar.set_postfix({'loss': f'{loss:.4f}'})
        
        return total_loss / num_batches
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_predictions = []
        all_targets = []
        
        progress_bar = tqdm(val_loader, desc="Validation")
        for batch in progress_bar:
            # 배치를 디바이스로 이동
            batch_device = {k: v.to(self.device) for k, v in batch.items() if k != 'original_text'}
            
            # 검증 스텝
            loss, predictions, targets = self.trainer.validate_step(batch_device)
            
            total_loss += loss
            num_batches += 1
            
            # 예측값과 타겟 수집
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            progress_bar.set_postfix({'val_loss': f'{loss:.4f}'})
        
        # 메트릭 계산
        all_predictions = np.vstack(all_predictions)
        all_targets = np.vstack(all_targets)
        
        regression_metrics = self.metrics_calculator.calculate_regression_metrics(
            all_targets, all_predictions
        )
        distribution_metrics = self.metrics_calculator.calculate_distribution_metrics(
            all_targets, all_predictions
        )
        
        # 메트릭 합치기
        metrics = {**regression_metrics, **distribution_metrics}
        
        return total_loss / num_batches, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
             num_epochs: int, save_dir: str = './checkpoints') -> Dict:
        """전체 훈련 실행"""
        
        # 저장 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)
        
        # WandB 초기화 (선택적)
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config.get('wandb_project', 'emotion-classification'),
                config=self.config
            )
        
        best_val_loss = float('inf')
        early_stop_counter = 0
        early_stop_patience = self.config.get('early_stop_patience', 5)
        
        print(f"Training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.config.get('model_name', 'klue/roberta-base')}")
        print(f"Loss function: {self.config.get('loss_type', 'kl_divergence')}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            # 훈련
            train_loss = self.train_epoch(train_loader)
            
            # 검증
            val_loss, metrics = self.validate_epoch(val_loader)
            
            # 스케줄러 업데이트
            if self.scheduler:
                self.scheduler.step(val_loss)
            
            # 기록 저장
            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['metrics'].append(metrics)
            
            # 결과 출력
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val MSE: {metrics['mse']:.4f}")
            print(f"Val MAE: {metrics['mae']:.4f}")
            print(f"Val Mean Correlation: {metrics['mean_correlation']:.4f}")
            print(f"Val KL Divergence: {metrics['mean_kl_divergence']:.4f}")
            
            # WandB 로깅
            if self.config.get('use_wandb', False):
                wandb.log({
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **metrics
                })
            
            # 최고 모델 저장
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                
                # 모델 저장
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'metrics': metrics,
                    'config': self.config
                }
                
                torch.save(checkpoint, os.path.join(save_dir, 'best_model.pt'))
                print(f"Best model saved with val_loss: {val_loss:.4f}")
            else:
                early_stop_counter += 1
            
            # Early stopping
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # 훈련 완료 후 처리
        if self.config.get('use_wandb', False):
            wandb.finish()
        
        return self.train_history
    
    def create_recommendation_system(self, lyrics_list: List[str], 
                                   emotion_predictions: np.ndarray,
                                   metadata_list: Optional[List[Dict]] = None) -> RecommendationEngine:
        """추천 시스템 생성"""
        if self.context_encoder is None:
            raise ValueError("Context encoder not initialized")
        
        print("Creating context vectors...")
        context_vectors = self.context_encoder.encode_batch(lyrics_list)
        
        print("Building recommendation system...")
        similarity_calculator = SimilarityCalculator(
            context_vectors=context_vectors,
            emotion_vectors=emotion_predictions,
            metadata=metadata_list
        )
        
        recommendation_engine = RecommendationEngine(
            context_encoder=self.context_encoder,
            similarity_calculator=similarity_calculator
        )
        
        return recommendation_engine
    
    def predict_emotions(self, data_loader: DataLoader) -> Tuple[np.ndarray, List[str]]:
        """감정 예측"""
        self.model.eval()
        
        all_predictions = []
        all_texts = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting emotions"):
                batch_device = {k: v.to(self.device) for k, v in batch.items() if k != 'original_text'}
                
                outputs = self.model(
                    input_ids=batch_device['input_ids'],
                    attention_mask=batch_device['attention_mask']
                )
                
                all_predictions.append(outputs['emotion_probs'].cpu().numpy())
                all_texts.extend(batch['original_text'])
        
        return np.vstack(all_predictions), all_texts
    
    def save_training_plots(self, save_dir: str):
        """훈련 과정 시각화 저장"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 손실 곡선
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['train_loss'], label='Train Loss')
        plt.plot(self.train_history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # 메트릭 곡선
        plt.subplot(1, 2, 2)
        mse_values = [m['mse'] for m in self.train_history['metrics']]
        correlation_values = [m['mean_correlation'] for m in self.train_history['metrics']]
        
        plt.plot(mse_values, label='MSE', color='red')
        plt.plot(correlation_values, label='Mean Correlation', color='blue')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Validation Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 감정별 상관계수 히트맵
        if self.train_history['metrics']:
            emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
            correlations = []
            
            for emotion in emotion_labels:
                corr_key = f'{emotion}_correlation'
                if corr_key in self.train_history['metrics'][-1]:
                    correlations.append(self.train_history['metrics'][-1][corr_key])
                else:
                    correlations.append(0.0)
            
            plt.figure(figsize=(8, 6))
            correlation_matrix = np.array(correlations).reshape(1, -1)
            sns.heatmap(correlation_matrix, annot=True, cmap='viridis', 
                       xticklabels=emotion_labels, yticklabels=['Correlation'])
            plt.title('Final Emotion-wise Correlations')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'emotion_correlations.png'), dpi=300, bbox_inches='tight')
            plt.close()

def load_model_from_checkpoint(checkpoint_path: str, config: Dict) -> TrainingPipeline:
    """체크포인트에서 모델 로드"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 파이프라인 생성
    pipeline = TrainingPipeline(config)
    
    # 모델 상태 로드
    pipeline.model.load_state_dict(checkpoint['model_state_dict'])
    pipeline.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"Best validation loss: {checkpoint['val_loss']:.4f}")
    
    return pipeline

if __name__ == "__main__":
    # 테스트용 설정
    config = {
        'model_name': 'klue/roberta-base',
        'num_emotions': 6,
        'dropout_rate': 0.1,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'loss_type': 'kl_divergence',
        'use_scheduler': True,
        'early_stop_patience': 5,
        'use_wandb': False,
        'use_context_encoder': True,
        'context_model_name': 'klue/bert-base'
    }
    
    print("Training pipeline configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")