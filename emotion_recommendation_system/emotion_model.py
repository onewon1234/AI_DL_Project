import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Optional, Tuple
import numpy as np

class EmotionClassificationModel(nn.Module):
    """RoBERTa 기반 감정 비율 예측 모델"""
    
    def __init__(self, model_name: str = "klue/roberta-base", num_emotions: int = 6, 
                 dropout_rate: float = 0.1, hidden_dim: Optional[int] = None):
        super().__init__()
        
        self.num_emotions = num_emotions
        self.model_name = model_name
        
        # RoBERTa 백본 로드
        self.config = AutoConfig.from_pretrained(model_name)
        self.roberta = AutoModel.from_pretrained(model_name)
        
        # 히든 차원 설정
        if hidden_dim is None:
            hidden_dim = self.config.hidden_size // 2
        
        # 분류 헤드
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_emotions),
            nn.Softmax(dim=-1)  # 확률 분포로 출력
        )
        
        # 문맥 벡터 추출을 위한 프로젝션 레이어
        self.context_projector = nn.Sequential(
            nn.Linear(self.config.hidden_size, hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, 
                return_context: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            input_ids: 토큰 ID 텐서 [batch_size, seq_len]
            attention_mask: 어텐션 마스크 [batch_size, seq_len]
            return_context: 문맥 벡터 반환 여부
            
        Returns:
            Dict containing:
                - emotion_probs: 감정 확률 분포 [batch_size, num_emotions]
                - context_vector: 문맥 벡터 [batch_size, hidden_dim] (if return_context=True)
        """
        # RoBERTa 인코딩
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # [CLS] 토큰의 표현 사용
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # 감정 확률 예측
        emotion_probs = self.classifier(pooled_output)
        
        result = {
            'emotion_probs': emotion_probs,
            'pooled_output': pooled_output
        }
        
        # 문맥 벡터 생성
        if return_context:
            context_vector = self.context_projector(pooled_output)
            result['context_vector'] = context_vector
            
        return result

class LossFunction:
    """다양한 손실 함수들을 제공하는 클래스"""
    
    @staticmethod
    def kl_divergence_loss(pred_probs: torch.Tensor, target_probs: torch.Tensor, 
                          reduction: str = 'mean') -> torch.Tensor:
        """
        KL Divergence Loss
        KL(P||Q) = Σ P(x) * log(P(x) / Q(x))
        """
        # 수치적 안정성을 위해 작은 값 추가
        eps = 1e-8
        pred_probs = pred_probs + eps
        target_probs = target_probs + eps
        
        # KL divergence 계산
        kl_div = target_probs * torch.log(target_probs / pred_probs)
        kl_div = kl_div.sum(dim=-1)
        
        if reduction == 'mean':
            return kl_div.mean()
        elif reduction == 'sum':
            return kl_div.sum()
        else:
            return kl_div
    
    @staticmethod
    def jensen_shannon_divergence(pred_probs: torch.Tensor, target_probs: torch.Tensor,
                                 reduction: str = 'mean') -> torch.Tensor:
        """
        Jensen-Shannon Divergence (대칭적이고 KL보다 안정적)
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = 0.5 * (P + Q)
        """
        eps = 1e-8
        pred_probs = pred_probs + eps
        target_probs = target_probs + eps
        
        # M = (P + Q) / 2
        m = 0.5 * (pred_probs + target_probs)
        
        # JS divergence
        js_div = 0.5 * LossFunction.kl_divergence_loss(pred_probs, m, 'none') + \
                 0.5 * LossFunction.kl_divergence_loss(target_probs, m, 'none')
        
        if reduction == 'mean':
            return js_div.mean()
        elif reduction == 'sum':
            return js_div.sum()
        else:
            return js_div
    
    @staticmethod
    def earth_mover_distance(pred_probs: torch.Tensor, target_probs: torch.Tensor,
                           reduction: str = 'mean') -> torch.Tensor:
        """
        Earth Mover's Distance (Wasserstein Distance)
        감정 간의 순서적 관계를 고려할 수 있음
        """
        # 누적 분포 계산
        pred_cdf = torch.cumsum(pred_probs, dim=-1)
        target_cdf = torch.cumsum(target_probs, dim=-1)
        
        # EMD = L1 거리
        emd = torch.abs(pred_cdf - target_cdf).sum(dim=-1)
        
        if reduction == 'mean':
            return emd.mean()
        elif reduction == 'sum':
            return emd.sum()
        else:
            return emd
    
    @staticmethod
    def focal_loss(pred_probs: torch.Tensor, target_probs: torch.Tensor,
                   alpha: float = 1.0, gamma: float = 2.0, 
                   reduction: str = 'mean') -> torch.Tensor:
        """
        Focal Loss for addressing class imbalance
        """
        eps = 1e-8
        pred_probs = pred_probs + eps
        
        # Cross entropy
        ce_loss = -target_probs * torch.log(pred_probs)
        ce_loss = ce_loss.sum(dim=-1)
        
        # Focal weight
        pt = (target_probs * pred_probs).sum(dim=-1)
        focal_weight = alpha * (1 - pt) ** gamma
        
        focal_loss = focal_weight * ce_loss
        
        if reduction == 'mean':
            return focal_loss.mean()
        elif reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EmotionTrainer:
    """감정 분류 모델 훈련을 위한 트레이너"""
    
    def __init__(self, model: EmotionClassificationModel, loss_type: str = 'kl_divergence',
                 loss_kwargs: Optional[Dict] = None):
        self.model = model
        self.loss_type = loss_type
        self.loss_kwargs = loss_kwargs or {}
        
        # 손실 함수 매핑
        self.loss_functions = {
            'kl_divergence': LossFunction.kl_divergence_loss,
            'js_divergence': LossFunction.jensen_shannon_divergence,
            'emd': LossFunction.earth_mover_distance,
            'focal': LossFunction.focal_loss,
            'mse': nn.MSELoss()
        }
        
        if loss_type not in self.loss_functions:
            raise ValueError(f"Unsupported loss type: {loss_type}")
    
    def compute_loss(self, pred_probs: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
        """손실 계산"""
        if self.loss_type == 'mse':
            return self.loss_functions[self.loss_type](pred_probs, target_probs)
        else:
            return self.loss_functions[self.loss_type](pred_probs, target_probs, **self.loss_kwargs)
    
    def train_step(self, batch: Dict[str, torch.Tensor], optimizer: torch.optim.Optimizer) -> float:
        """한 배치 훈련 스텝"""
        self.model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask']
        )
        
        # 손실 계산
        loss = self.compute_loss(outputs['emotion_probs'], batch['emotion_ratios'])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """검증 스텝"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask']
            )
            
            loss = self.compute_loss(outputs['emotion_probs'], batch['emotion_ratios'])
            
        return loss.item(), outputs['emotion_probs'], batch['emotion_ratios']

def compare_loss_functions():
    """다양한 손실 함수들의 특성 비교"""
    print("Loss Function Comparison:")
    print("=" * 50)
    
    # 샘플 데이터
    pred = torch.tensor([[0.8, 0.1, 0.05, 0.03, 0.01, 0.01]])
    target = torch.tensor([[0.6, 0.2, 0.1, 0.05, 0.03, 0.02]])
    
    print(f"Predicted: {pred[0].numpy()}")
    print(f"Target:    {target[0].numpy()}")
    print()
    
    # 각 손실 함수 계산
    losses = {
        'KL Divergence': LossFunction.kl_divergence_loss(pred, target),
        'JS Divergence': LossFunction.jensen_shannon_divergence(pred, target),
        'Earth Mover Distance': LossFunction.earth_mover_distance(pred, target),
        'Focal Loss': LossFunction.focal_loss(pred, target),
        'MSE': F.mse_loss(pred, target)
    }
    
    for name, loss in losses.items():
        print(f"{name:20}: {loss.item():.6f}")

if __name__ == "__main__":
    # 모델 테스트
    model = EmotionClassificationModel()
    
    # 샘플 입력
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    # Forward pass
    outputs = model(input_ids, attention_mask, return_context=True)
    
    print("Model output shapes:")
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")
    
    print("\n")
    compare_loss_functions()