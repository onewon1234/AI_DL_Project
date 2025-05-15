import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.generation.cache_utils import EncoderDecoderCache
import numpy as np
from sklearn.model_selection import train_test_split
import json

class SentenceOrderDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=200):
        """
        문장 순서 예측을 위한 데이터셋 클래스
        
        Args:
            texts: 입력 문장들 ([SEP] 토큰으로 구분됨)
            labels: 정답 순서 (공백으로 구분된 숫자 시퀀스)
            tokenizer: T5TokenizerFast 또는 AutoTokenizer 인스턴스
            max_length: 최대 입력 길이
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = str(self.labels[idx])
        
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            label,
            max_length=8,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze()
        }

class SentenceOrderPredictor:
    def __init__(self, model_name="paust/pko-t5-base", device=None, use_auto_classes=False):
        """
        문장 순서 예측 모델 클래스
        
        Args:
            model_name: 사용할 사전학습 모델 이름
            device: 학습에 사용할 디바이스 (GPU/CPU)
            use_auto_classes: AutoTokenizer와 AutoModelForSeq2SeqLM 사용 여부
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 모델과 토크나이저 초기화
        if use_auto_classes:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        else:
            self.tokenizer = T5TokenizerFast.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        self.model.to(self.device)
        
    def prepare_data(self, df, train_size=0.9, random_state=42):
        """
        학습 및 검증 데이터셋 준비
        
        Args:
            df: 입력 데이터프레임
            train_size: 학습 데이터 비율
            random_state: 랜덤 시드
        """
        texts = df['input_text'].tolist()
        labels = df['target_text'].tolist()
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, train_size=train_size, random_state=random_state
        )
        
        train_dataset = SentenceOrderDataset(train_texts, train_labels, self.tokenizer)
        val_dataset = SentenceOrderDataset(val_texts, val_labels, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset, batch_size=8):
        """
        학습 및 검증용 데이터로더 생성
        
        Args:
            train_dataset: 학습 데이터셋
            val_dataset: 검증 데이터셋
            batch_size: 배치 크기
        """
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        return train_loader, val_loader
    
    def save_checkpoint(self, epoch, model, optimizer, loss, path):
        """
        모델 체크포인트 저장
        
        Args:
            epoch: 현재 에폭
            model: 현재 모델 상태
            optimizer: 현재 옵티마이저 상태
            loss: 현재 손실값
            path: 저장 경로
        """
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, path)
    
    def load_checkpoint(self, path):
        """
        모델 체크포인트 불러오기
        
        Args:
            path: 체크포인트 파일 경로
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def save_history(self, history, path):
        """
        학습 히스토리 저장
        
        Args:
            history: 학습 히스토리 딕셔너리
            path: 저장 경로
        """
        with open(path, 'w') as f:
            json.dump(history, f)
    
    def load_history(self, path):
        """
        학습 히스토리 불러오기
        
        Args:
            path: 히스토리 파일 경로
        """
        with open(path, 'r') as f:
            return json.load(f)
    
    def predict_order(self, text):
        """
        문장 순서 예측
        
        Args:
            text: 입력 문장들 ([SEP] 토큰으로 구분됨)
        Returns:
            predicted_order: 예측된 순서 (정수 리스트)
        """
        self.model.eval()
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                max_length=200,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            outputs = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=8,
                num_beams=4,
                no_repeat_ngram_size=4,
                use_cache=True,
                past_key_values=None  # EncoderDecoderCache will be used internally
            )
            
            predicted_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_order = [int(x) for x in predicted_text.split()]
            
            return predicted_order

def compute_accuracy(pred_order, true_order):
    """
    예측 순서와 실제 순서 간의 정확도 계산
    
    Args:
        pred_order: 예측된 순서
        true_order: 실제 순서
    Returns:
        정확도 (0 또는 1)
    """
    if isinstance(true_order, str):
        true_order = [int(x) for x in true_order.split()]
    return int(np.array_equal(pred_order, true_order))