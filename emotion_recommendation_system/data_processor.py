import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import re
from sklearn.preprocessing import StandardScaler

class LyricsDataset(Dataset):
    """노래 가사와 감정 비율을 처리하는 데이터셋 클래스"""
    
    def __init__(self, lyrics: List[str], emotion_ratios: List[List[float]], 
                 tokenizer, max_length: int = 512):
        """
        Args:
            lyrics: 노래 가사 리스트
            emotion_ratios: 6개 감정군별 비율 리스트 [[감정1, 감정2, ..., 감정6], ...]
            tokenizer: RoBERTa 토크나이저
            max_length: 최대 시퀀스 길이
        """
        self.lyrics = lyrics
        self.emotion_ratios = np.array(emotion_ratios)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 감정 비율 정규화 (합이 1이 되도록)
        self.emotion_ratios = self.emotion_ratios / self.emotion_ratios.sum(axis=1, keepdims=True)
        
    def __len__(self):
        return len(self.lyrics)
    
    def __getitem__(self, idx):
        lyric = self.lyrics[idx]
        emotion_ratio = self.emotion_ratios[idx]
        
        # 텍스트 전처리
        processed_lyric = self.preprocess_text(lyric)
        
        # 토큰화
        encoding = self.tokenizer(
            processed_lyric,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'emotion_ratios': torch.FloatTensor(emotion_ratio),
            'original_text': lyric
        }
    
    def preprocess_text(self, text: str) -> str:
        """가사 텍스트 전처리"""
        # 특수 문자 정리
        text = re.sub(r'[^\w\s가-힣]', ' ', text)
        # 여러 공백을 하나로
        text = re.sub(r'\s+', ' ', text)
        # 앞뒤 공백 제거
        text = text.strip()
        return text

class DataProcessor:
    """데이터 로딩 및 전처리를 담당하는 클래스"""
    
    def __init__(self, model_name: str = "klue/roberta-base"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.emotion_labels = [
            'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust'
        ]
    
    def load_data_from_csv(self, file_path: str) -> Tuple[List[str], List[List[float]]]:
        """
        CSV 파일에서 가사와 감정 비율 데이터를 로드
        
        Expected CSV format:
        lyrics, joy, sadness, anger, fear, surprise, disgust
        "가사 내용", 0.2, 0.3, 0.1, 0.1, 0.2, 0.1
        """
        df = pd.read_csv(file_path)
        lyrics = df['lyrics'].tolist()
        emotion_ratios = df[self.emotion_labels].values.tolist()
        
        return lyrics, emotion_ratios
    
    def create_dataset(self, lyrics: List[str], emotion_ratios: List[List[float]], 
                      max_length: int = 512) -> LyricsDataset:
        """데이터셋 객체 생성"""
        return LyricsDataset(lyrics, emotion_ratios, self.tokenizer, max_length)
    
    def create_dataloader(self, dataset: LyricsDataset, batch_size: int = 16, 
                         shuffle: bool = True) -> DataLoader:
        """데이터로더 생성"""
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    def split_data(self, lyrics: List[str], emotion_ratios: List[List[float]], 
                   train_ratio: float = 0.8, val_ratio: float = 0.1) -> Tuple:
        """데이터를 훈련/검증/테스트로 분할"""
        total_len = len(lyrics)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        
        # 인덱스 셔플
        indices = np.random.permutation(total_len)
        
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]
        
        train_lyrics = [lyrics[i] for i in train_indices]
        train_emotions = [emotion_ratios[i] for i in train_indices]
        
        val_lyrics = [lyrics[i] for i in val_indices]
        val_emotions = [emotion_ratios[i] for i in val_indices]
        
        test_lyrics = [lyrics[i] for i in test_indices]
        test_emotions = [emotion_ratios[i] for i in test_indices]
        
        return (train_lyrics, train_emotions), (val_lyrics, val_emotions), (test_lyrics, test_emotions)

def create_sample_data(num_samples: int = 1000) -> Tuple[List[str], List[List[float]]]:
    """샘플 데이터 생성 (테스트용)"""
    import random
    
    # 샘플 가사 템플릿
    sample_lyrics_templates = [
        "사랑하는 마음이 너무 커서 행복해요",
        "이별의 아픔이 너무 슬퍼요",
        "화가 나서 참을 수 없어요",
        "무서운 밤이 두려워요",
        "놀라운 일이 일어났어요",
        "역겨운 기분이에요",
        "즐거운 하루를 보내고 있어요",
        "눈물이 나올 것 같아요",
        "분노가 치밀어 올라요",
        "무섭고 두려운 마음이에요"
    ]
    
    lyrics = []
    emotion_ratios = []
    
    for i in range(num_samples):
        # 랜덤하게 가사 선택 및 변형
        base_lyric = random.choice(sample_lyrics_templates)
        lyrics.append(f"{base_lyric} {i+1}번째 노래")
        
        # 랜덤 감정 비율 생성 (합이 1이 되도록)
        ratios = np.random.dirichlet(np.ones(6))
        emotion_ratios.append(ratios.tolist())
    
    return lyrics, emotion_ratios

if __name__ == "__main__":
    # 테스트 코드
    processor = DataProcessor()
    
    # 샘플 데이터 생성
    lyrics, emotion_ratios = create_sample_data(100)
    
    # 데이터셋 생성
    dataset = processor.create_dataset(lyrics, emotion_ratios)
    dataloader = processor.create_dataloader(dataset, batch_size=4)
    
    # 첫 번째 배치 확인
    for batch in dataloader:
        print("Input IDs shape:", batch['input_ids'].shape)
        print("Attention Mask shape:", batch['attention_mask'].shape)
        print("Emotion Ratios shape:", batch['emotion_ratios'].shape)
        print("First emotion ratios:", batch['emotion_ratios'][0])
        break