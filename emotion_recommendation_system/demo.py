#!/usr/bin/env python3
"""
감정 분석 기반 노래 추천 시스템 데모

사용자가 입력한 가사에 대해 실시간으로 감정을 분석하고
유사한 감정의 노래들을 추천하는 대화형 데모입니다.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from typing import List, Dict

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import DataProcessor, create_sample_data
from emotion_model import EmotionClassificationModel
from context_encoder import ContextEncoder, RecommendationEngine
from training_pipeline import TrainingPipeline

class EmotionRecommendationDemo:
    """감정 분석 및 추천 시스템 데모 클래스"""
    
    def __init__(self, model_path: str = None, sample_size: int = 100):
        """
        Args:
            model_path: 사전 훈련된 모델 경로 (없으면 샘플 데이터로 빠른 훈련)
            sample_size: 샘플 데이터 크기
        """
        self.emotion_labels = ['기쁨', '슬픔', '분노', '두려움', '놀라움', '혐오']
        
        print("🎵 감정 분석 기반 노래 추천 시스템 데모")
        print("=" * 50)
        
        # 모델 및 추천 시스템 초기화
        self.pipeline = None
        self.recommendation_engine = None
        self.sample_lyrics = []
        
        self._initialize_system(model_path, sample_size)
    
    def _initialize_system(self, model_path: str, sample_size: int):
        """시스템 초기화"""
        print("시스템 초기화 중...")
        
        # 설정
        config = {
            'model_name': 'klue/roberta-base',
            'context_model_name': 'klue/bert-base',
            'num_emotions': 6,
            'dropout_rate': 0.1,
            'learning_rate': 2e-5,
            'weight_decay': 0.01,
            'loss_type': 'kl_divergence',
            'batch_size': 8,
            'max_length': 256,
            'num_epochs': 3,  # 데모용으로 빠르게
            'use_scheduler': True,
            'early_stop_patience': 2,
            'use_wandb': False,
            'use_context_encoder': True,
        }
        
        if model_path and os.path.exists(model_path):
            print(f"사전 훈련된 모델 로드 중: {model_path}")
            from training_pipeline import load_model_from_checkpoint
            self.pipeline = load_model_from_checkpoint(model_path, config)
        else:
            print("새 모델을 빠르게 훈련 중...")
            self._quick_train(config, sample_size)
        
        print("추천 시스템 구축 중...")
        self._build_recommendation_system(sample_size)
        
        print("✅ 시스템 초기화 완료!")
        print()
    
    def _quick_train(self, config: Dict, sample_size: int):
        """빠른 훈련 (데모용)"""
        # 샘플 데이터 생성
        lyrics, emotion_ratios = create_sample_data(sample_size)
        self.sample_lyrics = lyrics
        
        # 데이터 전처리
        processor = DataProcessor(model_name=config['model_name'])
        (train_lyrics, train_emotions), (val_lyrics, val_emotions), _ = \
            processor.split_data(lyrics, emotion_ratios, train_ratio=0.8, val_ratio=0.2)
        
        # 데이터셋 생성
        train_dataset = processor.create_dataset(train_lyrics, train_emotions, config['max_length'])
        val_dataset = processor.create_dataset(val_lyrics, val_emotions, config['max_length'])
        
        train_loader = processor.create_dataloader(train_dataset, config['batch_size'], shuffle=True)
        val_loader = processor.create_dataloader(val_dataset, config['batch_size'], shuffle=False)
        
        # 훈련
        self.pipeline = TrainingPipeline(config)
        print("  - 모델 훈련 시작 (빠른 모드)")
        self.pipeline.train(train_loader, val_loader, config['num_epochs'], './demo_checkpoints')
    
    def _build_recommendation_system(self, sample_size: int):
        """추천 시스템 구축"""
        if not self.sample_lyrics:
            self.sample_lyrics, emotion_ratios = create_sample_data(sample_size)
        else:
            _, emotion_ratios = create_sample_data(len(self.sample_lyrics))
        
        # 감정 예측
        processor = DataProcessor(model_name=self.pipeline.config['model_name'])
        dataset = processor.create_dataset(self.sample_lyrics, emotion_ratios, 256)
        loader = processor.create_dataloader(dataset, 8, shuffle=False)
        
        emotion_predictions, _ = self.pipeline.predict_emotions(loader)
        
        # 메타데이터 생성
        metadata = []
        for i, lyric in enumerate(self.sample_lyrics):
            metadata.append({
                'song_id': i,
                'title': f'노래 {i+1}',
                'preview': lyric[:30] + '...' if len(lyric) > 30 else lyric
            })
        
        # 추천 엔진 생성
        self.recommendation_engine = self.pipeline.create_recommendation_system(
            lyrics_list=self.sample_lyrics,
            emotion_predictions=emotion_predictions,
            metadata_list=metadata
        )
    
    def analyze_emotion(self, text: str) -> np.ndarray:
        """텍스트의 감정 분석"""
        processor = DataProcessor(model_name=self.pipeline.config['model_name'])
        
        # 임시 감정 비율 (실제로는 사용되지 않음)
        temp_emotion = [1/6] * 6
        
        dataset = processor.create_dataset([text], [temp_emotion], 256)
        loader = processor.create_dataloader(dataset, 1, shuffle=False)
        
        predictions, _ = self.pipeline.predict_emotions(loader)
        return predictions[0]
    
    def get_recommendations(self, text: str, emotion_vector: np.ndarray, 
                          top_k: int = 5) -> List[Dict]:
        """추천 결과 반환"""
        recommendations = self.recommendation_engine.recommend_by_lyrics(
            query_lyrics=text,
            predicted_emotion=emotion_vector,
            top_k=top_k,
            context_weight=0.7
        )
        
        # 결과 포맷팅
        formatted_recs = []
        for rec in recommendations:
            formatted_recs.append({
                'title': rec['metadata']['title'],
                'preview': rec['metadata']['preview'],
                'similarity': rec['similarity_score'],
                'emotion': rec['emotion_vector']
            })
        
        return formatted_recs
    
    def print_emotion_analysis(self, emotion_vector: np.ndarray):
        """감정 분석 결과 출력"""
        print("📊 감정 분석 결과:")
        print("-" * 30)
        
        for i, (label, value) in enumerate(zip(self.emotion_labels, emotion_vector)):
            bar_length = int(value * 20)  # 0-20 길이 바
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{label:6s}: {bar} {value:.3f}")
        
        # 주요 감정 표시
        main_emotion_idx = np.argmax(emotion_vector)
        main_emotion = self.emotion_labels[main_emotion_idx]
        confidence = emotion_vector[main_emotion_idx]
        
        print(f"\n🎯 주요 감정: {main_emotion} (신뢰도: {confidence:.1%})")
        print()
    
    def print_recommendations(self, recommendations: List[Dict]):
        """추천 결과 출력"""
        print("🎵 추천 곡들:")
        print("-" * 30)
        
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec['title']}")
            print(f"   미리보기: {rec['preview']}")
            print(f"   유사도: {rec['similarity']:.3f}")
            
            # 추천곡의 주요 감정
            main_emotion_idx = np.argmax(rec['emotion'])
            main_emotion = self.emotion_labels[main_emotion_idx]
            print(f"   주요 감정: {main_emotion}")
            print()
    
    def run_interactive_demo(self):
        """대화형 데모 실행"""
        print("💬 대화형 모드를 시작합니다!")
        print("가사를 입력하면 감정을 분석하고 유사한 곡들을 추천해 드립니다.")
        print("종료하려면 'quit' 또는 'exit'를 입력하세요.")
        print()
        
        while True:
            try:
                # 사용자 입력
                user_input = input("🎤 가사를 입력하세요: ").strip()
                
                if user_input.lower() in ['quit', 'exit', '종료', 'q']:
                    print("👋 데모를 종료합니다. 감사합니다!")
                    break
                
                if not user_input:
                    print("❌ 가사를 입력해 주세요.")
                    continue
                
                print("\n🔍 분석 중...")
                
                # 감정 분석
                emotion_vector = self.analyze_emotion(user_input)
                
                # 결과 출력
                print(f"\n입력 가사: '{user_input}'")
                self.print_emotion_analysis(emotion_vector)
                
                # 추천
                recommendations = self.get_recommendations(user_input, emotion_vector)
                self.print_recommendations(recommendations)
                
                print("=" * 50)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 데모를 종료합니다. 감사합니다!")
                break
            except Exception as e:
                print(f"❌ 오류가 발생했습니다: {e}")
                print("다시 시도해 주세요.\n")

def main():
    """메인 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='감정 분석 기반 노래 추천 시스템 데모')
    parser.add_argument('--model_path', type=str, default=None,
                       help='사전 훈련된 모델 체크포인트 경로')
    parser.add_argument('--sample_size', type=int, default=100,
                       help='샘플 데이터 크기 (새로 훈련하는 경우)')
    parser.add_argument('--test_mode', action='store_true',
                       help='테스트 모드 (샘플 쿼리들만 실행)')
    
    args = parser.parse_args()
    
    # 데모 초기화
    demo = EmotionRecommendationDemo(args.model_path, args.sample_size)
    
    if args.test_mode:
        # 테스트 모드: 미리 정의된 쿼리들로 테스트
        test_queries = [
            "사랑하는 마음이 너무 커서 행복해요",
            "이별의 아픔이 너무 슬퍼요",
            "화가 나서 참을 수 없어요",
            "무서운 밤이 두려워요"
        ]
        
        print("🧪 테스트 모드: 샘플 쿼리들을 실행합니다.")
        print()
        
        for i, query in enumerate(test_queries, 1):
            print(f"테스트 {i}: '{query}'")
            emotion_vector = demo.analyze_emotion(query)
            demo.print_emotion_analysis(emotion_vector)
            
            recommendations = demo.get_recommendations(query, emotion_vector, top_k=3)
            demo.print_recommendations(recommendations)
            
            print("=" * 50)
            print()
    else:
        # 대화형 모드
        demo.run_interactive_demo()

if __name__ == "__main__":
    main()