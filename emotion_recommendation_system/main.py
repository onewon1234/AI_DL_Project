#!/usr/bin/env python3
"""
감정 분석 기반 노래 추천 시스템 메인 실행 파일

이 스크립트는 다음 기능들을 제공합니다:
1. 노래 가사 데이터 전처리
2. RoBERTa 기반 감정 분류 모델 훈련
3. BERT 기반 문맥 벡터 생성
4. 유사도 기반 추천 시스템 구축
5. 추천 결과 시각화 및 평가
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import torch
from torch.utils.data import DataLoader

from data_processor import DataProcessor, create_sample_data
from emotion_model import EmotionClassificationModel
from context_encoder import ContextEncoder, RecommendationEngine
from training_pipeline import TrainingPipeline, load_model_from_checkpoint
import matplotlib.pyplot as plt
import seaborn as sns

def create_config(args) -> Dict:
    """명령줄 인자로부터 설정 생성"""
    config = {
        'model_name': args.model_name,
        'context_model_name': args.context_model_name,
        'num_emotions': 6,
        'dropout_rate': args.dropout_rate,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'loss_type': args.loss_type,
        'batch_size': args.batch_size,
        'max_length': args.max_length,
        'num_epochs': args.num_epochs,
        'use_scheduler': True,
        'early_stop_patience': args.early_stop_patience,
        'use_wandb': args.use_wandb,
        'wandb_project': args.wandb_project,
        'use_context_encoder': True,
    }
    return config

def load_or_create_data(data_path: str, num_samples: int = 1000) -> tuple:
    """데이터 로드 또는 샘플 데이터 생성"""
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}")
        processor = DataProcessor()
        lyrics, emotion_ratios = processor.load_data_from_csv(data_path)
        print(f"Loaded {len(lyrics)} songs")
    else:
        print(f"Creating sample data with {num_samples} songs")
        lyrics, emotion_ratios = create_sample_data(num_samples)
        
        # 샘플 데이터를 CSV로 저장
        emotion_labels = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']
        df = pd.DataFrame(emotion_ratios, columns=emotion_labels)
        df.insert(0, 'lyrics', lyrics)
        
        sample_data_path = '/workspace/emotion_recommendation_system/sample_data.csv'
        df.to_csv(sample_data_path, index=False, encoding='utf-8')
        print(f"Sample data saved to {sample_data_path}")
    
    return lyrics, emotion_ratios

def train_model(config: Dict, train_loader: DataLoader, val_loader: DataLoader, 
                save_dir: str) -> TrainingPipeline:
    """모델 훈련"""
    print("\n" + "="*50)
    print("TRAINING EMOTION CLASSIFICATION MODEL")
    print("="*50)
    
    # 훈련 파이프라인 생성
    pipeline = TrainingPipeline(config)
    
    # 훈련 실행
    history = pipeline.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config['num_epochs'],
        save_dir=save_dir
    )
    
    # 훈련 결과 시각화 저장
    pipeline.save_training_plots(save_dir)
    
    return pipeline

def build_recommendation_system(pipeline: TrainingPipeline, lyrics: List[str], 
                              emotion_predictions: np.ndarray, 
                              metadata: Optional[List[Dict]] = None) -> RecommendationEngine:
    """추천 시스템 구축"""
    print("\n" + "="*50)
    print("BUILDING RECOMMENDATION SYSTEM")
    print("="*50)
    
    # 메타데이터 생성 (없는 경우)
    if metadata is None:
        metadata = [{'song_id': i, 'title': f'Song {i+1}'} for i in range(len(lyrics))]
    
    # 추천 시스템 생성
    recommendation_engine = pipeline.create_recommendation_system(
        lyrics_list=lyrics,
        emotion_predictions=emotion_predictions,
        metadata_list=metadata
    )
    
    return recommendation_engine

def demo_recommendation_system(recommendation_engine: RecommendationEngine, 
                             lyrics: List[str], save_dir: str):
    """추천 시스템 데모"""
    print("\n" + "="*50)
    print("RECOMMENDATION SYSTEM DEMO")
    print("="*50)
    
    # 샘플 쿼리들
    sample_queries = [
        "사랑하는 마음이 너무 커서 행복해요",
        "이별의 아픔이 너무 슬퍼요", 
        "화가 나서 참을 수 없어요",
        "무서운 밤이 두려워요"
    ]
    
    demo_results = []
    
    for i, query_lyrics in enumerate(sample_queries):
        print(f"\n--- Query {i+1}: {query_lyrics} ---")
        
        # 감정 예측 (실제로는 훈련된 모델로 예측해야 하지만, 여기서는 샘플 사용)
        # 각 쿼리에 맞는 대표적인 감정 분포 생성
        if i == 0:  # 행복
            query_emotion = np.array([0.7, 0.1, 0.05, 0.05, 0.05, 0.05])
        elif i == 1:  # 슬픔
            query_emotion = np.array([0.1, 0.7, 0.05, 0.05, 0.05, 0.05])
        elif i == 2:  # 분노
            query_emotion = np.array([0.05, 0.1, 0.7, 0.05, 0.05, 0.05])
        else:  # 두려움
            query_emotion = np.array([0.05, 0.1, 0.05, 0.7, 0.05, 0.05])
        
        print(f"Predicted emotion: {query_emotion}")
        
        # 추천 결과
        recommendations = recommendation_engine.recommend_by_lyrics(
            query_lyrics=query_lyrics,
            predicted_emotion=query_emotion,
            top_k=5,
            context_weight=0.7
        )
        
        print(f"Top 5 recommendations:")
        for j, rec in enumerate(recommendations):
            print(f"  {j+1}. Score: {rec['similarity_score']:.4f}")
            print(f"     Lyrics: {lyrics[rec['index']][:50]}...")
            print(f"     Emotion: {rec['emotion_vector']}")
        
        # 결과 저장
        demo_results.append({
            'query': query_lyrics,
            'query_emotion': query_emotion.tolist(),
            'recommendations': [
                {
                    'index': rec['index'],
                    'score': rec['similarity_score'],
                    'lyrics': lyrics[rec['index']],
                    'emotion': rec['emotion_vector'].tolist()
                } for rec in recommendations
            ]
        })
    
    # 결과를 JSON으로 저장
    with open(os.path.join(save_dir, 'demo_results.json'), 'w', encoding='utf-8') as f:
        json.dump(demo_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nDemo results saved to {save_dir}/demo_results.json")

def visualize_emotion_distributions(emotion_predictions: np.ndarray, save_dir: str):
    """감정 분포 시각화"""
    print("\n" + "="*50)
    print("VISUALIZING EMOTION DISTRIBUTIONS")
    print("="*50)
    
    emotion_labels = ['Joy', 'Sadness', 'Anger', 'Fear', 'Surprise', 'Disgust']
    
    # 평균 감정 분포
    mean_emotions = np.mean(emotion_predictions, axis=0)
    
    plt.figure(figsize=(15, 5))
    
    # 1. 평균 감정 분포 바 차트
    plt.subplot(1, 3, 1)
    bars = plt.bar(emotion_labels, mean_emotions, color='skyblue')
    plt.title('Average Emotion Distribution')
    plt.ylabel('Probability')
    plt.xticks(rotation=45)
    
    # 값 표시
    for bar, value in zip(bars, mean_emotions):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 2. 감정 분포 히스토그램
    plt.subplot(1, 3, 2)
    for i, emotion in enumerate(emotion_labels):
        plt.hist(emotion_predictions[:, i], alpha=0.7, label=emotion, bins=20)
    plt.title('Emotion Distribution Histograms')
    plt.xlabel('Probability')
    plt.ylabel('Frequency')
    plt.legend()
    
    # 3. 감정 간 상관관계 히트맵
    plt.subplot(1, 3, 3)
    correlation_matrix = np.corrcoef(emotion_predictions.T)
    sns.heatmap(correlation_matrix, 
                xticklabels=emotion_labels, 
                yticklabels=emotion_labels,
                annot=True, cmap='coolwarm', center=0)
    plt.title('Emotion Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'emotion_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Emotion analysis plots saved to {save_dir}/emotion_analysis.png")

def evaluate_recommendation_system(recommendation_engine: RecommendationEngine,
                                 lyrics: List[str], emotion_predictions: np.ndarray,
                                 save_dir: str):
    """추천 시스템 평가"""
    print("\n" + "="*50)
    print("EVALUATING RECOMMENDATION SYSTEM")
    print("="*50)
    
    # 다양한 유사도 측정 방법 비교
    num_test_queries = min(10, len(lyrics))
    test_indices = np.random.choice(len(lyrics), num_test_queries, replace=False)
    
    similarity_methods = ['cosine', 'kl_divergence', 'js_divergence']
    context_weights = [0.5, 0.7, 0.9]
    
    evaluation_results = {}
    
    for method in similarity_methods:
        for weight in context_weights:
            key = f"{method}_weight_{weight}"
            similarities = []
            
            for idx in test_indices:
                query_lyrics = lyrics[idx]
                query_emotion = emotion_predictions[idx]
                
                if method == 'cosine':
                    # 하이브리드 추천 사용
                    recommendations = recommendation_engine.recommend_by_lyrics(
                        query_lyrics=query_lyrics,
                        predicted_emotion=query_emotion,
                        top_k=5,
                        context_weight=weight
                    )
                    # 첫 번째 추천의 유사도 점수 사용
                    if recommendations:
                        similarities.append(recommendations[0]['similarity_score'])
                else:
                    # 감정만 기반 추천
                    recommendations = recommendation_engine.recommend_by_emotion_only(
                        query_emotion=query_emotion,
                        top_k=5,
                        method=method
                    )
                    if recommendations:
                        similarities.append(recommendations[0]['similarity_score'])
            
            evaluation_results[key] = {
                'mean_similarity': np.mean(similarities) if similarities else 0,
                'std_similarity': np.std(similarities) if similarities else 0,
                'num_queries': len(similarities)
            }
    
    # 결과 시각화
    plt.figure(figsize=(12, 8))
    
    methods_data = []
    for key, result in evaluation_results.items():
        parts = key.split('_')
        method = parts[0]
        weight = float(parts[2]) if len(parts) > 2 else 0.0
        methods_data.append({
            'method': method,
            'weight': weight,
            'mean_similarity': result['mean_similarity']
        })
    
    df_eval = pd.DataFrame(methods_data)
    
    # 방법별 성능 비교
    plt.subplot(2, 1, 1)
    method_means = df_eval.groupby('method')['mean_similarity'].mean()
    bars = plt.bar(method_means.index, method_means.values, color=['lightblue', 'lightgreen', 'lightcoral'])
    plt.title('Average Similarity by Method')
    plt.ylabel('Mean Similarity Score')
    
    for bar, value in zip(bars, method_means.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # Context weight별 성능 (cosine만)
    plt.subplot(2, 1, 2)
    cosine_data = df_eval[df_eval['method'] == 'cosine']
    if not cosine_data.empty:
        plt.plot(cosine_data['weight'], cosine_data['mean_similarity'], 'o-', linewidth=2, markersize=8)
        plt.title('Performance by Context Weight (Cosine Similarity)')
        plt.xlabel('Context Weight')
        plt.ylabel('Mean Similarity Score')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'recommendation_evaluation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 결과를 JSON으로 저장
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    print(f"Evaluation results saved to {save_dir}/evaluation_results.json")
    print(f"Evaluation plots saved to {save_dir}/recommendation_evaluation.png")

def main():
    parser = argparse.ArgumentParser(description='Emotion-based Music Recommendation System')
    
    # 데이터 관련
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to lyrics dataset CSV file')
    parser.add_argument('--num_samples', type=int, default=1000,
                       help='Number of sample songs to generate if no data file provided')
    
    # 모델 관련
    parser.add_argument('--model_name', type=str, default='klue/roberta-base',
                       help='RoBERTa model name')
    parser.add_argument('--context_model_name', type=str, default='klue/bert-base',
                       help='BERT model name for context encoding')
    parser.add_argument('--loss_type', type=str, default='kl_divergence',
                       choices=['kl_divergence', 'js_divergence', 'emd', 'focal', 'mse'],
                       help='Loss function type')
    
    # 훈련 관련
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--dropout_rate', type=float, default=0.1,
                       help='Dropout rate')
    parser.add_argument('--early_stop_patience', type=int, default=5,
                       help='Early stopping patience')
    
    # 실행 관련
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                       help='Path to model checkpoint to load')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training and only run recommendation demo')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='emotion-music-recommendation',
                       help='WandB project name')
    
    args = parser.parse_args()
    
    # 설정 생성
    config = create_config(args)
    
    # 결과 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 설정 저장
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Emotion-based Music Recommendation System")
    print("="*50)
    print(f"Configuration saved to {args.save_dir}/config.json")
    
    # 1. 데이터 로드/생성
    lyrics, emotion_ratios = load_or_create_data(args.data_path, args.num_samples)
    
    # 2. 데이터 전처리
    processor = DataProcessor(model_name=config['model_name'])
    
    # 데이터 분할
    (train_lyrics, train_emotions), (val_lyrics, val_emotions), (test_lyrics, test_emotions) = \
        processor.split_data(lyrics, emotion_ratios, train_ratio=0.8, val_ratio=0.1)
    
    # 데이터셋 및 데이터로더 생성
    train_dataset = processor.create_dataset(train_lyrics, train_emotions, config['max_length'])
    val_dataset = processor.create_dataset(val_lyrics, val_emotions, config['max_length'])
    test_dataset = processor.create_dataset(test_lyrics, test_emotions, config['max_length'])
    
    train_loader = processor.create_dataloader(train_dataset, config['batch_size'], shuffle=True)
    val_loader = processor.create_dataloader(val_dataset, config['batch_size'], shuffle=False)
    test_loader = processor.create_dataloader(test_dataset, config['batch_size'], shuffle=False)
    
    print(f"Data split: Train={len(train_lyrics)}, Val={len(val_lyrics)}, Test={len(test_lyrics)}")
    
    # 3. 모델 훈련 또는 로드
    if args.skip_training and args.checkpoint_path:
        print("Loading model from checkpoint...")
        pipeline = load_model_from_checkpoint(args.checkpoint_path, config)
    elif not args.skip_training:
        pipeline = train_model(config, train_loader, val_loader, args.save_dir)
    else:
        print("Training new model...")
        pipeline = train_model(config, train_loader, val_loader, args.save_dir)
    
    # 4. 전체 데이터셋에 대한 감정 예측
    all_dataset = processor.create_dataset(lyrics, emotion_ratios, config['max_length'])
    all_loader = processor.create_dataloader(all_dataset, config['batch_size'], shuffle=False)
    
    emotion_predictions, predicted_texts = pipeline.predict_emotions(all_loader)
    
    # 5. 추천 시스템 구축
    recommendation_engine = build_recommendation_system(pipeline, lyrics, emotion_predictions)
    
    # 6. 감정 분포 시각화
    visualize_emotion_distributions(emotion_predictions, args.save_dir)
    
    # 7. 추천 시스템 데모
    demo_recommendation_system(recommendation_engine, lyrics, args.save_dir)
    
    # 8. 추천 시스템 평가
    evaluate_recommendation_system(recommendation_engine, lyrics, emotion_predictions, args.save_dir)
    
    # 9. 추천 엔진 저장
    recommendation_engine.save_model(os.path.join(args.save_dir, 'recommendation_engine.pkl'))
    
    print("\n" + "="*50)
    print("SYSTEM COMPLETE!")
    print("="*50)
    print(f"All results saved to: {args.save_dir}")
    print("Files generated:")
    print("  - config.json: Configuration settings")
    print("  - best_model.pt: Trained emotion classification model")
    print("  - recommendation_engine.pkl: Complete recommendation system")
    print("  - training_curves.png: Training progress visualization")
    print("  - emotion_analysis.png: Emotion distribution analysis")
    print("  - recommendation_evaluation.png: Recommendation system evaluation")
    print("  - demo_results.json: Recommendation demo results")
    print("  - evaluation_results.json: Quantitative evaluation results")

if __name__ == "__main__":
    main()