import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional, Union
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pickle
import os

class ContextEncoder:
    """BERT 기반 문맥 벡터 생성기"""
    
    def __init__(self, model_name: str = "klue/bert-base", 
                 sentence_transformer_model: str = "jhgan/ko-sroberta-multitask"):
        """
        Args:
            model_name: BERT 모델 이름
            sentence_transformer_model: Sentence-BERT 모델 이름 (한국어 지원)
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # BERT 모델 및 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert_model = AutoModel.from_pretrained(model_name).to(self.device)
        self.bert_model.eval()
        
        # Sentence-BERT 모델 로드 (더 효과적인 문장 임베딩)
        self.sentence_model = SentenceTransformer(sentence_transformer_model)
        
    def encode_with_bert(self, texts: List[str], max_length: int = 512,
                        pooling_strategy: str = 'cls') -> np.ndarray:
        """
        BERT를 사용하여 텍스트를 인코딩
        
        Args:
            texts: 인코딩할 텍스트 리스트
            max_length: 최대 시퀀스 길이
            pooling_strategy: 풀링 전략 ('cls', 'mean', 'max')
            
        Returns:
            문맥 벡터 배열 [num_texts, hidden_dim]
        """
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                # 토큰화
                inputs = self.tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    padding='max_length',
                    max_length=max_length
                ).to(self.device)
                
                # BERT 인코딩
                outputs = self.bert_model(**inputs)
                hidden_states = outputs.last_hidden_state  # [1, seq_len, hidden_dim]
                attention_mask = inputs['attention_mask']
                
                # 풀링 적용
                if pooling_strategy == 'cls':
                    # [CLS] 토큰 사용
                    pooled = hidden_states[:, 0, :]  # [1, hidden_dim]
                elif pooling_strategy == 'mean':
                    # 마스크된 평균
                    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    pooled = sum_embeddings / sum_mask
                elif pooling_strategy == 'max':
                    # 최대값 풀링
                    pooled = torch.max(hidden_states, dim=1)[0]
                else:
                    raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
                
                embeddings.append(pooled.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def encode_with_sentence_transformer(self, texts: List[str]) -> np.ndarray:
        """
        Sentence-BERT를 사용하여 텍스트를 인코딩 (더 효과적)
        
        Args:
            texts: 인코딩할 텍스트 리스트
            
        Returns:
            문맥 벡터 배열 [num_texts, embedding_dim]
        """
        return self.sentence_model.encode(texts, convert_to_numpy=True)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32,
                    use_sentence_transformer: bool = True) -> np.ndarray:
        """
        배치 단위로 텍스트 인코딩
        
        Args:
            texts: 인코딩할 텍스트 리스트
            batch_size: 배치 크기
            use_sentence_transformer: Sentence-BERT 사용 여부
            
        Returns:
            문맥 벡터 배열
        """
        if use_sentence_transformer:
            # Sentence-BERT는 내부적으로 배치 처리 지원
            return self.encode_with_sentence_transformer(texts)
        else:
            # BERT 배치 처리
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_embeddings = self.encode_with_bert(batch_texts)
                all_embeddings.append(batch_embeddings)
            return np.vstack(all_embeddings)

class SimilarityCalculator:
    """유사도 계산 및 검색을 위한 클래스"""
    
    def __init__(self, context_vectors: np.ndarray, emotion_vectors: np.ndarray,
                 metadata: Optional[List[Dict]] = None):
        """
        Args:
            context_vectors: 문맥 벡터 [num_songs, context_dim]
            emotion_vectors: 감정 벡터 [num_songs, 6]
            metadata: 각 노래의 메타데이터 (제목, 아티스트 등)
        """
        self.context_vectors = context_vectors
        self.emotion_vectors = emotion_vectors
        self.metadata = metadata or [{} for _ in range(len(context_vectors))]
        
        # 벡터 정규화
        self.normalized_context = self._normalize_vectors(context_vectors)
        self.normalized_emotion = self._normalize_vectors(emotion_vectors)
        
        # FAISS 인덱스 구축 (빠른 검색을 위해)
        self.context_index = None
        self.emotion_index = None
        self.combined_index = None
        self._build_faiss_indices()
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """벡터 정규화"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # 0으로 나누기 방지
        return vectors / norms
    
    def _build_faiss_indices(self):
        """FAISS 인덱스 구축"""
        # 문맥 벡터 인덱스
        context_dim = self.normalized_context.shape[1]
        self.context_index = faiss.IndexFlatIP(context_dim)  # Inner Product (cosine for normalized vectors)
        self.context_index.add(self.normalized_context.astype('float32'))
        
        # 감정 벡터 인덱스
        emotion_dim = self.normalized_emotion.shape[1]
        self.emotion_index = faiss.IndexFlatIP(emotion_dim)
        self.emotion_index.add(self.normalized_emotion.astype('float32'))
        
        # 결합 벡터 인덱스
        combined_vectors = np.hstack([self.normalized_context, self.normalized_emotion])
        combined_dim = combined_vectors.shape[1]
        self.combined_index = faiss.IndexFlatIP(combined_dim)
        self.combined_index.add(combined_vectors.astype('float32'))
    
    def find_similar_by_context(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """문맥 유사도 기반 검색"""
        query_normalized = self._normalize_vectors(query_vector.reshape(1, -1))
        scores, indices = self.context_index.search(query_normalized.astype('float32'), top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def find_similar_by_emotion(self, query_emotion: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """감정 유사도 기반 검색"""
        query_normalized = self._normalize_vectors(query_emotion.reshape(1, -1))
        scores, indices = self.emotion_index.search(query_normalized.astype('float32'), top_k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def find_similar_hybrid(self, query_context: np.ndarray, query_emotion: np.ndarray,
                           context_weight: float = 0.7, top_k: int = 10) -> List[Tuple[int, float]]:
        """문맥과 감정을 결합한 하이브리드 검색"""
        # 개별 유사도 계산
        context_similarities = cosine_similarity(
            self._normalize_vectors(query_context.reshape(1, -1)),
            self.normalized_context
        )[0]
        
        emotion_similarities = cosine_similarity(
            self._normalize_vectors(query_emotion.reshape(1, -1)),
            self.normalized_emotion
        )[0]
        
        # 가중 결합
        combined_similarities = (context_weight * context_similarities + 
                               (1 - context_weight) * emotion_similarities)
        
        # 상위 k개 선택
        top_indices = np.argsort(combined_similarities)[::-1][:top_k]
        return [(int(idx), float(combined_similarities[idx])) for idx in top_indices]
    
    def get_emotion_distribution_similarity(self, query_emotion: np.ndarray, 
                                          method: str = 'cosine') -> np.ndarray:
        """감정 분포 유사도 계산"""
        if method == 'cosine':
            return cosine_similarity(query_emotion.reshape(1, -1), self.emotion_vectors)[0]
        elif method == 'kl_divergence':
            # KL divergence (작을수록 유사)
            eps = 1e-8
            query_emotion = query_emotion + eps
            emotion_vectors = self.emotion_vectors + eps
            
            kl_divs = []
            for emotion_vec in emotion_vectors:
                kl_div = np.sum(query_emotion * np.log(query_emotion / emotion_vec))
                kl_divs.append(-kl_div)  # 음수로 변환 (클수록 유사)
            return np.array(kl_divs)
        elif method == 'js_divergence':
            # Jensen-Shannon divergence
            js_divs = []
            for emotion_vec in emotion_vectors:
                m = 0.5 * (query_emotion + emotion_vec)
                js_div = 0.5 * np.sum(query_emotion * np.log(query_emotion / m)) + \
                         0.5 * np.sum(emotion_vec * np.log(emotion_vec / m))
                js_divs.append(-js_div)  # 음수로 변환
            return np.array(js_divs)
        else:
            raise ValueError(f"Unsupported similarity method: {method}")

class RecommendationEngine:
    """추천 엔진 통합 클래스"""
    
    def __init__(self, context_encoder: ContextEncoder, similarity_calculator: SimilarityCalculator):
        self.context_encoder = context_encoder
        self.similarity_calculator = similarity_calculator
    
    def recommend_by_lyrics(self, query_lyrics: str, predicted_emotion: np.ndarray,
                           top_k: int = 10, context_weight: float = 0.7) -> List[Dict]:
        """가사 기반 추천"""
        # 쿼리 가사의 문맥 벡터 생성
        query_context = self.context_encoder.encode_with_sentence_transformer([query_lyrics])[0]
        
        # 하이브리드 유사도 계산
        similar_indices = self.similarity_calculator.find_similar_hybrid(
            query_context, predicted_emotion, context_weight, top_k
        )
        
        # 결과 포맷팅
        recommendations = []
        for idx, score in similar_indices:
            recommendation = {
                'index': idx,
                'similarity_score': score,
                'context_vector': self.similarity_calculator.context_vectors[idx],
                'emotion_vector': self.similarity_calculator.emotion_vectors[idx],
                'metadata': self.similarity_calculator.metadata[idx]
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def recommend_by_emotion_only(self, query_emotion: np.ndarray, 
                                 top_k: int = 10, method: str = 'cosine') -> List[Dict]:
        """감정만 기반 추천"""
        similarities = self.similarity_calculator.get_emotion_distribution_similarity(
            query_emotion, method
        )
        
        # 상위 k개 선택
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        recommendations = []
        for idx in top_indices:
            recommendation = {
                'index': idx,
                'similarity_score': similarities[idx],
                'emotion_vector': self.similarity_calculator.emotion_vectors[idx],
                'metadata': self.similarity_calculator.metadata[idx]
            }
            recommendations.append(recommendation)
        
        return recommendations
    
    def save_model(self, save_path: str):
        """모델 저장"""
        save_data = {
            'context_vectors': self.similarity_calculator.context_vectors,
            'emotion_vectors': self.similarity_calculator.emotion_vectors,
            'metadata': self.similarity_calculator.metadata,
            'context_encoder_model': self.context_encoder.model_name
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
    
    @classmethod
    def load_model(cls, load_path: str, context_encoder_model: Optional[str] = None):
        """모델 로드"""
        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Context encoder 재생성
        model_name = context_encoder_model or save_data['context_encoder_model']
        context_encoder = ContextEncoder(model_name)
        
        # Similarity calculator 재생성
        similarity_calculator = SimilarityCalculator(
            save_data['context_vectors'],
            save_data['emotion_vectors'],
            save_data['metadata']
        )
        
        return cls(context_encoder, similarity_calculator)

if __name__ == "__main__":
    # 테스트 코드
    print("Testing Context Encoder...")
    
    # 샘플 텍스트
    sample_texts = [
        "사랑하는 마음이 너무 커서 행복해요",
        "이별의 아픔이 너무 슬퍼요",
        "화가 나서 참을 수 없어요"
    ]
    
    # Context encoder 테스트
    encoder = ContextEncoder()
    
    # BERT 인코딩
    bert_embeddings = encoder.encode_with_bert(sample_texts)
    print(f"BERT embeddings shape: {bert_embeddings.shape}")
    
    # Sentence-BERT 인코딩
    sbert_embeddings = encoder.encode_with_sentence_transformer(sample_texts)
    print(f"Sentence-BERT embeddings shape: {sbert_embeddings.shape}")
    
    # 샘플 감정 벡터
    emotion_vectors = np.random.dirichlet(np.ones(6), size=3)
    print(f"Emotion vectors shape: {emotion_vectors.shape}")
    
    # Similarity calculator 테스트
    similarity_calc = SimilarityCalculator(sbert_embeddings, emotion_vectors)
    
    # 유사도 검색 테스트
    query_context = sbert_embeddings[0]
    query_emotion = emotion_vectors[0]
    
    similar_context = similarity_calc.find_similar_by_context(query_context, top_k=2)
    similar_emotion = similarity_calc.find_similar_by_emotion(query_emotion, top_k=2)
    similar_hybrid = similarity_calc.find_similar_hybrid(query_context, query_emotion, top_k=2)
    
    print(f"Similar by context: {similar_context}")
    print(f"Similar by emotion: {similar_emotion}")
    print(f"Similar by hybrid: {similar_hybrid}")