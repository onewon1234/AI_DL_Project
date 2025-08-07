# 감정 분석 기반 노래 추천 시스템

RoBERTa와 BERT를 활용한 한국어 노래 가사 기반 감정 분석 및 추천 시스템입니다.

## 🎵 주요 기능

### 1. 감정 분류 모델
- **모델**: KLUE-RoBERTa-base 기반
- **출력**: 6개 감정군별 비율 예측 (기쁨, 슬픔, 분노, 두려움, 놀라움, 혐오)
- **손실 함수**: KL Divergence, Jensen-Shannon Divergence, Earth Mover's Distance, Focal Loss, MSE 지원

### 2. 문맥 벡터 생성
- **모델**: KLUE-BERT-base 또는 Sentence-BERT (ko-sroberta-multitask)
- **기능**: 가사의 의미적 문맥을 벡터로 인코딩

### 3. 하이브리드 추천 시스템
- **문맥 유사도**: Sentence-BERT 임베딩 기반 코사인 유사도
- **감정 유사도**: 감정 분포 간 KL/JS Divergence 또는 코사인 유사도
- **FAISS 인덱스**: 빠른 대규모 검색 지원

## 📁 프로젝트 구조

```
emotion_recommendation_system/
├── data_processor.py          # 데이터 전처리 및 토큰화
├── emotion_model.py           # RoBERTa 감정 분류 모델
├── context_encoder.py         # BERT 문맥 벡터 생성
├── training_pipeline.py       # 훈련 파이프라인
├── main.py                   # 메인 실행 파일
├── requirements.txt          # 의존성 패키지
├── README.md                # 프로젝트 설명서
└── demo.py                  # 간단한 데모 스크립트
```

## 🛠️ 설치 및 환경 설정

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 필수 패키지
- torch >= 1.9.0
- transformers >= 4.20.0
- sentence-transformers >= 2.2.0
- faiss-cpu >= 1.7.0
- scikit-learn >= 1.0.0
- pandas, numpy, matplotlib, seaborn

## 🚀 사용법

### 1. 기본 실행 (샘플 데이터)
```bash
python main.py --num_samples 1000 --num_epochs 5
```

### 2. 커스텀 데이터 사용
```bash
python main.py --data_path your_data.csv --num_epochs 10
```

#### 데이터 형식 (CSV)
```csv
lyrics,joy,sadness,anger,fear,surprise,disgust
"사랑하는 마음이 너무 커서 행복해요",0.7,0.1,0.05,0.05,0.05,0.05
"이별의 아픔이 너무 슬퍼요",0.1,0.7,0.05,0.05,0.05,0.05
```

### 3. 고급 설정
```bash
python main.py \
  --model_name klue/roberta-base \
  --context_model_name klue/bert-base \
  --loss_type kl_divergence \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --num_epochs 20 \
  --save_dir ./results \
  --use_wandb
```

### 4. 사전 훈련된 모델 로드
```bash
python main.py \
  --checkpoint_path ./results/best_model.pt \
  --skip_training
```

## 📊 출력 파일

훈련 완료 후 다음 파일들이 생성됩니다:

- `config.json`: 모델 설정
- `best_model.pt`: 최고 성능 모델 체크포인트
- `recommendation_engine.pkl`: 완전한 추천 시스템
- `training_curves.png`: 훈련 과정 시각화
- `emotion_analysis.png`: 감정 분포 분석
- `recommendation_evaluation.png`: 추천 성능 평가
- `demo_results.json`: 추천 데모 결과
- `evaluation_results.json`: 정량적 평가 결과

## 🎯 손실 함수 비교

### KL Divergence (기본값)
- 확률 분포 간 차이 측정
- 비대칭적, 정보 이론 기반
- 감정 분포 예측에 적합

### Jensen-Shannon Divergence
- KL Divergence의 대칭적 버전
- 더 안정적인 훈련
- 0~1 범위로 정규화

### Earth Mover's Distance
- 감정 간 순서적 관계 고려
- 유사한 감정 간 거리 최소화
- 회귀 문제에 효과적

### Focal Loss
- 클래스 불균형 문제 해결
- 어려운 샘플에 집중
- 희귀 감정 분류 향상

## 🔍 추천 알고리즘

### 1. 문맥 기반 추천
```python
# 가사의 의미적 유사도만 사용
recommendations = engine.recommend_by_lyrics(
    query_lyrics="사랑하는 마음이 커요",
    predicted_emotion=emotion_vector,
    context_weight=1.0  # 문맥만 사용
)
```

### 2. 감정 기반 추천
```python
# 감정 분포 유사도만 사용
recommendations = engine.recommend_by_emotion_only(
    query_emotion=emotion_vector,
    method='kl_divergence'
)
```

### 3. 하이브리드 추천 (권장)
```python
# 문맥과 감정을 결합
recommendations = engine.recommend_by_lyrics(
    query_lyrics="사랑하는 마음이 커요",
    predicted_emotion=emotion_vector,
    context_weight=0.7  # 문맥 70%, 감정 30%
)
```

## 📈 성능 평가

시스템은 다음 메트릭으로 평가됩니다:

### 감정 분류 성능
- **MSE/MAE**: 감정 비율 예측 정확도
- **Pearson 상관계수**: 예측-실제 감정 상관관계
- **KL/JS Divergence**: 분포 차이 측정

### 추천 성능
- **유사도 점수**: 상위 추천 항목들의 평균 유사도
- **다양성**: 추천 결과의 감정적 다양성
- **적중률**: 사용자 선호와의 일치도

## 🎮 데모 스크립트

간단한 대화형 데모를 실행하려면:

```bash
python demo.py
```

이 스크립트는 사용자 입력을 받아 실시간으로 감정을 분석하고 유사한 곡들을 추천합니다.

## 🔧 커스터마이징

### 1. 새로운 손실 함수 추가
`emotion_model.py`의 `LossFunction` 클래스에 새 메서드를 추가하세요.

### 2. 감정 카테고리 변경
기본 6개 감정 외에 다른 감정 체계를 사용하려면 `DataProcessor`의 `emotion_labels`를 수정하세요.

### 3. 다른 언어 지원
모델 이름을 해당 언어의 BERT/RoBERTa 모델로 변경하세요:
```python
--model_name bert-base-multilingual-cased
--context_model_name sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

## 📚 참고 문헌

- [KLUE: Korean Language Understanding Evaluation](https://arxiv.org/abs/2105.09680)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License

## 📞 문의사항

프로젝트에 대한 문의나 개선 제안이 있으시면 이슈를 등록해 주세요.