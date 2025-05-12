# 문맥 기반 문장 순서 예측 프로젝트

이 프로젝트는 [Dacon 문맥 기반 문장 순서 예측 AI 경진대회](https://dacon.io/competitions/official/236489/overview/description)에 참가하여, 주어진 문서의 문장들을 올바른 순서로 배열하는 딥러닝 모델을 개발하는 것을 목표로 합니다. 

## 프로젝트 개요

- **목표**: 주어진 4개의 문장을 올바른 순서로 재배열
- **접근법**:
  - Pairwise 방식: 문장 쌍 간 순서를 예측하여 전체 순서 결정
  - Seq2Seq 방식: 전체 문장을 입력받아 순서를 직접 생성
- **사용 모델**:
  - `KLUE-RoBERTa`
  - `KLUE-BERT`
  - `KoElectra`
  - `T5 (KE-T5)`

## 프로젝트 진행 계획

**1주차 – 초기 설계 및 데이터 분배**  
- 목표 설정, 방식 결정 (Pairwise / Seq2Seq)  
- 데이터 구조 파악 및 모델 담당자 배정

**2~3주차 – 전처리 및 모델 학습**  
- 공통 전처리 진행  
- 각자 모델별 학습 및 실험 (`KLUE-RoBERTa`, `KLUE-BERT`, `KoElectra`, `T5`)  
- 실험 결과 비교

**4주차 – 모델 선정 및 추가 실험**  
- 성능 우수 모델 선정  
- 파인튜닝 및 추가 실험 진행

**5주차 – 발표 자료 작성**  
- 결과 정리 및 PPT 제작  
- 발표 준비 및 리허설

## 폴더 구조

```
📂 data/           # 원본 데이터
📂 preprocess/     # 전처리 코드 + 공통 데이터 저장
📂 KLUE-RoBERTA/   # Pairwise 방식 실험
📂 KLUE-BERT/      # Pairwise 방식 실험
📂 KoElectra/      # Pairwise 방식 실험
📂 T5/             # Seq2Seq 방식 실험
```

## 데이터셋

- 제공 파일:
  - `train.csv`: 학습용 데이터, 총 **7,350개 샘플**
  - `test.csv`: 테스트용 데이터, 총 **1,780개 샘플**
- 각 샘플은 4개의 문장(`sentence_0` ~ `sentence_3`)으로 구성
- `train.csv`에는 정답 순서를 나타내는 `answer` 컬럼 포함
- `test.csv`에는 정답이 없으며, 예측 결과를 `submission.csv`로 제출해야 함

## 전처리

- **Pairwise 모델용**: 각 샘플의 문장 4개에서 가능한 문장 순서쌍 12개를 생성하고, 올바른 순서를 기준으로 label 부여 (예: A가 B보다 앞이면 `1`, 아니면 `0`)  
- **Seq2Seq 모델용**: 4개의 문장을 하나의 입력으로 합치고, 정답 순서를 문자열로 생성 (예: `"2 0 1 3"`)

## 모델링 전략

### 1. Pairwise Ranking (KLUE-RoBERTa, KLUE-BERT, KoElectra)
- 입력: 문장 A, 문장 B → 두 문장의 순서를 비교
- 학습: 각 쌍에 대해 "A가 B보다 앞인가?"를 이진 분류
- 예측: 12개 쌍의 예측을 바탕으로 가능한 문장 순서 조합 중 가장 일관된 조합 선택

### 2. Seq2Seq Prediction (T5)
- 입력: `[CLS] 문장 0 [SEP] 문장 1 [SEP] 문장 2 [SEP] 문장 3`
- 출력: `"2 0 1 3"` 형태의 순서 문자열
- 사전학습된 T5 모델(`KE-T5`) 기반 fine-tuning

## 평가 방법

- 정확한 순서를 맞춘 sample 비율 (Accuracy)
- 제출 형식: `submission.csv` (`id`, `pred`) 형태