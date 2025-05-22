# KETI-AIR/ke-t5-small-ko를 이용한 문장 순서 예측
# 이 스크립트는 KETI-AIR/ke-t5-small-ko 모델을 사용하여 문장 순서 예측을 구현합니다.
# 1. 데이터 로드 및 전처리
# 2. 모델 및 학습 설정
# 3. Early stopping과 체크포인팅을 적용한 학습
# 4. 모델 평가
# 5. 하이퍼파라미터 튜닝

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.optim import AdamW
from tqdm import tqdm
import os
from t5_utils import SentenceOrderPredictor, compute_accuracy

# 한글 폰트 설정 (matplotlib)
plt.rc('font', family='Malgun Gothic')

# 결과 저장을 위한 디렉토리 생성
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('history', exist_ok=True)
os.makedirs('grid_search_results', exist_ok=True)

# 1. 데이터 로드 및 준비
print("데이터 로드 중...")
df = pd.read_csv('../data/cleaned_seq2seq.csv')
print("데이터셋 크기:", len(df))
print("\n샘플 데이터:")
print(df.head(5))

df['input_text'] = "아래 문장들의 올바른 순서를 숫자로 예측하세요: " + df['input_text']

# KETI-AIR/ke-t5-small-ko 모델 초기화
predictor = SentenceOrderPredictor(model_name="KETI-AIR/ke-t5-small-ko")

# 데이터셋 준비 (9:1 비율로 분할)
train_dataset, val_dataset = predictor.prepare_data(df)

# 데이터로더 생성
batch_size = 8  # GPU 메모리에 따라 조정 가능
train_loader, val_loader = predictor.create_dataloaders(train_dataset, val_dataset, batch_size=batch_size)
print(f"학습 데이터 크기: {len(train_dataset)}")
print(f"검증 데이터 크기: {len(val_dataset)}")

# 2. 학습 설정
num_epochs = 10
learning_rate = 2e-5
patience = 3  # Early stopping 인내심
min_delta = 1e-4  # 최소 개선 기준

optimizer = AdamW(predictor.model.parameters(), lr=learning_rate)

history = {
    'train_loss': [],
    'val_loss': [],
    'val_accuracy': []
}

def validate(model, val_loader, device, tokenizer):
    """검증 데이터에 대한 모델 평가"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss = outputs.loss
            total_loss += loss.item()
            predictions = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=8,
                num_beams=4,
                no_repeat_ngram_size=4
            )
            pred_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in predictions]

            # 라벨은 디코딩하지 않고, pad 토큰(-100 또는 tokenizer.pad_token_id)만 제외
            label_orders = []
            for label in labels:
                label_ids = [id for id in label.tolist() if id != -100 and id != tokenizer.pad_token_id]
                label_orders.append(label_ids)

            try:
                pred_orders = [list(map(int, text.split())) for text in pred_texts]
                correct = sum(compute_accuracy(pred, label) for pred, label in zip(pred_orders, label_orders))
                total_correct += correct
                total_samples += len(input_ids)
            except ValueError as e:
                print(f"Warning: 잘못된 예측 형식 발견 - {e}")
                print(f"Predictions: {pred_texts}")
                continue
    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return avg_loss, accuracy

# 3. Early Stopping을 적용한 학습
best_val_loss = float('inf')
best_val_accuracy = 0
best_epoch = -1
patience_counter = 0

for epoch in range(num_epochs):
    predictor.model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} - Training'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(predictor.device)
        attention_mask = batch['attention_mask'].to(predictor.device)
        labels = batch['labels'].to(predictor.device)
        outputs = predictor.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    val_loss, val_accuracy = validate(predictor.model, val_loader, predictor.device, predictor.tokenizer)
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(val_loss)
    history['val_accuracy'].append(val_accuracy)
    print(f'Epoch {epoch+1}/{num_epochs}')
    print(f'Train Loss: {avg_train_loss:.4f}')
    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Validation Accuracy: {val_accuracy:.4f}')
    if val_loss < best_val_loss - min_delta:
        print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}')
        best_val_loss = val_loss
        best_val_accuracy = val_accuracy
        best_epoch = epoch
        patience_counter = 0
        predictor.save_checkpoint(
            epoch,
            predictor.model,
            optimizer,
            val_loss,
            f'checkpoints/keti_t5_best_model.pt'
        )
    else:
        patience_counter += 1
        print(f'No improvement for {patience_counter} epochs (best val_loss: {best_val_loss:.4f} at epoch {best_epoch+1})')
    if patience_counter >= patience:
        print(f'\nEarly stopping triggered after epoch {epoch+1}')
        print(f'Best validation loss: {best_val_loss:.4f}')
        print(f'Best validation accuracy: {best_val_accuracy:.4f}')
        print(f'Best epoch: {best_epoch+1}')
        break
print('\n학습 완료!')
print(f'Best epoch: {best_epoch+1}')
print(f'Best validation loss: {best_val_loss:.4f}')
print(f'Best validation accuracy: {best_val_accuracy:.4f}')
predictor.save_history(history, 'history/keti_t5_history.json')

# 4. 학습 결과 시각화
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train 및 Validation Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')
plt.legend()
plt.tight_layout()
plt.show()

# 5. 하이퍼파라미터 튜닝 함수
def train_with_params(learning_rate, batch_size, max_epochs, train_dataset, val_dataset, model_name="KETI-AIR/ke-t5-small-ko", use_auto_classes=False):
    """주어진 하이퍼파라미터로 KETI-AIR/ke-t5-small-ko 모델 학습"""
    predictor = SentenceOrderPredictor(model_name=model_name, use_auto_classes=use_auto_classes)
    optimizer = AdamW(predictor.model.parameters(), lr=learning_rate)
    train_loader, val_loader = predictor.create_dataloaders(train_dataset, val_dataset, batch_size=batch_size)
    best_val_loss = float('inf')
    best_val_accuracy = 0
    best_epoch = -1
    patience_counter = 0
    patience = 3
    min_delta = 1e-4
    for epoch in range(max_epochs):
        predictor.model.train()
        total_train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{max_epochs} - Training'):
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(predictor.device)
            attention_mask = batch['attention_mask'].to(predictor.device)
            labels = batch['labels'].to(predictor.device)
            outputs = predictor.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_accuracy = validate(predictor.model, val_loader, predictor.device, predictor.tokenizer)
        print(f'Epoch {epoch+1}/{max_epochs}')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {val_loss:.4f}')
        print(f'Validation Accuracy: {val_accuracy:.4f}')
        if val_loss < best_val_loss - min_delta:
            print(f'Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}')
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            print(f'No improvement for {patience_counter} epochs (best val_loss: {best_val_loss:.4f} at epoch {best_epoch+1})')
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    return best_val_loss, best_val_accuracy

def run_grid_search(train_dataset, val_dataset, model_name="KETI-AIR/ke-t5-small-ko", use_auto_classes=False):
    """KETI-AIR/ke-t5-small-ko 모델로 그리드 서치 수행"""
    param_grid = {
        'learning_rate': [1e-5, 2e-5, 5e-5],
        'batch_size': [4, 8, 16],
        'max_epochs': [5, 10]
    }
    results = []
    best_params = None
    best_accuracy = 0
    total_trials = len(param_grid['learning_rate']) * len(param_grid['batch_size']) * len(param_grid['max_epochs'])
    trial_count = 0
    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for epochs in param_grid['max_epochs']:
                trial_count += 1
                print(f'\n[Trial {trial_count}/{total_trials}]')
                print(f'Parameters: Learning Rate={lr}, Batch Size={bs}, Max Epochs={epochs}')
                val_loss, val_acc = train_with_params(
                    learning_rate=lr,
                    batch_size=bs,
                    max_epochs=epochs,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    model_name=model_name,
                    use_auto_classes=use_auto_classes
                )
                results.append({
                    'learning_rate': lr,
                    'batch_size': bs,
                    'max_epochs': epochs,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc
                })
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_params = {
                        'learning_rate': lr,
                        'batch_size': bs,
                        'max_epochs': epochs
                    }
                print(f'Results: Val Loss={val_loss:.4f}, Val Accuracy={val_acc:.4f}')
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('val_accuracy', ascending=False)
    results_df.to_csv(f'grid_search_results/keti_t5_results.csv', index=False)
    print('\n=== 그리드 서치 완료 ===')
    print('\n최적의 하이퍼파라미터:')
    print(f'Learning Rate: {best_params["learning_rate"]}')
    print(f'Batch Size: {best_params["batch_size"]}')
    print(f'Max Epochs: {best_params["max_epochs"]}')
    print(f'Best Validation Accuracy: {best_accuracy:.4f}')
    return results_df, best_params

# 그리드 서치 실행 예시 (주석 처리)
# results_df, best_params = run_grid_search(
#     train_dataset=train_dataset,
#     val_dataset=val_dataset,
#     model_name="KETI-AIR/ke-t5-small-ko",
#     use_auto_classes=False
# )
# print(results_df.head())

# 6. 최적 모델로 예측 테스트
predictor.load_checkpoint('checkpoints/keti_t5_best_model.pt')
test_text = df['input_text'].iloc[0]
true_order = df['target_text'].iloc[0]
predicted_order = predictor.predict_order(test_text)
print("입력 텍스트:")
print(test_text)
print("\n예측된 순서:", predicted_order)
print("실제 순서:", true_order)
print("정확도:", compute_accuracy(predicted_order, true_order)) 
