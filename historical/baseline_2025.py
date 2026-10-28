#필요한 라이브러리 임포트
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. UCI 저장소의 오픈데이터('Bank Marketing') 로드 
bank_marketing = fetch_ucirepo(id=222)  #Bank Marketing 데이터셋
X = bank_marketing.data.features    #특징 데이터(나이, 직업, 대출 여부 등)
y = bank_marketing.data.targets     #타겟 데이터(마케팅 성공 여부('yes'/'no'))

# 2. 결측치 처리
data = pd.concat([X, y], axis=1).dropna()   #NaN이 포함된 행 제거

# 3. 범주형 변수 원-핫 인코딩 
cat_cols = data.select_dtypes(include=['object', 'category']).columns.drop('y')     #'y' 제외한 문자열형 변수 선택
data = pd.get_dummies(data, columns=cat_cols)   #원-핫 인코딩 수행

# 4. 타겟 인코딩 ('y' 컬럼: yes/no → 1/0)
data['y'] = data['y'].map({'yes': 1, 'no': 0})

# 5. 특성과 타겟 분리
X = data.drop('y', axis=1).values   #X: 입력 피처
y = data['y'].values                #y: 타겟 값 (1/0)

# 6. 표준화 (각 피처를 평균 0, 분산 1로 스케일링)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 9. 신경망 모델 정의 (다층 퍼셉트론 구조, 드롭아웃)
class BankMarketingMLP(nn.Module):
    def __init__(self, input_dim, dropout1=0.5, dropout2=0.4):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, 128)    #첫 번째 완전연결층
        self.bn1 = nn.BatchNorm1d(128)          #배치 정규화
        self.dropout1 = nn.Dropout(dropout1)    #드롭아웃
        
        self.fc2 = nn.Linear(128, 64)           #두 번째 완전연결층
        self.bn2 = nn.BatchNorm1d(64)           #배치 정규화
        self.dropout2 = nn.Dropout(dropout2)    #드롭아웃
        
        self.fc3 = nn.Linear(64, 1)             #마지막 출력층 (이진 분류)
        
        #He 초기화
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.LeakyReLU(negative_slope=0.01)(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.LeakyReLU(negative_slope=0.01)(x)
        x = self.dropout2(x)

        return self.fc3(x)  #출력층


# 데이터 분할 (훈련 64%, 검증 16%, 테스트 20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# 넘파이 배열을 파이토치 텐서로 변환
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 하이퍼파라미터 후보 설정 및 조기 종료 변수 초기화
learning_rates = [0.1, 0.001, 0.0005]   #학습률
batch_sizes = [32, 64, 128]             #배치 사이즈
dropout1_rates = [0.3, 0.4, 0.5]        #첫 번째 드롭아웃 비율
dropout2_rates = [0.3, 0.4, 0.5]        #두 번째 드롭아웃 비율
patience = 10                           #조기 종료 기준(10 epoch 동안 개선 없으면 조기 종료)
best_params = None                      #최적 하이퍼파라미터 저장 변수
best_acc = 0                            #최고 정확도 저장

#모든 하이퍼파라미터 조합에 대해 학습 및 평가 수행
for lr in learning_rates:               #학습률 후보군 반복
    for bs in batch_sizes:              #배치 크기 후보군 반복
        for dr1 in dropout1_rates:      #첫 번째 드롭아웃 비율 반복
            for dr2 in dropout2_rates:  #두 번째 드롭아웃 비율 반복
                print(f"\n=== lr={lr}, bs={bs}, dr1={dr1}, dr2={dr2} ===")
            
                # 모델 초기화 (드롭아웃 반영)
                model = BankMarketingMLP(input_dim=X_train.shape[1], dropout1=dr1, dropout2=dr2)
                
                # 양성 클래스 비율을 바탕으로 클래스 불균형 보정용 가중치 계산
                pos = y_train.sum()             #양성 샘플 개수
                neg = len(y_train) - pos        #음성 샘플 개수
                pos_weight = torch.tensor(neg / pos)    #양성 샘플의 상대적 가중치
                
                #클래스 가중치를 포함한 이진 분류 손실 함수 설정
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                #Adam 옵티마이저 설정
                optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
                #학습률을 주기적으로 감소시키는 CosineAnnealingLR 스케줄러
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
            
                #과적합 방지를 위해, 드롭아웃 비율이 작을 경우 학습률 조정
                if dr2 < 0.4:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr * 0.75  # 25% 감소

                # 조기 종료를 위한 초기값 설정
                best_val_loss = float('inf')    #현재까지의 최저 검증 손실
                no_improve = 0                  #개선되지 않은 epoch 수
            
                # 학습 루프
                for epoch in range(200):        #최대 200 epoch까지 반복
                    model.train()               #학습 모드 설정
                    # 배치 학습
                    for i in range(0, len(X_train), bs):
                        X_batch = X_train_tensor[i:i+bs]
                        y_batch = y_train_tensor[i:i+bs]
                        
                        optimizer.zero_grad()               #기울기 초기화
                        outputs = model(X_batch)            #순전파(logit 반환)
                        loss = criterion(outputs, y_batch)  #손실 계산

                        #손실이 NaN이면 훈련 중단(예외적 상황 방지용)
                        if torch.isnan(loss):
                            print(f"훈련 손실 NaN 발생! lr={lr}, dr1={dr1}, dr2={dr2}")
                            break  # 내부 배치 루프 종료

                        loss.backward()     #역전파로 기울기 계산
                        
                        #기울기 클리핑 (폭발 방지)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, error_if_nonfinite=True)
                        
                        optimizer.step()    #가중치 업데이트
                
                    # epoch 단위로 학습률 업데이트 
                    scheduler.step()

                    # 검증 손실 계산
                    model.eval()        #평가 모드
                    with torch.no_grad():
                        val_logits = model(X_val_tensor)    #logit 반환
                        val_loss = criterion(val_logits, y_val_tensor).item()   #검증 손실 계산
                        
                    if np.isnan(val_loss):  # numpy로 변환 후 NaN인지 검사
                        print(f"검증 손실 NaN 발생! lr={lr}, dr1={dr1}, dr2={dr2}")
                        break  # epoch 루프 종료
                        
                    # 10 epoch마다 로그 출력
                    if epoch % 10 == 0:
                        print(f"Epoch {epoch}, Val Loss: {val_loss:.4f}")
                    
                    # 조기 종료 체크
                    if val_loss < best_val_loss:    #검증 손실이 개선된 경우
                        best_val_loss = val_loss
                        no_improve = 0      #개선 횟수 초기화
                    else:       #검증 손실이 개선되지 않을 경우
                        no_improve +=1      #개선되지 않음 카운트 증가
                        if no_improve >= patience:      #일정 횟수 이상 개선되지 않으면 종료
                            print(f"Early stopping at epoch {epoch}")
                            break
               
            
                # 테스트셋 평가
                model.eval()
                with torch.no_grad():
                    logits = model(X_test_tensor)       #로짓 계산
                    pred_prob = torch.sigmoid(logits)   #확률 변환
                    pred_label = (pred_prob > 0.5).float()  #0.5 기준 이진 분류
                    accuracy = (pred_label == y_test_tensor).float().mean().item()  #정확도 계산
                    print(f"Test Accuracy: {accuracy:.4f}")
                
                    # 최적 하이퍼파라미터 갱신
                    if accuracy > best_acc:
                        best_acc = accuracy
                        best_params = (lr, bs, dr1, dr2)

#최종 결과 출력
print(f"\n최적 하이퍼파라미터: lr={best_params[0]}, bs={best_params[1]}, dr1={best_params[2]}, dr2={best_params[3]}")
print(f"최고 테스트 정확도: {best_acc:.4f}")