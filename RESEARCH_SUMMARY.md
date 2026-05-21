# 📄 Research Summary

## EMNIST Balanced 데이터셋 기반 오토인코더의 하이퍼파라미터 최적화 및 이상탐지 성능 평가

> 숭실대학교 전자정보공학부 IT융합전공 | 고급인공신경망 수업 최종 프로젝트

---

## 🧠 연구 배경 & 문제 정의

### 왜 오토인코더인가?

오토인코더는 **정상 데이터만으로 학습**하는 비지도 학습 모델이다.

```
정상 데이터 학습 → 정상 입력: 낮은 재구성 오차
                 → 이상 입력: 높은 재구성 오차 → 이상 판정
```

제조 품질 검사, 금융 사기 탐지, 네트워크 침입 탐지 등  
**레이블 없는 실제 환경**에서 강력한 이상탐지 도구로 활용 가능

---

### 핵심 문제: MSELoss의 딜레마

```
MSELoss의 장점          MSELoss의 단점
─────────────────       ─────────────────────────
이상치에 민감            Sharp Loss Landscape 형성
재구성 오차 극대화        → 학습 초기 불안정성
높은 이상탐지 구분력      → Sharp Minima 수렴 위험
                         → 하이퍼파라미터 민감도 증가
```

**연구 목표**: MSELoss의 구분력은 유지하면서 학습 불안정성을 해결하는 최적화 전략 탐색

---

## 🏗️ 모델 구조

### 6-Layer Bottleneck Autoencoder

```
784 → 600 → 512 → 256 → 128 → 64 → [32] → 64 → 128 → 256 → 512 → 600 → 784
      ←────────── Encoder ──────────→  Latent  ←──────────── Decoder ──────────→
```

- **입력**: EMNIST 손글씨 이미지 (28×28 = 784 픽셀)
- **Latent Vector**: 32차원 (핵심 특징 압축)
- **이상탐지**: Validation 데이터 재구성 오차 분포의 95~99 백분위수를 임계값으로 설정

---

## 🧪 실험 & 결과

### Step 1. Layer 수 최적화

| Layer 수 | 구조 | Test Accuracy | 판정 |
|:---:|:---|:---:|:---:|
| 5 | 784-512-256-128-64-32 | 80.04% | Underfitting |
| **6** | **784-600-512-256-128-64-32** | **81.94%** | **최적 ✅** |
| 7 | 784-600-512-400-256-128-64-32 | 79.65% | Overfitting |

> **결론**: 층이 많다고 무조건 좋은 게 아님. 데이터 복잡도와 모델 표현력 사이의 균형점이 존재

---

### Step 2. Learning Rate × Epoch 최적화

| Learning Rate | Epoch 20 | Epoch 50 | Epoch 70 |
|:---:|:---:|:---:|:---:|
| 0.01 | 52.76% | 52.76% | 52.76% |
| 0.0001 | 74.15% | 78.16% | 79.81% |
| **0.0018** | 79.95% | **81.94%** | 79.74% |

> **LR 0.01**: Hessian 최대 고유값 > LR → Sharp Minima 발산  
> **LR 0.0001**: 안정적이나 수렴 속도 과도하게 느림  
> **LR 0.0018 × Epoch 50**: 최적 균형점 달성

---

### Step 3. Batch Size 최적화

| Batch Size | Test Accuracy | 특성 |
|:---:|:---:|:---|
| 32 | 81.42% | 불안정, 느린 학습 |
| **64** | **82.87%** | **최적 균형 ✅** |
| 128 | 80.17% | Sharp Minima 수렴 경향 |
| 256 | 80.60% | 일반화 성능 저하 |

> Keskar et al. (ICLR 2017) 이론 실증:  
> 큰 Batch Size → Sharp Minima → 일반화 성능 하락

---

### Step 4. Optimizer 비교

| Optimizer | Test Accuracy | Train Loss | 수렴 안정성 |
|:---:|:---:|:---:|:---:|
| Adam | 81.42% | 0.0145 | ⚠️ Unstable |
| NAdam | 81.67% | 0.0141 | 〰️ Moderate |
| **RAdam** | **81.94%** | **0.0139** | **✅ Stable** |

**RAdam이 우수한 이유:**
```
Adam의 문제점
초기 학습 단계에서 적응형 학습률의 분산이 과도하게 큼
→ Sharp Landscape에서 불안정 수렴

RAdam의 해결책
분산 보정 항(Variance Rectification Term) 도입
→ 초기 분산 자동 제어
→ 분산 불안정 시 모멘텀 SGD로 자동 전환
→ Warm-up 없이도 안정적 초기 학습 가능
```

---

### Step 5. Loss Function 비교

| Loss Function | Test Accuracy | Train Loss | 특성 |
|:---:|:---:|:---:|:---|
| **MSELoss** | **81.94%** | 0.0139 | 이상치 민감, 높은 구분력 |
| HuberLoss | 80.93% | **0.0071** | 이상치 강건, 안정적 학습 |
| L1Loss | 79.96% | 0.0460 | 이상치 둔감, 낮은 탐지력 |

> MSELoss의 제곱 패널티가 정상/이상 데이터 재구성 오차 차이를 극대화  
> → 이상탐지 성능에 직접적으로 기여

---

## 🔑 핵심 발견: RAdam × Swish 시너지 메커니즘

```
MSELoss
    │
    ▼
Sharp Loss Landscape 형성
    │
    ├──→ [Swish 활성화 함수]
    │         비단조성(Non-Monotonicity)으로
    │         손실 곡면 평탄화 (Smoothing)
    │         → Flat Minima 수렴 유도
    │
    └──→ [RAdam Optimizer]
              분산 보정으로
              초기 학습 경로 안정화
              → 최적해 탐색 효율 극대화
    │
    ▼
MSELoss의 높은 이상탐지 구분력 유지
+ 안정적 학습 (딜레마 해결)
```

---

## 📊 최종 성능

| Dataset | Accuracy | Precision | Recall | F1-Score |
|:---:|:---:|:---:|:---:|:---:|
| Validation | 87.25% | 93.13% | 80.44% | 86.32% |
| **Test** | **81.94%** | **82.87%** | **80.53%** | **81.68%** |

- Validation-Test 성능 차이 **약 5%** → 과적합 없는 안정적 일반화 확인
- F1-Score **81.68%** → Precision/Recall 균형 달성
- 실제 이상탐지 시스템 적용 가능한 수준

---

## ⚙️ 최적 하이퍼파라미터 구성

```
┌─────────────────────────────────────────┐
│  Architecture  : 6-Layer Autoencoder    │
│  784-600-512-256-128-64-32 (Symmetric)  │
│                                         │
│  Optimizer     : RAdam                  │
│  Activation    : Swish (SiLU)           │
│  Loss Function : MSELoss                │
│                                         │
│  Learning Rate : 0.0018                 │
│  Batch Size    : 64                     │
│  Epochs        : 50                     │
│  Latent Dim    : 32                     │
└─────────────────────────────────────────┘
```

---

## 🔭 향후 연구 방향

1. **Cyclical Learning Rate** 적용 → 동적 학습률 스케줄링으로 성능 개선
2. **Information Bottleneck 이론** 적용 → Latent Vector 정보량 최적화
3. **Edge Device 경량화** → NPU/FPGA 배포를 위한 SW-HW Co-design
4. **전이 학습** → 다양한 도메인(제조, 금융, 네트워크) 적용 가능성 검증

---

## 📚 참고 논문

| 번호 | 논문 | 저널/학회 | 연도 |
|:---:|:---|:---:|:---:|
| [1] | Pang et al., "Deep learning for anomaly detection: A review" | ACM Computing Surveys | 2021 |
| [2] | Goodfellow et al., "Deep Learning" | MIT Press | 2016 |
| [3] | An & Cho, "Variational autoencoder based anomaly detection" | Special Lecture on IE | 2015 |
| [4] | Cohen et al., "EMNIST: Extending MNIST to handwritten letters" | IJCNN | 2017 |
| [10] | Keskar et al., "On large-batch training for deep learning" | **ICLR** | 2017 |
| [11] | Liu et al., "On the variance of the adaptive learning rate and beyond" | **ICLR** | 2020 |
| [12] | Ramachandran et al., "Searching for activation functions" | arXiv | 2017 |
