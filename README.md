# 🔍 EMNIST Autoencoder Anomaly Detection

> 오토인코더 기반 이상탐지 시스템의 하이퍼파라미터 최적화 연구  
> Hyperparameter Optimization for Autoencoder-based Anomaly Detection

---

## 📊 Final Results

| Dataset | Accuracy | Precision | Recall | F1-Score |
|:---:|:---:|:---:|:---:|:---:|
| Validation | 87.25% | 93.13% | 80.44% | 86.32% |
| **Test** | **81.94%** | **82.87%** | **80.53%** | **81.68%** |

---

## 🎯 핵심 발견 — RAdam × Swish 시너지

MSELoss는 이상탐지에 강력하지만, 오차를 제곱하는 특성으로 인해  
**Sharp Loss Landscape**를 형성 → 학습 초기 불안정성 유발

이를 해결하기 위해 두 가지 전략을 결합:

| 전략 | 방법 | 효과 |
|:---:|:---:|:---:|
| 학습 안정화 | **RAdam** (분산 보정 메커니즘) | 초기 발산 방지, 안정적 수렴 |
| 곡면 평탄화 | **Swish** 활성화 함수 | Sharp → Flat Minima 수렴 |

**Optimizer 비교 실험 결과:**

| Optimizer | Test Accuracy | 수렴 안정성 |
|:---:|:---:|:---:|
| Adam | 81.42% | Unstable |
| NAdam | 81.67% | Moderate |
| **RAdam** | **81.94%** | **Stable** ✅ |

---

## 🏗️ 모델 구조

```
Input (784)
    ↓
[Encoder]  784 → 600 → 512 → 256 → 128 → 64
    ↓
Latent Vector (32) ← 핵심 특징 압축
    ↓
[Decoder]  64 → 128 → 256 → 512 → 600
    ↓
Output (784)

이상탐지 원리:
정상 데이터만 학습 → 비정상 입력 시 재구성 오차 급증
→ Threshold 기반 이상 판정
```

---

## ⚙️ 최적 하이퍼파라미터

```
Architecture  : 6-Layer Bottleneck Autoencoder
Optimizer     : RAdam
Activation    : Swish (SiLU)
Loss Function : MSELoss
Learning Rate : 0.0018
Batch Size    : 64
Epochs        : 50
Latent Dim    : 32
```

---

## 🧪 실험 과정

단일 변수 제어(CVS) 방식으로 체계적 하이퍼파라미터 탐색 수행

```
Step 1. Layer 구조 최적화  (5~7 Layer 비교)
    ↓
Step 2. Learning Rate × Epoch 최적화
    ↓
Step 3. Batch Size 최적화  (Sharp/Flat Minima 실험적 검증)
    ↓
Step 4. Optimizer 비교  (Adam / RAdam / NAdam)
    ↓
Step 5. Loss Function 비교  (MSE / L1 / Huber)
```

**Layer 수에 따른 성능 비교:**

| Layer 수 | 구조 | Test Accuracy | 상태 |
|:---:|:---:|:---:|:---:|
| 5 | 784-512-256-128-64-32 | 80.04% | Underfitting |
| **6** | **784-600-512-256-128-64-32** | **81.94%** | **Optimal ✅** |
| 7 | 784-600-512-400-256-128-64-32 | 79.65% | Overfitting |

---

## 🚀 실행 방법

```bash
# 1. Google Colab에서 노트북 열기
# 2. 패키지 설치
pip install torch_optimizer

# 3. 노트북 순서대로 실행
```

---

## 📚 주요 참고 논문

- **RAdam**: Liu et al., *"On the Variance of the Adaptive Learning Rate and Beyond"*, ICLR 2020
- **Sharp Minima**: Keskar et al., *"On Large-Batch Training for Deep Learning"*, ICLR 2017
- **Swish**: Ramachandran et al., *"Searching for Activation Functions"*, arXiv 2017
- **EMNIST**: Cohen et al., *"EMNIST: Extending MNIST to Handwritten Letters"*, IJCNN 2017

---

## 🛠️ 사용 기술

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)
