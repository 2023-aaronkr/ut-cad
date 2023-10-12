---
layout: page
title: Midterm Guide / 중간고사 가이드
permalink: /midterm/
---

{:toc}

---

1. [영상처리 소개와 영상시스템](#1-영상처리-소개와-영상시스템)
2. [CAD용 Python 소개](#2-cad용-python-소개)
3. [DICOM 이미지](#3-dicom-이미지)
4. [2D 분류 모델 (머신러닝)](#4-2d-분류-모델-머신러닝)

---

## 1. 영상처리 소개와 영상시스템

- Visual spectrum (전자기 스펙트럼)
- X-ray (X선)
- Gamma ray (감마선)
- Ultrasound (초음파)
- Ionizing radiation (이온화 방사선)
- Direct imaging (직접 영상)
- Indirect imaging (간접 영상)
- Gray scale (그레이 스케일)
- Contrast (대비)
- Spatial resolution (공간 해상도)
- Temporal resolution (시간 해상도)
- Noise (노이즈)
- Signal-to-noise ratio (신호 대 잡음 비)
- Continuous-to-continuous imaging (연속-연속 영상)
- Continuous-to-discrete imaging (연속-이산 영상)
- Image enhancement (영상 개선)
- Image restoration (영상 복원)
- Image analysis (영상 분석)
- Image compression (영상 압축)
- Image synthesis (영상 합성)
- Image recognition (영상 인식)
- Image reconstruction (영상 재구성)
- Image registration (영상 등록)
- Image segmentation (영상 분할)

[목차로 돌아가기](#midterm-guide--중간고사-가이드)

---

## 2. CAD용 Python 소개

- X-ray (X선)
- CT (Computed Tomography, 컴퓨터 단층 촬영)
- MRI (Magnetic Resonance Imaging, 자기 공명 영상)
- PET (Positron Emission Tomography, 양전자 방출 단층 촬영)
- SPECT (Single Photon Emission Computed Tomography, 단일 광자 방출 단층 촬영)
- US (Ultrasound, 초음파)
- Fundal Imaging (안저 촬영)
- Microscopy (현미경)
- Mammography (유방 촬영)
- Radiologist (방사선 전문의)
- Diagnosing Clinician (진단 전문의)
- Pathologist (병리학자)
- Histogram (히스토그램)

---

- **X-ray (엑스레이)**: 단일 이미지를 캡처하기 위해 단일 방향에서 신체에 x- 선이라고하는 방사선 유형을 투사하는 2D 이미징 기술.
- **Ultrasound (초음파)**: 고주파 음파를 사용하여 이미지를 생성하는 2D 이미징 기술.
- **Computed Tomography (컴퓨터 단층 촬영 (CT))**: 인체 주위의 여러 각도에서 x- 레이를 방출하여보다 다른 각도에서 더 세밀한 부분을 포착하는 3D 이미징 기술.
- **Magnetic Resonance Imaging (자기 공명 영상 (MRI))**: 강한 자기장과 전파를 사용하여 모든 각도에서 신체 부위의 이미지를 만드는 3D 이미징 기술.
- **2D 이미징**: 사진을 단일 각도로 촬영하는 이미징 기술.
- **3D 이미징**: 이미지를 다른 각도에서 촬영하여 많은 양의 이미지를 만드는 이미징 기술.

---

- **Mammogram (유방 조영술)**: 유방 영상 촬영에 특화된 2D 엑스레이의 일종.
- **Digital Pathology (디지털 병리학)**: 세포 수준의 생물학적 물질의 현미경 이미지를 디지털화하는 2D 이미징 유형.
- **Radiologist (방사선사)**: 의료 영상 데이터를 판독하도록 훈련받은 전문 임상의입니다.
- **PACS (Picture Archiving and Communication System)**: 병원 내외에서 의료 이미지를 저장하고 보는 데 사용되는 사진 보관 및 통신 시스템.
- **Screening (선별검사)**: 특정 질병에 대한 위험군에 속하는 개인을 대상으로 수행되는 테스트 유형.
- **Sensitivity (민감도)**: 테스트에서 반환된 정확하게 식별된 양성 사례의 비율.
- **Specificity (특이성)**: 테스트에서 반환되는 정확하게 식별된 부정적인 사례의 비율.

[목차로 돌아가기](#midterm-guide--중간고사-가이드)

---

## 3. DICOM 이미지

- **DICOM (Digital Imaging and Communications in Medicine, 의학 영상 및 통신용 디지털 이미징)**: 의료 영상을 저장하고 전송하는 데 사용되는 표준.
- **PACS (Picture Archiving and Communication System, 사진 보관 및 통신 시스템)**: 병원 내외에서 의료 이미지를 저장하고 보는 데 사용되는 사진 보관 및 통신 시스템.
- **DICOM header attributes (DICOM 헤더 속성)**: DICOM 이미지에 대한 정보를 포함하는 헤더의 일부.
- **EMR (Electronic Medical Record, 전자 의료 기록)**: 환자의 의료 정보를 저장하는 전자 시스템.

---

**Normalizing Pixel Intensities (픽셀 강도 정규화)**

```python
# 픽셀 강도 정규화
pixel_mean = np.mean(old_img) # mean: 평균
pixel_std = np.std(old_img) # std: 표준편차
normalize = (old_img - pixel_mean) / pixel_std
```

---

- **Histograms (히스토그램)**: 픽셀 강도의 분포를 보여주는 그래프. (Useful for assessing distributions of a single variable.) (단일 변수의 분포를 평가하는 데 유용합니다.)
- **Scatterplots (산점도)**: 두 변수 간의 관계를 보여주는 그래프. (Useful for assessing relationships between two variables.) (두 변수 간의 관계를 평가하는 데 유용합니다.)
- **Co-occurrence matrix (공존 행렬)**: 픽셀 강도의 공존을 보여주는 행렬. (Useful for assessing how frequently different variables occur together.) (서로 다른 변수가 얼마나 자주 함께 발생하는지 평가하는 데 유용합니다.)

[목차로 돌아가기](#midterm-guide--중간고사-가이드)

---

## 4. 2D 분류 모델 (머신러닝)

- **Machine learning (머신러닝)**: 컴퓨터가 데이터를 사용하여 학습하고 예측을 수행하는 방법을 배우는 분야.
  - 연구자들은 사전 정의된 특성 강도 특성, 모양 특성, 위치 특성 등을 수행하는데 상당한 시간이 걸립니다.
  - Example: Otsu's method (예: 오츠의 방법)
- **Deep learning (딥러닝)**: 머신러닝의 한 유형으로, 인공 신경망을 사용하여 데이터를 학습하고 예측을 수행하는 방법을 배우는 분야.
  - DL은 귀하를 위한 기능을 발견하여 모델 개발을 더 빠르고 정확하게 만듭니다.
  - Example: Convolutional neural networks (CNNs) (예: 합성곱 신경망 (CNN))

---

- **Otsu's method (오츠의 방법):** 이미지의 픽셀 강도 분포를 사용하여 이미지를 이진화하는 방법.
- **Thresholding (임계값 처리)**: 이미지를 이진화하는 방법.
- **Binary image (이진 이미지)**: 픽셀 강도가 0 또는 1로 표시되는 이미지.

---

- **Convolutional neural networks (CNNs, 합성곱 신경망)**: 이미지를 분류하는 데 사용되는 딥러닝 모델.
- **Convolutional layer (합성곱층)**: 이미지의 특징을 추출하는 데 사용되는 레이어.
  - Each layer consists of a set of filters which are matrices of weighted values (각 레이어는 일련의 필터로 구성되며, 이는 가중치 값의 행렬입니다.)
  - As the filters move across the image, they perform a convolution operation to detect features in the image (필터가 이미지를 통과하면서 합성곱 연산을 수행하여 이미지의 특징을 감지합니다.)
  - The output of a convolutional layer is a set of feature maps (합성곱층의 출력은 특징 맵의 집합입니다.)
  - As layers get deeper, they can detect more complex features (레이어가 깊어질수록 더 복잡한 특징을 감지할 수 있습니다.)
- **Pooling layer (풀링층)**: 이미지의 크기를 줄이는 데 사용되는 레이어.
- **Fully connected layer (완전 연결층)**: 이미지를 분류하는 데 사용되는 레이어.
- **Softmax function (소프트맥스 함수)**: 각 클래스에 대한 확률을 반환하는 함수.
- **Dataset (데이터셋)**: 모델을 학습시키기 위해 사용되는 이미지의 집합.
  - Training set (학습 세트): 모델을 학습시키기 위해 사용되는 이미지의 하위 집합.
  - Validation set (검증 세트): 모델을 학습시키기 위해 사용되는 이미지의 하위 집합.
  - Test set (테스트 세트): 모델을 평가하기 위해 사용되는 이미지의 하위 집합.
  - Dataset split - usually 80% training, 10% validation, 10% test (데이터셋 분할 - 일반적으로 80% 학습, 10% 검증, 10% 테스트)
- **Epoch (에포크)**: 모델이 데이터셋의 모든 이미지를 한 번씩 본 것을 의미합니다.
- **Batch size (배치 크기)**: 모델이 한 번에 처리하는 이미지의 수.
- **Learning rate (학습률)**: 모델이 학습하는 속도.
- **Loss function (손실 함수)**: 모델이 얼마나 잘 작동하는지 측정하는 함수.
- **Optimizer (최적화기)**: 모델이 손실 함수를 최소화하는 방법을 결정하는 방법.
- **Hyperparameters (하이퍼파라미터)**: 모델의 학습률, 배치 크기, 에포크 수 등과 같은 설정.
- **Training (학습)**: 모델이 데이터셋의 이미지를 사용하여 손실 함수를 최소화하는 방법을 배우는 것.
- **Validation (검증)**: 모델이 학습 중에 데이터셋의 이미지를 사용하여 손실 함수를 최소화하는 방법을 평가하는 것.
- **Test (테스트)**: 모델이 학습 후에 데이터셋의 이미지를 사용하여 손실 함수를 최소화하는 방법을 평가하는 것.
- **Overfitting (과적합)**: 모델이 학습 데이터에 대해 너무 잘 작동하지만 새로운 데이터에 대해 잘 작동하지 않는 것.
  - To prevent overfitting, we can add dropout layers to our model. (과적합을 방지하기 위해 모델에 드롭아웃 레이어를 추가할 수 있습니다.)
- **Underfitting (과소적합)**: 모델이 학습 데이터에 대해 잘 작동하지 않는 것.
- **Early stopping (조기 종료)**: 모델이 과적합되는 것을 방지하기 위해 학습을 중지하는 것.
- **Data augmentation (데이터 증강)**: 모델이 학습하는 동안 데이터셋의 이미지를 변형하는 것.
- **Transfer learning (전이 학습)**: 모델이 이미 학습된 모델의 일부를 사용하여 학습하는 것.
- **Fine-tuning (미세 조정)**: 모델이 이미 학습된 모델의 일부를 사용하여 학습하는 것.
  - In transfer learning / fine-tuning, we can freeze the early layers and add only new layers to train. In Keras, we set the `trainable` parameter to `False` to freeze a layer. (전이 학습 / 미세 조정에서 초기 레이어를 고정하고 새 레이어만 추가하여 학습할 수 있습니다. Keras에서 `trainable` 매개변수를 `False`로 설정하여 레이어를 고정합니다.)
- **Preprocessing (전처리)**: 모델이 학습하기 전에 데이터셋의 이미지를 변형하는 것.
  - Normalization (정규화): 이미지의 픽셀 강도를 0과 1 사이로 변환하는 것.
  - Standardization (표준화): 이미지의 픽셀 강도를 평균 0, 표준 편차 1로 변환하는 것.
  - Augmentation (증강): 이미지를 변형하는 것. (Never augment the validation or test set.) (검증 세트 또는 테스트 세트를 증강하지 마십시오.)
  - Resizing (크기 조정): 이미지의 크기를 변경하는 것.

---

- **Gold Standard (최적 표준):** 가장 높은 감도와 정확도로 질병을 감지하는 방법입니다.
- **Ground Truth:** 알고리즘의 출력과 비교하고 성능을 설정하는 데 사용되는 레이블입니다.
- **Silver Standard (실버 표준):** 여러 가지 라벨 소스를 고려하여 실제 정보를 생성하는 방법

[목차로 돌아가기](#midterm-guide--중간고사-가이드)
