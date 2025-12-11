# Construction Photo Clustering Fine-tuning Strategy

## 1. 문제 정의 및 특성 분석
사용자의 데이터셋은 다음과 같은 독특한 특징을 가집니다.
- **Before-During-After (B-D-A) 시퀀스**: 시간차를 두고 같은 장소를 촬영함.
- **Transient Objects (일시적 객체)**: 'During(공사 중)' 사진에는 작업자, 크레인, 포크레인 등 배경을 가리는 객체가 존재함.
- **Fine-grained Similarity**: "컨테이너의 왼쪽"과 "컨테이너의 오른쪽"은 시각적으로 매우 비슷하지만(같은 컨테이너), 공사 보고서 관점에서는 **다른 클러스터**여야 함.
- **Limited Data per Batch**: GPS로 1차 필터링된 10~30장의 소규모 그룹 내에서 정밀 분류 필요.

## 2. 핵심 접근법: Deep Metric Learning (Triplet Loss)
단순히 이미지가 무엇인지 분류하는 것이 아니라, **이미지 간의 유사도(거리)**를 학습해야 합니다.

### 학습 목표
- **Anchor (기준)**: 공사 전(Before) 사진
- **Positive (정답)**: 같은 장소의 공사 후(After) 혹은 공사 중(During) 사진
- **Negative (오답)**: GPS는 가깝지만 다른 부분을 찍은 사진 (예: 바로 옆 나무, 컨테이너 반대편)

모델은 `Distance(Anchor, Positive) < Distance(Anchor, Negative)` 가 되도록 학습합니다.

## 3. 모델 아키텍처 추천
### Backbone Network
- **ResNet50** 또는 **EfficientNet-B0**: 적절한 성능과 속도 균형.
- **DINOv2 (ViT)**: (고급 옵션) 구조적인 특징(Structure)을 잡는 데 매우 강력하여, 색감이나 날씨 변화에 강인함.

### Embedding Head
Backbone의 마지막 분류(Classification) 레이어를 제거하고, 128~512차원의 벡터를 출력하는 **Embedding Layer**를 부착합니다.

## 4. 데이터셋 구성 및 전처리 (Data Strategy)
### Hard Negative Mining (중요)
- 단순히 전혀 다른 사진을 Negative로 쓰면 학습이 쉽지만 성능이 오르지 않습니다.
- **Hard Negative**: 같은 공사 현장이지만 앵글이 살짝 다르거나, 비슷한 구조물을 포함한 "틀리기 쉬운 오답"을 Negative로 주어야 모델이 "컨테이너 왼쪽 vs 오른쪽"을 구별할 수 있습니다.

### Augmentation (증강 기법)
- **Color Jitter**: 야외 날씨/조명 변화 대응 (밝기, 대비 조절).
- **Random Erasing / Cutout**: '공사 중' 사진의 작업자/장비를 시뮬레이션하기 위해 이미지 일부를 랜덤하게 지우고 학습시킴.
- **Geometric**: 약한 회전 및 Perspective 변환 (손으로 찍어 각도가 틀어진 것 대응).
- **Flip 금지**: 구조물의 좌우가 바뀌면 다른 장소가 될 수 있으므로 좌우 반전은 신중해야 함.

## 5. 학습 프로세스
1. **데이터 준비**: 기존에 사람이 수동으로 분류 완료한 클러스터 데이터를 `train/cluster_id/images...` 형태로 정리.
2. **Triplet 생성**: 같은 폴더 내에서 2장(Anchor, Positive), 다른 폴더에서 1장(Negative)을 뽑아 배치 구성.
3. **Loss 계산**: Triplet Margin Loss 사용.
4. **평가**: 별도의 Test 셋에서 클러스터링 정확도(ARI, Silhouette Score) 측정.
