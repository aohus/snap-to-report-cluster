# 전략

## 전략 S1 – 균형형(balanced baseline)

“일반적인 현장 사진”에 대해 precision/recall 둘 다 적당히 균형 맞추는 세트.

```
geo_param_list_balanced = [
    # B1: 기본 베이스라인
    GeoParams(
        max_features=1500,
        ratio_thresh=0.75,
        ransac_reproj_thresh=5.0,
        min_good_matches=15,
        geo_threshold=0.25,
    ),
    # B2: 매칭 조금 느슨, RANSAC 살짝 타이트
    GeoParams(
        max_features=1500,
        ratio_thresh=0.78,    # 매칭 수 ↑
        ransac_reproj_thresh=4.0,  # inlier 조건 약간 엄격
        min_good_matches=15,
        geo_threshold=0.25,
    ),
    # B3: 특징점 수 줄여서 속도 중시
    GeoParams(
        max_features=1000,
        ratio_thresh=0.75,
        ransac_reproj_thresh=5.0,
        min_good_matches=12,
        geo_threshold=0.25,
    ),
]
```

사용 전략 1. 처음에는 B1 하나로 돌려서 baseline F1/ARI/실제 눈으로 본 품질 체크. 2. 같거나 비슷한 job에서 B2, B3 를 추가로 돌려 보고
• “과하게 합쳐지나?” → B1, B3로
• “너무 쪼개지나?” → B2 쪽으로

## 전략 S2 – 고정밀(high precision) 세트

“다른 장소가 끼어드는 것”을 웬만하면 절대 허용하지 않겠다가 목표일 때.

특징:
• ratio를 낮추거나(0.7 근처) / RANSAC reproj를 낮추고(3~4)
• geo_threshold를 0.3~0.4 정도로 상당히 높게 잡습니다.

```
geo_param_list_high_precision = [
    # P1: 꽤 엄격한 설정
    GeoParams(
        max_features=2000,
        ratio_thresh=0.7,     # ratio test 엄격
        ransac_reproj_thresh=4.0,
        min_good_matches=20,  # 매칭 많이 필요
        geo_threshold=0.35,   # inlier 비율 높아야 통과
    ),
    # P2: RANSAC 더 타이트
    GeoParams(
        max_features=2000,
        ratio_thresh=0.7,
        ransac_reproj_thresh=3.0,
        min_good_matches=20,
        geo_threshold=0.4,
    ),
    # P3: 특징점 수 줄이고 아주 뚜렷한 매칭 위주
    GeoParams(
        max_features=1200,
        ratio_thresh=0.68,    # 아주 엄격
        ransac_reproj_thresh=3.0,
        min_good_matches=18,
        geo_threshold=0.4,
    ),
]
```

기대 효과
• 같은 장소라도 구도가 꽤 달라지면 “다른 장면”으로 갈라질 가능성이 높음.
• 하지만 “완전 엉뚱한 장소끼리 묶이는” 경우는 거의 안 나와야 함.
• 사람이 나무/펜스 일부만 보고 판단해도 “이건 확실히 같은 시점”이라고 느끼는 정도만 클러스터에 남게 하는 쪽.

언제 쓰면 좋냐
• 나중에 이 결과를 교육용/자료집처럼 보여 줄 때
“어지간해서는 오분류가 있으면 안 된다” 쪽.

## 전략 S3 – 고재현(high recall) 세트

반대로:

“같은 장소인데, 시점이 좀 달라지거나 나뭇잎/차량이 가려져 있어도 최대한 묶어줘”

가 목표라면 기하 문턱을 완화합니다.

특징:
• ratio_thresh를 높이거나(0.78~0.8)
• geo_threshold를 낮추고(0.15~0.2)
• RANSAC reproj를 조금 키웁니다(5~6).

```
geo_param_list_high_recall = [
    # R1: 완화된 기준
    GeoParams(
        max_features=1500,
        ratio_thresh=0.78,    # 매칭 많이 허용
        ransac_reproj_thresh=5.0,
        min_good_matches=10,
        geo_threshold=0.2,    # inlier 비율 낮아도 통과
    ),
    # R2: reprojection 오차 더 느슨
    GeoParams(
        max_features=1500,
        ratio_thresh=0.8,
        ransac_reproj_thresh=6.0,
        min_good_matches=10,
        geo_threshold=0.18,
    ),
    # R3: 특징점 수 줄이고 더 느슨, 속도 조금 고려
    GeoParams(
        max_features=1000,
        ratio_thresh=0.8,
        ransac_reproj_thresh=6.0,
        min_good_matches=8,
        geo_threshold=0.15,
    ),
]
```

기대 효과
• 같은 펜스/건물을 다른 각도에서 찍은 사진들이 웬만하면 하나의 “장소 그룹” 안에 들어옵니다.
• 대신, 비슷한 구조(예: 양쪽에 동일한 철제 펜스가 있는 두 장소)가 많을수록
“살짝 다른 장소인데 붙어버리는” 리스크가 생깁니다.
• 이때는 Stage 2 임베딩 threshold와 함께 튜닝해야 합니다.
• APGeM 기준 similarity_threshold를 0.8 이상으로 유지하면서
geo_threshold만 완화하면, 그래도 다른 장소 섞이는 위험은 어느 정도 줄일 수 있습니다.

## 전략 S4 – 저텍스처/속도 최적화 세트

조경/공사 사진 특성상:
• 하늘, 잔디, 흙, 단색 벽 → 로컬 특징이 별로 없음
• MacBook에서 여러 실험 돌릴 때 SIFT가 병목이 되기 쉬움

그래서:
• max_features를 줄이고 (800~1200),
• min_good_matches를 낮추고 (8~12),
• geo_threshold를 중간 정도(0.22~0.28)로 두어
“특징점 적은 사진도 일정 부분 통과”하게 조정.

```
geo_param_list_low_texture_fast = [
    # F1: 비교적 빠르고, 텍스처 적어도 어느 정도 버팀
    GeoParams(
        max_features=1000,
        ratio_thresh=0.75,
        ransac_reproj_thresh=5.0,
        min_good_matches=10,
        geo_threshold=0.22,
    ),
    # F2: 특징점 더 줄이고, 기준은 약간 타이트
    GeoParams(
        max_features=800,
        ratio_thresh=0.75,
        ransac_reproj_thresh=4.5,
        min_good_matches=8,
        geo_threshold=0.25,
    ),
    # F3: 속도 최우선 (실험 초기 coarse search 용)
    GeoParams(
        max_features=600,
        ratio_thresh=0.78,
        ransac_reproj_thresh=5.0,
        min_good_matches=8,
        geo_threshold=0.2,
    ),
]
```

운용 전략 1. grid search 초기에는 F1~F3 위주로 돌려서 “대략적인 좋은 영역”을 찾고 2. 거기서 성능이 잘 나오는 config 근처로 S1/S2/S3 계열에서 더 촘촘하게 튜닝.

# 실험 전략: 어떻게 돌릴지 제안

6-1. 2단계 튜닝 구조 1. 1단계 (탐색 단계)
• 각 전략에서 대표 세트 1~2개씩만 뽑아서:
• geo_param_list = B1, P1, R1, F1, F3 정도 → 총 5개
• GPS/임베딩 파라미터 몇 개와 곱해서 coarse grid search
• 지표:
• 클러스터 수, 평균 클러스터 크기, noise 비율
• 내부 지표(silhouette, DBI)
• 있다면 ARI/NMI 2. 2단계 (세밀 튜닝)
• 1단계에서 “사람 눈 + 정량지표” 기준으로 상위 1~2개 전략 고른 다음
• 그 근처에서만 파라미터 한두 축씩 조정:
• 예: P1이 좋아 보이면
• ratio_thresh: 0.68 / 0.7 / 0.72
• geo_threshold: 0.33 / 0.35 / 0.38
• min_good_matches: 18 / 20

6-2. 전략별 우선순위 추천
• “지금 PDF처럼 꽤 정교한 전/중/후 세트”가 목표라면:
• APGeM / APGeM+CLIP 임베딩 + S1(B1) / S2(P1) 먼저.
• “일단 job 전체를 비슷한 장소끼리라도 빠르게 대강 묶고 싶다”:
• F1, F3 + APGeM 임베딩으로 속도 우선.
• “GPS / 임베딩 튜닝을 먼저 보고, 기하는 나중에 미세조정”:
• geo는 B1/F1 정도로 고정한 뒤,
similarity_threshold, knn_k, GPS eps_m 먼저 sweep.
