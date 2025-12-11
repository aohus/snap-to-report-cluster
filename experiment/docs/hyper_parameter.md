1. Deep 특징 결합 관련 파라미터

1-1. 결합 가중치

```python
# extract_deep_features 내부 (예시)
# CLIP: 0.45, EfficientNet: 0.30, ViT: 0.20, 전통 특징: 0.05
vec_clip = clip_n * 0.45
vec_eff  = eff_n  * 0.30
vec_vit  = vit_n  * 0.20
vec_trad = trad_n * 0.05
combined = concat([vec_clip, vec_eff, vec_vit, vec_trad])
```

    •	역할
    •	어떤 모델의 임베딩을 더 신뢰할지 결정하는 비율입니다.
    •	combined feature 공간에서 해당 부분의 스케일을 키우는 효과라고 보면 됨.
    •	값을 키우면 (예: CLIP 0.45 → 0.6)
    •	CLIP 임베딩이 전체 거리/유사도에 더 큰 영향.
    •	“장소의 의미/장면 수준”을 더 신뢰하게 됨.
    •	값을 줄이면
    •	해당 모델의 정보가 약해지고, 다른 블록이 상대적으로 주도권을 가짐.

튜닝 방향 예시:
• “비슷한 카페/비슷한 공원 구분이 잘 안 된다” → EfficientNet 쪽 가중치를 조금 올려서 texture 비중 ↑
• “비슷한 구도지만 완전히 다른 장소가 자꾸 섞인다” → CLIP 비중 ↑, 전통/texture 비중 ↓

2. 전역 유사도 기반 그래프 구성 파라미터

2-1. similarity_threshold (DeepClusterer 생성자 인자)

```python
sim = 1.0 - dist  # cosine distance -> similarity
if sim < self.similarity_threshold:
    continue  # 이웃으로 안 봄
```

• 역할
• “임베딩 상에서 이 정도 이상 비슷해야 간선 후보로 인정한다”는 기준.
• 크게 하면 (0.6 → 0.8)
• 더 엄격: 정말 비슷한 것들만 연결.
• 결과:
• precision ↑ (다른 장소가 섞일 확률 ↓),
• recall ↓ (같은 장소인데 조금 다른 각도/조명인 것들이 끊길 수 있음),
• 간선 수 ↓ → 속도는 좋아질 수 있음.
• 작게 하면 (0.6 → 0.4)
• 느슨해짐: 꽤 다른 이미지도 일단 후보로 삼음.
• 결과:
• recall ↑ (같은 장소 놓칠 확률 ↓),
• precision ↓ (비슷한 다른 장소 섞이기 쉬움),
• SIFT/RANSAC 호출 횟수 ↑ → 전체 속도 ↓.

2-2. knn_k (advanced_clustering 인자)

```python
k = min(max(10, int(np.sqrt(n)) + 1), n)
nn = NearestNeighbors(n_neighbors=k, metric="cosine")
```

• 역할
• 각 이미지에 대해 몇 개의 이웃을 그래프에서 연결할지 결정.
• 키우면 (예: 20 → 40)
• 더 많은 이웃 후보 → 간선 수 ↑ → 더 연결성이 풍부한 그래프.
• 효과:
• 클러스터가 잘 이어져서 과분할(너무 잘게 쪼개짐) 완화.
• 하지만 서로 전혀 다른 것들이 연결될 가능성도 증가.
• SIFT/RANSAC 호출 수 ↑ → 속도 감소.
• 줄이면 (예: 20 → 5)
• 간선 수 ↓ → 그래프가 더 희소해짐.
• 효과:
• 과분할 위험 ↑ (같은 장소인데 끊겨서 여러 클러스터로 나뉠 수 있음).
• 속도는 빠름.

실전에서는:
• similarity_threshold를 먼저 적당히 맞추고,
• 그 다음 knn_k로 “붙는 정도”를 조절하는 느낌으로 튜닝하면 안정적입니다.

⸻

3. 기하 검증(LocalGeometryMatcher) 파라미터

```python
class LocalGeometryMatcher:
    def __init__(
        self,
        max_features: int = 2000,
        ratio_thresh: float = 0.75,
        ransac_reproj_thresh: float = 5.0,
        min_good_matches: int = 10,
    ):
```

3-1. max_features
• 역할
• SIFT에서 추출할 최대 keypoint 수.
• 키우면
• 더 많은 local feature → 더 많은 매칭 후보 → 복잡한 장면/각도 차이가 큰 경우에도 매칭이 잘 잡힐 수 있음.
• 대신 SIFT 계산 비용 및 RANSAC 비용 ↑ (속도 감소).
• 줄이면
• 속도는 빠르지만, 특히 texture가 빈약한 장면에서 매칭이 잘 안 잡힐 수 있음.

3-2. ratio_thresh (Lowe’s ratio test)

```python
if m.distance < self.ratio_thresh * n.distance:
    good.append(m)
```

• 역할
• 매칭의 “품질 필터링” 기준.
• 값이 작을수록 (0.75 → 0.6) 더 엄격: 좋은 매칭만 통과.
• → inlier 비율은 좋아지지만, 전체 good match 수는 줄어듦.
• 너무 작으면 min_good_matches를 못 채워서 geo_score가 자주 0이 될 수 있음.
• 값이 클수록 (0.75 → 0.9)
• 느슨해져서 매칭 수는 많아지지만,

잘못된 매칭도 섞여서 RANSAC이 더 힘들어짐.

3-3. ransac_reproj_thresh

```python
H, mask = cv2.findHomography(
    src_pts, dst_pts,
    cv2.RANSAC,
    ransacReprojThreshold=self.ransac_reproj_thresh
)
```

• 역할
• RANSAC에서 “inlier로 인정할 최대 reprojection error”.
• 크게 하면 (5.0 → 10.0)
• inlier로 인정되는 매칭이 많아짐 → inlier 비율 ↑ (하지만 진짜로 구조가 다른 두 이미지를 “같다”고 착각할 위험도 ↑)
• 작게 하면 (5.0 → 2.0)
• 더 엄격: 진짜로 잘 맞는 매칭만 inlier로 인정.
• 각도가 많이 다르거나 노이즈가 많은 장면은 inlier가 너무 적어져서 geo_score가 낮아질 수 있음.

3-4. min_good_matches
• 역할
• ratio test를 통과한 매칭이 이 수보다 적으면 아예 0점 처리.
• 크게 하면 (10 → 20)
• 안정된 매칭만 geo_score 계산 대상으로 삼기 때문에, 엉뚱한 매칭이 점수를 주는 경우 줄어듦.
• 하지만 keypoint가 적은 이미지(단순한 벽, 하늘, 잔디)에서는 매칭이 부족해 항상 0점이 될 수 있음.
• 작게 하면
• 더 많은 쌍에 geo_score가 계산됨 → recall ↑, 그러나 노이즈도 같이 늘어날 수 있음.

3-5. geo_score_thresh (advanced_clustering 내부)

```python
score_geo = self.geo_matcher.geo_score(photo_files[i], photo_files[j])
if score_geo < geo_threshold:
    continue
edges.append((i, j))
```

• 역할
• “기하적으로 이 정도 이상 맞아야 같은 장소 후보로 인정한다” 기준.
• 값을 키우면 (0.2 → 0.5)
• 아주 구조적으로 잘 맞는 쌍만 간선으로 사용.
• 결과:
• 다른 장소 섞일 위험 ↓,
• 대신 각도/가림이 심한 같은 장소는 끊길 위험 ↑.
• 값을 줄이면 (0.2 → 0.1)
• geo 검증의 문턱이 낮아져서, CLIP 임베딩만 좋으면 웬만하면 간선이 생김.
• precision보다 recall 쪽으로 기울고, SIFT/RANSAC의 역할이 약해짐.

⸻

4. 사람 제거(마스킹) 관련 파라미터

```python
if float(score) < 0.8:
    continue
if int(label) == 1:  # person
    person_boxes.append(box)
```

    •	DETR의 score threshold (0.8)

• 역할
• “사람으로 인식할 확률이 이 이상일 때만 마스킹”.
• threshold를 내리면 (0.8 → 0.5)
• 더 많은 영역을 “사람일 수도 있는 것”으로 블러 처리.
• 장점:
• 사람/동적 객체가 더 잘 지워져서 장소 구조에 집중.
• 단점:
• 나무, 기둥, 간판 같은 것도 오검출되면 배경 손실 → 임베딩이 꼬일 수 있음.
• threshold를 올리면
• 정말 확실한 사람만 가려서, 배경 손실은 적지만“작은 사람/부분 가려진 사람”은 남을 수 있음.

remove_people 플래그는 이 전체 과정을 켜고 끄는 스위치:
• 사람 많이 나오는 사진 세트라면 True가 유리.
• 속도/단순성을 우선하고, 사람 비중이 적으면 False로 두고 바로 feature 추출하는 것도 선택지.

⸻

5. 클러스터 품질 관련 파라미터 / 지표
   • avg_similarity
   • 각 클러스터의 feature들 vs centroid 코사인 유사도 평균.
   • 높을수록 클러스터 내부가 “밀집된” 상태.
   • quality_score = avg_similarity \* count
   • “얼마나 밀집 + 얼마나 많이 모였는가” 복합 지표.
   • 정렬할 때 중요한/대표적인 클러스터부터 보고 싶을 때 사용.

튜닝과 직접 연결되는 파라미터는 아니지만,
• similarity_threshold, geo_threshold, knn_k 를 바꿔가면서
• avg_similarity 분포, quality_score 분포를 같이 보면
어느 쪽으로 bias가 생기는지 감이 잡힙니다.

6. 실제 튜닝 시 추천 전략 (요약)
   1. 1단계 – 전역 임베딩/가중치
      • CLIP: 0.4–0.6 사이
      • EfficientNet: 0.2–0.4
      • ViT: 처음엔 0.0~0.2 정도로 놓고 “진짜 도움이 되는지” 실험 후 유지/제거 결정
      • 전통 특징: 0.05~0.1 정도로, 큰 영향은 안 주되 보조로 사용
   2. 2단계 – 전역 유사도 기준
      • similarity_threshold: 0.5, 0.6, 0.7 정도로 sweep
      • knn_k: 대략 sqrt(n) 근처에서 ± 몇 개 조정 (예: 10, 20, 30)
   3. 3단계 – 기하 검증 기준
      • geo_threshold: 0.1, 0.2, 0.3 비교
      • max_features: 1000, 2000 정도에서 “속도 vs 안정성” 비교
   4. 4단계 – 사람 마스킹
      • 사람 많은 데이터면 remove_people=True, score >= 0.8
      • 사람 적으면 remove_people=False 로 한 번 돌려 보고, 실제 impact 비교
