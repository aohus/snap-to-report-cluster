# SIFT 기반 기하 검증 파라미터 정리

이 문서는 다음 네 가지 파라미터가 **어떤 단계에서**, **무엇을** 의미하는지 정리한 내용입니다.

- `ratio_thresh`
- `ransac_reproj_thresh`
- `geo_threshold`
- `min_good_matches`

예시는 SIFT + BFMatcher + RANSAC(homography) 파이프라인을 기준으로 합니다.

---

## 0. 전체 흐름

두 이미지를 비교할 때 전체 흐름은 대략 이렇게 동작합니다.

1. 각 이미지에서 **SIFT 특징점/디스크립터** 추출
2. 두 이미지 사이 **descriptor 매칭** (KNN 매칭)
3. **Lowe ratio test** 적용 → `ratio_thresh` 사용  
   → 애매한 매칭 제거, `good_matches`만 남김
4. `good_matches` 개수가 `min_good_matches` 미만이면  
   → 데이터가 부족하다고 보고 **실패 처리**
5. `good_matches`로 RANSAC 기반 **homography 추정**  
   → inlier / outlier 분리할 때 `ransac_reproj_thresh` 사용
6. inlier 비율(= inliers / len(good_matches))을 계산하고  
   이 값을 `geo_score`로 사용
7. `geo_score >= geo_threshold` 이면  
   → **기하적으로 잘 맞는 한 장면**이라고 판단

각 파라미터는 이 흐름에서 다른 역할을 합니다.

---

## 1. `ratio_thresh` – 매칭 품질 필터

### 무엇을 하는가?

KNN 매칭에서, 각 descriptor에 대해:

- 1등 매칭 거리: `d1`
- 2등 매칭 거리: `d2`

를 얻고, 아래와 같이 **Lowe ratio test**를 적용합니다.

```python
if d1 < ratio_thresh * d2:
    good_matches.append(m)  # m = 첫 번째 매칭
```

- `ratio_thresh` **작을수록** (예: 0.7)
  - “1등이 2등보다 충분히 더 가까울 때만” 좋은 매칭으로 인정
  - 즉, **엄격한 필터** → 매칭 수는 줄지만 신뢰도 높은 매칭만 남음
- `ratio_thresh` **클수록** (예: 0.8, 0.85)
  - 1등과 2등이 비슷해도 허용 → 매칭 수는 늘지만, **헷갈리는 매칭(오탐)**도 섞일 수 있음

---

## 2. `min_good_matches` – 기하 검증을 시도할 만큼 매칭이 충분한가?

### 무엇을 하는가?

ratio test를 통과한 매칭 리스트를 `good_matches`라고 할 때:

```python
if len(good_matches) < min_good_matches:
    # 기하 추정을 시도하기엔 데이터가 부족
    return 0.0  # 혹은 "매칭 실패" 처리
```

- Homography를 추정하려면 **이론적으로 4점 이상** 필요하지만,
- 실제로는 outlier도 섞이므로 **더 많은 매칭**이 필요합니다.
- `min_good_matches`는  
  → “RANSAC으로 기하 추정을 시도할 최소 데이터량”에 대한 기준입니다.

---

## 3. `ransac_reproj_thresh` – 기하학적으로 얼마나 정확해야 inlier라고 볼까?

### 무엇을 하는가?

OpenCV의 `cv2.findHomography` RANSAC 호출 시:

```python
H, mask = cv2.findHomography(
    pts1,
    pts2,
    cv2.RANSAC,
    ransacReprojThreshold=ransac_reproj_thresh,
)
```

- `ransac_reproj_thresh`는 **reprojection error 허용 범위(픽셀)** 입니다.
- 한 점을 homography로 옮겼을 때,  
  실제 대응점과의 거리가 이 값보다 작으면 **inlier**,  
  크면 **outlier**로 간주됩니다.

- 값이 **작을수록** (예: 2.0~3.0)
  - “거의 정확히 맞아야 inlier” → 아주 엄격한 기하 일치만 인정
- 값이 **클수록** (예: 5.0~6.0)
  - 각도/렌즈 왜곡 차이를 어느 정도 허용  
  - 대신 잘못된 매칭도 일부 inlier로 들어올 수 있음

---

## 4. `geo_threshold` – 기하적으로 얼마나 “많이” 맞아야 같은 장소라고 볼까?

### 무엇을 하는가?

RANSAC 결과로 얻은 mask를 사용해:

```python
inliers = mask.sum()
total = len(good_matches)
geo_score = inliers / total  # 0.0 ~ 1.0
```

- `geo_score` = “ratio test 통과한 좋은 매칭 중 **얼마나 많은 비율이 inlier인가**”
- 이 값과 `geo_threshold`를 비교하여:

```python
if geo_score >= geo_threshold:
    # 같은 장소일 가능성이 높다고 판단
else:
    # 기하적으로 일치도가 부족 → 다른 장면으로 취급
```

- `geo_threshold` **작을수록** (예: 0.2)
  - inlier 비율이 조금만 돼도 같은 장소로 인정 → 재현율↑, 정밀도↓
- `geo_threshold` **클수록** (예: 0.35~0.4)
  - inlier가 충분히 많아야 같은 장소로 인정 → 정밀도↑, 재현율↓

---

## 5. 예시: 네 파라미터가 함께 작동하는 그림

같은 작업 장소, 비슷한 각도에서 찍은 “나무 절지 전/후” 사진 두 장 비교:

1. SIFT detect & compute  
   - A: 800개 feature, B: 900개 feature
2. KNN 매칭 → 300쌍
3. ratio test (`ratio_thresh = 0.75`)  
   → `good_matches = 40`
4. `min_good_matches = 10` 이라면  
   → 40 ≥ 10 → RANSAC 진행
5. RANSAC (`ransac_reproj_thresh = 4.0`)  
   → inliers = 24, outliers = 16  
   → `geo_score = 24 / 40 = 0.6`
6. `geo_threshold`에 따라:
   - 0.25 → 0.6 ≥ 0.25 → 같은 장소로 인정
   - 0.4  → 0.6 ≥ 0.4  → 여전히 같은 장소로 인정
   - 0.7  → 0.6 < 0.7  → 같은 장소로 보지 않음

---

## 6. 작업 사진(전/중/후)용 추천 범위 (요약)

**목표:**  
같은 작업 장소 + 비슷한 각도에서 찍은 전/중/후는 한 클러스터,  
다른 장소나 각도 많이 다른 샷은 확실히 분리.

현실적인 기본값 범위:

- `ratio_thresh` : **0.68 ~ 0.75** (기본 0.7 근처)
- `min_good_matches` : **8 ~ 12**
- `ransac_reproj_thresh` : **3.0 ~ 4.0**
- `geo_threshold` : **0.3 ~ 0.4**

이렇게 두면,
- 나무/풀/사람 상태는 어느 정도 달라도,
- 계단, 벽, 난간, 길 곡률 같은 구조물이 비슷한 샷끼리만
  “같은 장면”으로 엮이는 경향을 갖게 됩니다.
