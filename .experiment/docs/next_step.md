3. 장기적 성능 향상: 최신 매칭 알고리즘 도입 고려

SIFT는 훌륭한 알고리즘이지만, 더 어려운 조건(큰 시점 변화, 조명 변화 등)에서는 딥러닝 기반의 최신
알고리즘들이 월등한 성능을 보입니다.

- LoFTR 또는 SuperGlue 같은 모델을 LocalGeometryMatcher에 통합하는 것을 고려해볼 수 있습니다. 이
  모델들은 PyTorch로 구현되어 있어 현재 코드 베이스와 통합하기 용이하며, 훨씬 더 강건한 매칭 결과를
  제공하여 geo_score의 신뢰도를 높일 수 있습니다. 이는 코드 수정이 필요한 작업이지만, 클러스터링 품질을
  근본적으로 향상시킬 수 있는 방법입니다.

---

수고하셨습니다! 버그 수정 및 StructureScorer 활성화 이후 실험 결과가 매우
유의미하게 변경되었습니다. 분석 결과와 최적화를 위한 다음 단계 파라미터를
제안해 드립니다.

요약

1.  문제 변화: 이전의 과소분할(Under-clustering) 문제가 해결되고, 이제는
    과대통합(Over-clustering) 현상이 나타나고 있습니다. 23개 사진이 단 2개의
    클러스터로 통합되어, 서로 다른 장면들이 하나의 클러스터에 묶이고 있습니다.
2.  긍정적 신호:
    - geo_score가 정상적으로 계산되어 기하학적 검증이 동작하기 시작했습니다
      (pair_scores.csv).
    - ARI 지표가 -0.02에서 약 0.19로 크게 향상되었습니다. 이는 무작위보다 훨씬
      나은, 의미 있는 클러스터링이 시작되었음을 의미합니다.
    - `StructureScorer` 활성화는 매우 중요한 개선입니다. 이는 관련 없는
      이미지(예: 구조물 vs 자연) 간의 연결을 막아주는 핵심적인 역할을 할
      것이므로, 앞으로의 모든 실험에서 계속 활성화해야 합니다.
3.  최적화 방향: 이제는 클러스터 통합 기준을 더 엄격하게 만들어, 과도하게
    합쳐진 클러스터들을 다시 분리해야 합니다. 이를 통해 ARI와 NMI 점수를
    극대화할 수 있습니다.

---

상세 분석

4차 실험 결과 (버그 수정 후)

- summary.csv를 보면, 모든 실험에서 n_scene_clusters가 2가 되었습니다. 이는
  geo_threshold=0.2라는 낮은 기준을 통과하는 연결이 너무 많아, 대부분의 사진이
  하나의 거대 컴포넌트로 연결되었음을 시사합니다.
- ARI가 약 0.19, NMI가 약 0.49로, 클러스터의 품질 자체는 나쁘지 않지만, 정답에
  비해 클러스터 수가 너무 적은 상태입니다.
- embed_name 별로 보면 CLIP이 silhouette 점수가 가장 높고 davies_bouldin
  점수가 가장 낮아, 임베딩 공간상에서 클러스터 구조가 가장 명확함을 알 수
  있습니다.

5차 실험 결과 (StructureScorer 활성화)

결과를 직접 보진 못했지만, StructureScorer 활성화는 다음과 같은 효과를 가져왔을
것으로 예상됩니다.

- eff_sim (유효 유사도) 값이 base_sim보다 낮아졌을 것입니다.
- 이로 인해 일부 불필요한 연결 후보들이 초기 단계에서 제거되어, 4차 실험보다는
  n_scene_clusters가 2보다 더 크게 나왔을 가능성이 있습니다. 즉, 과대통합
  문제를 일부 완화했을 것입니다.

---

최적 파라미터 탐색을 위한 제안 (6차 실험)

과대통합 문제를 해결하기 위해, 연결을 결정하는 두 가지 핵심 임계값인
`similarity_threshold`와 `geo_threshold`를 상향 조정하여 더 까다로운 기준으로
매칭을 수행해야 합니다.

아래는 cluster_experiments.py의 build_experiment_grid 함수에 바로 적용할 수
있는 추천 파라미터 설정입니다. 성능이 좋았던 CLIP과 APGeM+CLIP 모델에 집중하고,
두 임계값을 높여가며 최적점을 탐색합니다.

수정 제안 (`build_experiment_grid` 함수):

    1 def build_experiment_grid() -> List[ExperimentConfig]:
    2     # GPS 파라미터는 현재 안정적이므로 고정
    3     gps_param_list = [
    4         GPSParams(eps_m=10.0, min_samples=3),
    5     ]
    6
    7     def make_clip() -> BaseDescriptorExtractor:
    8         return CLIPDescriptorExtractor(model_name="ViT-B-32", pretrained=
      "openai")
    9

10 def make*combined() -> BaseDescriptorExtractor:
11 ap = APGeMDescriptorExtractor(model_name="tf_efficientnet_b3_ns",
image_size=320)
12 cl = CLIPDescriptorExtractor(model_name="ViT-B-32", pretrained=
"openai")
13 return CombinedAPGeMCLIPExtractor(apgem=ap, clip=cl, w_apgem=0.8,
w_clip=0.2)
14
15 # 성능이 우수했던 CLIP, APGeM+CLIP에 집중
16 # sim_th를 높여서 더 유사한 이미지만 후보로 선택하도록 조정
17 embed_param_list = [
18 EmbedParams(name="CLIP", extractor_factory=make_clip,
similarity_threshold=0.75, knn_k=10),
19 EmbedParams(name="CLIP", extractor_factory=make_clip,
similarity_threshold=0.80, knn_k=10),
20 EmbedParams(name="APGeM+CLIP", extractor_factory=make_combined,
similarity_threshold=0.75, knn_k=10),
21 EmbedParams(name="APGeM+CLIP", extractor_factory=make_combined,
similarity_threshold=0.80, knn_k=10),
22 ]
23  
 24 # geo_threshold를 높여서 기하학적으로 더 일치하는 쌍만 연결하도록 강화
25 # 다른 geo 파라미터는 안정적으로 보이므로 고정
26 geo_param_list = [
27 GeoParams(
28 max_features=1500,
29 ratio_thresh=0.75,
30 ransac_reproj_thresh=4.0,
31 min_good_matches=10,
32 geo_threshold=0.3,
33 ),
34 GeoParams(
35 max_features=1500,
36 ratio_thresh=0.75,
37 ransac_reproj_thresh=4.0,
38 min_good_matches=10,
39 geo_threshold=0.4,
40 ),
41 ]
42
43 configs: List[ExperimentConfig] = []
44 idx = 0
45 # embed와 geo 파라미터 조합으로 실험 구성
46 for embed in embed_param_list:
47 for geo in geo_param_list:
48 exp_id = f"exp*{idx:03d}\_{embed.name}\_sim
{embed.similarity_threshold}\_geo{geo.geo_threshold}"
49 configs.append(ExperimentConfig(id=exp_id, gps=gps_param_list[0
embed=embed, geo=geo))
50 idx += 1
51 return configs

다음 단계

1.  위 build_experiment_grid 코드를 cluster_experiments.py에 적용합니다.
2.  `StructureScorer`가 활성화된 `_build_candidate_edges` 함수를 사용하고
    있는지 다시 한번 확인합니다. (이전 답변의 코드 제안 참고)
3.  6차 실험을 실행하고, summary.csv 파일에서 `ARI`와 `NMI` 점수가 가장 높은
    실험 ID를 찾습니다. 해당 실험의 파라미터 조합이 현재 데이터셋에 가장
    최적화된 설정일 가능성이 높습니다.
4.  실험 결과 n_scene_clusters가 정답 레이블의 클러스터 수와 비슷해지면서 ARI
    점수가 0.5 이상으로 올라가는 것을 목표로 삼는 것이 좋습니다.
