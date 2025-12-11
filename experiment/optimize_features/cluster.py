import pickle

import numpy as np
import optuna
from pyproj import Geod
from sklearn.metrics import adjusted_rand_score

try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    import hdbscan as HDBSCAN

# 기존 HybridCluster 로직을 가져오되, 파라미터를 __init__에서 받도록 수정
class TunableHybridCluster:
    def __init__(self, params: dict):
        self.geod = Geod(ellps="WGS84")
        
        # Optuna가 제안한 파라미터들
        self.strict_thresh = params['strict_thresh'] # 예: 0.15
        self.loose_thresh = params['loose_thresh']   # 예: 0.35
        self.eps = params['eps']                     # 예: 3.0
        self.max_gps_tol = params['max_gps_tol']     # 예: 50.0
        
        # 가중치 강도 조절 계수 (튜닝 대상)
        self.w_merge = params.get('w_merge', 0.1)  # 병합 시 거리 축소 비율
        self.w_split = params.get('w_split', 5.0)  # 분리 시 거리 확대 비율

    def run_clustering(self, photos, features):
        """
        이미 추출된 features를 입력받아 클러스터링만 수행 (API 호출 X)
        """
        # 1. 가중치 거리 행렬 계산
        dist_matrix = self._compute_matrix(photos, features)
        
        # 2. HDBSCAN
        try:
            clusterer = HDBSCAN(
                min_cluster_size=2,
                min_samples=2,
                metric='precomputed',
                cluster_selection_epsilon=self.eps, # 튜닝된 엡실론 사용
                cluster_selection_method='leaf'
            )
            labels = clusterer.fit_predict(dist_matrix)
        except Exception:
            # 실패 시 모두 -1(노이즈) 처리
            labels = np.full(len(photos), -1)
            
        return labels

    def _compute_matrix(self, photos, features):
        n = len(photos)
        dist_matrix = np.zeros((n, n))
        coords = np.array([[p.lat, p.lon] for p in photos])
        
        # GPS 거리 계산 (반복문 최적화를 위해 단순화하거나 geod 유지)
        # 1000장이면 반복문이 50만번 돌기 때문에, 여기서 시간이 좀 걸림.
        # 최적화: pdist 등을 쓰면 좋지만, 가중치 로직 때문에 이중 루프 유지
        
        for i in range(n):
            for j in range(i + 1, n):
                _, _, gps_dist = self.geod.inv(coords[i][1], coords[i][0], coords[j][1], coords[j][0])
                
                weight_factor = 1.0
                
                if features[i] is not None and features[j] is not None:
                    similarity = np.dot(features[i], features[j]) # features는 이미 norm 되어 있다고 가정
                    struct_dist = 1.0 - similarity
                    
                    # --- [튜닝 포인트: 동적 가중치 로직] ---
                    if struct_dist < self.strict_thresh:
                        weight_factor = self.w_merge # 예: 0.1
                    elif struct_dist > self.loose_thresh:
                        weight_factor = self.w_split # 예: 5.0
                    else:
                        # 선형 보간: strict와 loose 사이를 부드럽게 연결
                        # (복잡하면 단순 1.0 처리해도 됨)
                        slope = (self.w_split - self.w_merge) / (self.loose_thresh - self.strict_thresh)
                        weight_factor = self.w_merge + slope * (struct_dist - self.strict_thresh)

                final_dist = gps_dist * weight_factor
                
                if gps_dist > self.max_gps_tol:
                     # w_merge보다 조금이라도 크면 컷
                    if weight_factor > (self.w_merge + 0.1): 
                        final_dist = 1000.0

                dist_matrix[i][j] = dist_matrix[j][i] = final_dist
        
        return dist_matrix

# --- [Optuna Objective Function] ---

def objective(trial):
    # 1. 데이터 로드 (캐시된 데이터)
    with open("dataset_cache.pkl", "rb") as f:
        data = pickle.load(f)
    photos = data['photos']
    features = data['features']
    
    # 정답 라벨 추출 (Ground Truth)
    # PhotoMeta 객체에 'label_id' 같은 속성이 있다고 가정
    true_labels = [p.label_id for p in photos] 
    
    # 2. 하이퍼파라미터 탐색 범위 설정
    params = {
        # 구조적 유사도 임계값 (Vertex AI 기준)
        "strict_thresh": trial.suggest_float("strict_thresh", 0.10, 0.25),
        "loose_thresh": trial.suggest_float("loose_thresh", 0.30, 0.50),
        
        # HDBSCAN Epsilon (Weighted Meter)
        "eps": trial.suggest_float("eps", 1.0, 5.0),
        
        # GPS 허용 오차
        "max_gps_tol": trial.suggest_float("max_gps_tol", 30.0, 60.0),
        
        # 가중치 강도
        "w_merge": trial.suggest_float("w_merge", 0.05, 0.3), # 작을수록 강한 결합
        "w_split": trial.suggest_float("w_split", 3.0, 8.0),  # 클수록 강한 분리
    }
    
    # 논리적 제약 조건 (Strict < Loose 여야 함)
    if params["strict_thresh"] >= params["loose_thresh"]:
        # 말이 안 되는 조합은 가지치기(Pruning)
        raise optuna.TrialPruned()

    # 3. 클러스터링 실행
    clusterer = TunableHybridCluster(params)
    pred_labels = clusterer.run_clustering(photos, features)
    
    # 4. 성능 평가 (Adjusted Rand Index)
    # ARI는 1.0에 가까울수록 정답과 완벽히 일치함 (0.0은 무작위)
    score = adjusted_rand_score(true_labels, pred_labels)
    
    return score

# --- [Main Execution] ---

if __name__ == "__main__":
    # 1. 스터디 생성 (Maximize ARI)
    study = optuna.create_study(direction="maximize")
    
    # 2. 최적화 실행 (n_trials=100번 시도)
    print("Start optimization...")
    study.optimize(objective, n_trials=100)
    
    # 3. 결과 출력
    print("Best score:", study.best_value)
    print("Best params:", study.best_params)
    
    # 시각화 (Jupyter 환경이라면)
    # optuna.visualization.plot_optimization_history(study).show()
    # optuna.visualization.plot_param_importances(study).show()