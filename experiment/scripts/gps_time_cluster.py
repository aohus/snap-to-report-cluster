from __future__ import annotations

import asyncio
import csv
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# matplotlib
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    
from domain.extractors.metadata_extractor import MetadataExtractor
from photometa import PhotoMeta
from PIL import Image, ImageFile

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# sklearn
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

DEVICE = None

from dataclasses import dataclass
from typing import Optional


@dataclass
class PairDebugRecord:
    photo_path_i: str
    photo_path_j: str
    gps_cluster_index: int  # 이 쌍이 속한 GPS 클러스터 번호

# -------------------------------------------------------------------
# Stage 1: GPS DBSCAN
# -------------------------------------------------------------------

# EARTH_RADIUS_M = 6371000.0  # meters


# def latlon_to_xy_m(lat: float, lon: float, lat0: float, lon0: float) -> Tuple[float, float]:
#     lat_rad = math.radians(lat)
#     lon_rad = math.radians(lon)
#     lat0_rad = math.radians(lat0)
#     lon0_rad = math.radians(lon0)
#     x = (lon_rad - lon0_rad) * math.cos((lat_rad + lat0_rad) / 2.0) * EARTH_RADIUS_M
#     y = (lat_rad - lat0_rad) * EARTH_RADIUS_M
#     return x, y


# def gps_dbscan_cluster(
#     photos: List[PhotoMeta],
#     eps_m: float,
#     min_samples: int,
# ) -> List[int]:
#     if not photos:
#         return []
#     lat0 = sum(p.lat for p in photos) / len(photos)
#     lon0 = sum(p.lon for p in photos) / len(photos)

#     coords = np.array(
#         [latlon_to_xy_m(p.lat, p.lon, lat0, lon0) for p in photos],
#         dtype=np.float32,
#     )
#     db = DBSCAN(eps=eps_m, min_samples=min_samples, metric="euclidean")
#     labels = db.fit_predict(coords)
#     return labels.tolist()

def make_gps_cluster(config):
    return GPSClusterer(config)

@dataclass
class GPSParams:
    eps_m: float
    min_samples: int


@dataclass
class ExperimentConfig:
    id: str
    gps: GPSParams


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    n_photos: int
    n_gps_clusters: int
    n_scene_clusters: int
    metrics: Dict[str, float]
    stage_times: Dict[str, float]


# -------------------------------------------------------------------
# 지표 계산
# -------------------------------------------------------------------
def compute_external_metrics(
    true_labels: List[int],
    pred_labels: List[int],
) -> Dict[str, float]:
    t = np.array(true_labels)
    p = np.array(pred_labels)
    mask = t >= 0
    t = t[mask]
    p = p[mask]

    metrics: Dict[str, float] = {}
    metrics["ARI"] = float(adjusted_rand_score(t, p))
    metrics["NMI"] = float(normalized_mutual_info_score(t, p))
    return metrics



# -------------------------------------------------------------------
# 시각화: 지도 / Mosaic
# -------------------------------------------------------------------

def plot_map_clusters(
    photos: List[PhotoMeta],
    labels: List[int],
    out_path: Path,
) -> None:
    labels_arr = np.array(labels)
    lat = np.array([p.lat for p in photos])
    lon = np.array([p.lon for p in photos])

    plt.figure(figsize=(8, 8))
    unique_labels = sorted(set(labels_arr))
    for lbl in unique_labels:
        mask = labels_arr == lbl
        if lbl == -1:
            plt.scatter(lon[mask], lat[mask], s=10, alpha=0.3, label="noise", marker="x")
        else:
            plt.scatter(lon[mask], lat[mask], s=20, alpha=0.7, label=f"cluster {lbl}")

    plt.xlabel("longitude")
    plt.ylabel("latitude")
    plt.title("Cluster result on map")
    plt.legend(markerscale=1.5, fontsize="small", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def create_mosaic_for_experiment(
    exp_id: str,
    photos: List[PhotoMeta],
    labels: List[int],
    out_path: Path,
    thumb_size: Tuple[int, int] = (256, 256),
    max_photos_per_cluster: int = 5,
) -> None:
    labels_arr = np.array(labels)
    unique_labels = [lbl for lbl in sorted(set(labels_arr)) if lbl >= 0]
    if not unique_labels:
        logger.warning(f"[{exp_id}] no clusters to visualize.")
        return

    n_clusters = len(unique_labels)
    thumb_w, thumb_h = thumb_size
    cols = max_photos_per_cluster
    rows = n_clusters

    mosaic_w = cols * thumb_w
    mosaic_h = rows * thumb_h

    mosaic = Image.new("RGB", (mosaic_w, mosaic_h), color=(255, 255, 255))

    for row_idx, lbl in enumerate(unique_labels):
        indices = [i for i, l in enumerate(labels_arr) if l == lbl]
        cluster_photos = sorted(
            [photos[i] for i in indices],
            key=lambda p: (p.timestamp or datetime.min, str(p.id)),
        )
        cluster_photos = cluster_photos[:max_photos_per_cluster]

        for col_idx, p in enumerate(cluster_photos):
            try:
                img = Image.open(p.path).convert("RGB")
            except Exception:
                continue
            img = img.resize(thumb_size)
            x0 = col_idx * thumb_w
            y0 = row_idx * thumb_h
            mosaic.paste(img, (x0, y0))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mosaic.save(out_path)


# -------------------------------------------------------------------
# summary.csv 기록
# -------------------------------------------------------------------

def write_summary_row(
    csv_path: Path,
    exp: ExperimentConfig,
    n_photos: int,
    n_gps_clusters: int,
    metrics: Dict[str, float],
    stage_times: Dict[str, float],
) -> None:
    fieldnames = [
        "exp_id",
        "gps_eps_m",
        "gps_min_samples",
        "n_photos",
        "n_gps_clusters",
        "time_gps_dbscan",
        "ARI",
        "NMI",
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {
            "exp_id": exp.id,
            "gps_eps_m": exp.gps.eps_m,
            "gps_min_samples": exp.gps.min_samples,
            "n_photos": n_photos,
            "n_gps_clusters": n_gps_clusters,
            "time_gps_dbscan": stage_times.get("gps_dbscan", float("nan")),
            "ARI": f"{metrics.get("ARI", float("nan")):.4f}",
            "NMI": f"{metrics.get("NMI", float("nan")):.4f}",
        }
        writer.writerow(row)


def write_debug(pair_csv_path, pair_records):
    with pair_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "photo_path_i",
            "photo_path_j",
            "gps_cluster_index",
        ])
        for r in pair_records:
            writer.writerow([
                r.photo_path_i,
                r.photo_path_j,
                r.gps_cluster_index,
                f"{r.base_sim:.6f}",
                f"{r.eff_sim:.6f}",
                int(r.passed_sim),
                "" if r.geo_score is None else f"{r.geo_score:.6f}",
                "" if r.passed_geo is None else int(r.passed_geo),
            ])

# -------------------------------------------------------------------
# 실험 1개 실행
# -------------------------------------------------------------------

def run_one_experiment(
    exp: ExperimentConfig,
    photos: List[PhotoMeta],
    true_labels: Optional[List[int]],
    base_out_dir: Path,
    scorer: dict,
) -> ExperimentResult:
    exp_dir = base_out_dir / exp.id
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Stage1: GPS
    t0 = time.perf_counter()
    gps = make_gps_cluster()
    # gps_labels = gps(photos, eps_m=exp.gps.eps_m, min_samples=exp.gps.min_samples)
    gps_labels = gps.cluster(photos, config)
    t1 = time.perf_counter()
    stage_times: Dict[str, float] = {"gps_dbscan": t1 - t0}
    n_gps_clusters = len({l for l in gps_labels if l >= 0})

    metrics: Dict[str, float] = {}

    if true_labels is not None:
        external_metrics = compute_external_metrics(true_labels, gps_labels)
        metrics.update(external_metrics)

    # 시각화 저장
    plot_map_clusters(photos, gps_labels, out_path=exp_dir / f"{exp.id}_map.png")
    create_mosaic_for_experiment(exp.id, photos, gps_labels, out_path=exp_dir / f"{exp.id}_mosaic.png")

    # summary.csv
    summary_csv = base_out_dir / "summary.csv"
    write_summary_row(summary_csv, exp, n_photos, n_gps_clusters, metrics, stage_times)

    pair_csv_path = base_out_dir / "pair_scores.csv"
    write_debug(pair_csv_path=pair_csv_path, pair_records=pair_records)

    return ExperimentResult(
        config=exp,
        n_photos=n_photos,
        n_gps_clusters=n_gps_clusters,
        metrics=metrics,
        stage_times=stage_times,
    )


def build_experiment_grid() -> List[ExperimentConfig]:
    gps_param_list = [
        GPSParams(eps_m=10.0, min_samples=3),
    ]

def run_all_experiments(
    photos: List[PhotoMeta],
    true_labels: Optional[List[int]],
    out_dir: Path,
) -> List[ExperimentResult]:
    configs = build_experiment_grid()

    results: List[ExperimentResult] = []
    for exp in configs:
        logger.info(f"=== Running {exp.id} ===")
        res = run_one_experiment(exp, photos, true_labels, out_dir)
        results.append(res)
    return results


async def get_photos(image_dir):
    image_paths = [
        str(Path(image_dir) / fname)
        for fname in os.listdir(image_dir)
        if fname.lower().endswith(("png", "jpg", "jpeg"))
    ]
    ext = MetadataExtractor()
    tasks = [ext.extract(p) for p in image_paths]
    return await asyncio.gather(*tasks)


if __name__ == "__main__":
    # 예시: photos 리스트와 true_labels 는 실제 코드에서 채워 넣어야 함
    photos: List[PhotoMeta] = []
    true_labels: Optional[List[int]] = None
    out_dir = Path("./.experiment/exp_results/11th")

    image_dir = "/Users/aohus/Workspaces/github/job-report-creator/backend/assets/어제/"
    photos = asyncio.run(get_photos(image_dir))
    true_labels = [int(photo.original_name.split("-")[0]) for photo in photos]
    logger.info(f"true labels: {true_labels}")
    results = run_all_experiments(photos, true_labels, out_dir)


