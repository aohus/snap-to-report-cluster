import asyncio
import io
import json
import logging
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import List

import aiofiles
import aiofiles.os
import aioshutil
from app.core.config import JobConfig
from app.models.photometa import PhotoMeta
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


import asyncio
import io
import json
import logging
import math
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import List

import aiofiles
import aiofiles.os
import aioshutil
import matplotlib.pyplot as plt
from app.core.config import JobConfig
from app.models.photometa import PhotoMeta
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)


class OutputGenerator:
    def __init__(self, config: JobConfig):
        self.config = config
        self.output_root = Path(self.config.IMG_OUTPUT_DIR)
        self.report_root = Path(self.config.REPORT_DIR)

    def save_cluster_visualization_sync(self, scene_groups: List[List[PhotoMeta]], dpi: int = 400):
        """
        위치 기반 클러스터 시각화 (고해상도 버전).

        - dpi: 출력 이미지 해상도(기본 400). 보고 낮으면 600까지 올려도 됨.
        """
        logger.info("Generating high-resolution cluster visualization map...")

        # 먼저 전체 좌표 범위 계산
        all_lats = []
        all_lons = []
        for group in scene_groups:
            for photo in group:
                if photo.lat is not None and photo.lon is not None:
                    all_lats.append(photo.lat)
                    all_lons.append(photo.lon)

        if not all_lats:
            logger.warning("No photos with GPS data found for visualization.")
            return

        lat_min, lat_max = min(all_lats), max(all_lats)
        lon_min, lon_max = min(all_lons), max(all_lons)

        # 여유 margin
        lat_margin = (lat_max - lat_min) * 0.05 or 1e-4
        lon_margin = (lon_max - lon_min) * 0.05 or 1e-4

        lat_min -= lat_margin
        lat_max += lat_margin
        lon_min -= lon_margin
        lon_max += lon_margin

        # 위경도 비율에 맞게 figure 비율 설정
        lat_range = lat_max - lat_min
        lon_range = lon_max - lon_min
        aspect = lon_range / lat_range if lat_range > 0 else 1.0

        base_size = 12  # 기준 크기 (inches)
        width = base_size * aspect if aspect > 1 else base_size
        height = base_size / aspect if aspect > 1 else base_size

        fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)

        # 클러스터별 색상
        colors = [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in scene_groups]

        for i, group in enumerate(scene_groups):
            lats = [p.lat for p in group if p.lat is not None]
            lons = [p.lon for p in group if p.lon is not None]

            if not lats:
                continue

            ax.scatter(
                lons,
                lats,
                color=colors[i],
                label=f"Cluster {i+1} ({len(lats)})",
                s=4,        # 점 크기 (기존 1 → 8로 키움)
                alpha=0.7,  # 살짝 투명
                linewidths=0,
            )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Photo Cluster Visualization (High Resolution)")

        ax.set_xlim(lon_min, lon_max)
        ax.set_ylim(lat_min, lat_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.3, alpha=0.5)

        # 클러스터가 너무 많으면 legend가 그림을 가릴 수 있으니 위치/폰트 조정
        ax.legend(loc="upper right", fontsize=8, markerscale=1.5, framealpha=0.8)

        plt.tight_layout()

        output_path = self.report_root / "cluster_visualization.png"
        try:
            fig.savefig(output_path, dpi=dpi)
            logger.info(f"Cluster visualization saved to: {output_path} (dpi={dpi})")
        except Exception as e:
            logger.error(f"Failed to save cluster visualization: {e}")
        finally:
            plt.close(fig)

    async def copy_and_rename_group(self, group: List[PhotoMeta], scene_idx: int) -> Path:
        sorted_group = sorted(
            group,
            key=lambda p: (p.timestamp is None, p.timestamp if p.timestamp is not None else 0.0, p.path)
        )

        scene_dir = self.output_root / f"scene_{scene_idx:03d}"
        await aiofiles.os.makedirs(scene_dir, exist_ok=True)

        copy_tasks = []
        for i, photo in enumerate(sorted_group, start=1):
            src = Path(photo.path)
            ext = src.suffix.lower()
            dst_name = f"img_{i:03d}{ext}"
            dst = scene_dir / dst_name
            copy_tasks.append(aioshutil.copy2(src, dst))
        
        await asyncio.gather(*copy_tasks)
        return scene_dir

    async def save_group_montage(self, scene_dir: Path):
        try:
            image_files = sorted(
                [p for p in await aiofiles.os.scandir(scene_dir) if p.is_file() and p.name.lower().endswith((".jpg", ".jpeg", ".png"))]
            )
        except FileNotFoundError:
            logger.warning(f"Scene directory not found for montage: {scene_dir}")
            return

        if not image_files:
            return

        thumbs = []
        for f in image_files:
            try:
                # Note: Pillow's image processing is synchronous and will block the event loop.
                thumb = self._create_thumbnail(Path(f.path))
                thumbs.append(thumb)
            except Exception as e:
                logger.warning(f"Failed to create thumbnail for {f.path}: {e}")

        if not thumbs:
            return

        # Note: Pillow's image creation is synchronous and will block the event loop.
        montage = self._create_montage_image(thumbs)
        montage_path = scene_dir / "scene_montage.jpg"
        
        buffer = io.BytesIO()
        montage.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)

        async with aiofiles.open(montage_path, 'wb') as f:
            await f.write(buffer.read())

    def _create_thumbnail(self, file_path: Path) -> Image.Image:
        img = Image.open(file_path)
        img = ImageOps.exif_transpose(img)
        img.thumbnail(self.config.THUMB_SIZE)
        return img

    def _create_montage_image(self, thumbs: List[Image.Image]) -> Image.Image:
        cols = self.config.MONTAGE_COLS
        rows = math.ceil(len(thumbs) / cols)
        thumb_w, thumb_h = self.config.THUMB_SIZE
        montage_w = cols * thumb_w
        montage_h = rows * thumb_h

        montage = Image.new("RGB", (montage_w, montage_h), color=(240, 240, 240))

        for idx, thumb in enumerate(thumbs):
            row, col = idx // cols, idx % cols
            x = col * thumb_w + (thumb_w - thumb.width) // 2
            y = row * thumb_h + (thumb_h - thumb.height) // 2
            montage.paste(thumb, (x, y))
        return montage

    async def create_mosaic(self, scene_groups: List[List[PhotoMeta]]):
        if not scene_groups:
            logger.info("No clusters to visualize in mosaic.")
            return
        
        # Note: This image creation part is synchronous and will block the event loop.
        mosaic = self._create_mosaic_image(scene_groups)
        
        mosaic_path = Path(self.config.IMG_OUTPUT_DIR) / "clusters_mosaic.jpg"
        
        buffer = io.BytesIO()
        mosaic.save(buffer, format="JPEG", quality=90)
        buffer.seek(0)

        async with aiofiles.open(mosaic_path, 'wb') as f:
            await f.write(buffer.read())

        logger.info(f"Mosaic saved to: {mosaic_path}")

    def _create_mosaic_image(self, scene_groups: List[List[PhotoMeta]]) -> Image.Image:
        rows_paths = [
            [p.path for p in sorted(g, key=lambda p: (p.timestamp is None, p.timestamp if p.timestamp is not None else 0.0, p.path))[:self.config.MONTAGE_COLS]]
            for g in scene_groups
        ]
        
        num_rows = len(rows_paths)
        num_cols = max(len(row) for row in rows_paths) if rows_paths else 0
        if num_cols == 0:
            return Image.new("RGB", (1, 1))

        thumb_w, thumb_h = self.config.THUMB_SIZE
        mosaic = Image.new("RGB", (num_cols * thumb_w, num_rows * thumb_h), (255, 255, 255))

        for row_idx, paths in enumerate(rows_paths):
            for col_idx, img_path in enumerate(paths):
                try:
                    thumb = self._create_thumbnail(Path(img_path))
                    thumb_bg = Image.new("RGB", self.config.THUMB_SIZE, (240, 240, 240))
                    x = (self.config.THUMB_SIZE[0] - thumb.size[0]) // 2
                    y = (self.config.THUMB_SIZE[1] - thumb.size[1]) // 2
                    thumb_bg.paste(thumb, (x, y))
                except Exception as e:
                    logger.warning(f"Failed to open {img_path} for mosaic: {e}")
                    thumb_bg = Image.new("RGB", self.config.THUMB_SIZE, (200, 200, 200))
                
                mosaic.paste(thumb_bg, (col_idx * thumb_w, row_idx * thumb_h))
        return mosaic

    async def save_meta(self, scene_groups: List[List[PhotoMeta]]):
        meta_result = {"meta": {"stats": {}, "scene": {}}, "scenes": {}}
        scene_info = {}
        cluster_size = []
        for idx, scene_group in enumerate(scene_groups, start=1):
            loc = f"s_{idx:03d}"
            scene_info[loc] = len(scene_group)
            cluster_size.append(len(scene_group))
            scene_details = {
                os.path.basename(scene.path): {
                    "lat": scene.lat, "lon": scene.lon, "alt": scene.alt, "timestamp": scene.timestamp
                } for scene in scene_group
            }
            meta_result["scenes"][loc] = scene_details
        meta_result["meta"]["info"] = scene_info
        meta_result["meta"]["stats"] = {
            "평균": sum(cluster_size) / len(cluster_size),
            "분포": Counter(cluster_size)
        }

        await aiofiles.os.makedirs(self.report_root, exist_ok=True)
        filepath = self.report_root / f"report_{time.time()}.json"
        
        async with aiofiles.open(filepath, "w", encoding="utf-8") as f:
            await f.write(json.dumps(meta_result, indent=4, ensure_ascii=False))
        logger.info(f"Saved metadata for {len(scene_groups)} scenes to {filepath}")
