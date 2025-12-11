import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from PIL import ExifTags, Image

# Add project root and app directory to sys.path to import app modules
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
app_dir = project_root / "app"
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

try:
    from app.services.metadata_extractor import MetadataExtractor
except ImportError as e:
    print(f"Error: Could not import app.services.metadata_extractor: {e}")
    print("Ensure your PYTHONPATH includes the project root and 'app' directory.")
    sys.exit(1)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".bmp", ".gif"}
GPS_TAGS = {v: k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"}

extractor = MetadataExtractor()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def get_file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def process_directory(input_root: Path, output_root: Path, current_dir: Path, threshold=0.9):
    """
    current_dir: input_root 아래 탐색 중인 디렉토리
    threshold: 이미지 비율 기준 (기본값 0.9 = 90%)
    """

    # 디렉토리 아래 "직접 포함된 파일"만 조사 (하위 dir 제외)
    files = [f for f in current_dir.iterdir() if f.is_file()]
    if not files:
        return  # 파일 없으면 무시

    img_files = [f for f in files if is_image_file(f)]

    # 이미지 비율 계산
    ratio = len(img_files) / len(files)
    if ratio < threshold:
        return  # 이미지 비율 부족 → 스킵

    # 상대 경로 계산
    rel_path = current_dir.relative_to(input_root)
    out_dir_name = str(rel_path).replace("/", "-")
    out_dir = output_root / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 메타데이터 수집
    count_photos = 0
    count_gps = 0
    size_bins = {"0-1MB": 0, "1-10MB": 0, "10MB-": 0}
    date_dist = defaultdict(int)

    for img in img_files:
        exif = extractor._export_exif_sync(str(img))
        lat, lon, direction = extractor._get_gps_from_exif(exif)
        timestamp = extractor._parse_datetime_from_exif(exif)
        if timestamp:
            date = datetime.fromtimestamp(timestamp).strftime('%Y%m%d')
        else:
            date = 'NONE'
        count_photos += 1

        # 파일 사이즈 분류
        mb = get_file_size_mb(img)
        if mb < 1:
            size_bins["0-1MB"] += 1
        elif mb < 10:
            size_bins["1-10MB"] += 1
        else:
            size_bins["10MB-"] += 1

        if date:
            date_dist[date] += 1

        if lat and lon:
            count_gps += 1

        shutil.copy2(img, out_dir / img.name)

    # JSON 메타파일 생성
    metadata = {
        "사진 수": count_photos,
        "유의미한 gps 정보 있는 사진 수": count_gps,
        "사진용량": size_bins,
        "사진날짜": dict(sorted(date_dist.items()))
    }

    with open(out_dir / f"metadata-{count_gps}:{count_photos}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] {current_dir} → {out_dir} 생성 완료 (사진 {count_photos}장)")


def traverse(input_root: str, output_root: str):
    input_root = Path(input_root)
    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True)

    for dirpath, dirnames, filenames in os.walk(input_root):
        current_dir = Path(dirpath)
        process_directory(input_root, output_root, current_dir)


# ---------------- 실행 예시 ----------------
if __name__ == "__main__":
    traverse("/Volumes/SSD(APFS)/사진(프로그램용)", '/Volumes/SSD(APFS)/작업사진_구조화')