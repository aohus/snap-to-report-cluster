import asyncio
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
from PIL.ExifTags import GPSTAGS, TAGS

# Allow loading truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add cluster-backend/app to sys.path so we can import from there
# assuming this script is in cluster-backend/.experiment/
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
app_dir = project_root / 'app'

sys.path.append(str(app_dir))
sys.path.append(str(project_root)) # For 'config' if it's in root of app

try:
    from common.models import PhotoMeta
    from config import ClusteringConfig
    from services.clustering.gps import GPSClusterer
except ImportError:
    # Fallback if structure is different (e.g. app)
    # Try adding app
    sys.path.pop()
    app_dir = project_root / 'app'
    sys.path.append(str(app_dir))
    from common.models import PhotoMeta
    from config import ClusteringConfig
    from services.clustering.gps import GPSClusterer

def get_decimal_from_dms(dms, ref):
    degrees = dms[0]
    minutes = dms[1]
    seconds = dms[2]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def get_gps_from_image(image_path):
    try:
        img = Image.open(image_path)
        exif = img._getexif()
        if not exif:
            return None, None, None

        gps_info = {}
        for tag, value in exif.items():
            decoded = TAGS.get(tag, tag)
            if decoded == 'GPSInfo':
                for t in value:
                    sub_decoded = GPSTAGS.get(t, t)
                    gps_info[sub_decoded] = value[t]
        
        if not gps_info:
            return None, None, None

        lat = None
        lon = None
        timestamp = None

        if 'GPSLatitude' in gps_info and 'GPSLatitudeRef' in gps_info:
            lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
        
        if 'GPSLongitude' in gps_info and 'GPSLongitudeRef' in gps_info:
            lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
            
        return lat, lon, timestamp
    except Exception as e:
        # print(f"Error extracting GPS from {image_path}: {e}")
        return None, None, None

async def process_dataset(dataset_name, assets_dir):
    print(f"\n=== Processing Dataset: {dataset_name} ===")
    
    # Find the actual directory handling unicode composition
    target_dir = None
    for d in os.listdir(assets_dir):
        # Normalize check? Or just loose match
        if dataset_name in d or d in dataset_name: # rough check
             if d.startswith(dataset_name[0:2]): # Check '1차' part
                 target_dir = assets_dir / d
                 if dataset_name in str(target_dir): # Ensure match
                     break
                 # If exact match failed but it looks similar (NFC/NFD), use it
                 if dataset_name.replace('_', '') in d.replace('_', ''):
                     target_dir = assets_dir / d
                     break

    if not target_dir or not target_dir.exists():
        # Try direct path
        target_dir = assets_dir / dataset_name
        if not target_dir.exists():
            print(f"Dataset directory not found: {dataset_name}")
            # Try to find by listing
            candidates = [p for p in assets_dir.iterdir() if p.is_dir()]
            print(f"Available directories: {[p.name for p in candidates]}")
            return

    print(f"Using directory: {target_dir}")
    
    photos = []
    count = 0
    
    # Recursive glob or just top level? Assuming top level for now based on previous `set1`
    for file_path in target_dir.glob("*"):
        if file_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
            
        lat, lon, _ = get_gps_from_image(file_path)
        if lat is not None and lon is not None:
            photos.append(PhotoMeta(
                id=file_path.name,
                path=str(file_path),
                original_name=file_path.name,
                lat=lat,
                lon=lon,
                timestamp=count 
            ))
        count += 1
        
    print(f"Found {len(photos)} photos with GPS out of {count} files.")
    
    if not photos:
        print("No photos with GPS found.")
        return

    config = ClusteringConfig()
    clusterer = GPSClusterer(config)
    clusters = await clusterer.cluster(photos)
    
    print(f"Result: {len(clusters)} clusters found.")
    
    sizes = [len(c) for c in clusters]
    if not sizes:
        return

    avg_size = np.mean(sizes)
    median_size = np.median(sizes)
    
    print(f"Cluster sizes: {sorted(sizes)}")
    print(f"Average size: {avg_size:.2f}")
    print(f"Median size: {median_size:.2f}")
    
    # Distribution
    from collections import Counter
    dist = Counter(sizes)
    print("Size distribution:")
    for size in sorted(dist.keys()):
        print(f"  Size {size}: {dist[size]} clusters")

async def main():
    assets_dir = Path(project_root) / 'assets'
    if not assets_dir.exists():
        # Try referencing from current working dir if project_root calculation failed
        assets_dir = Path("cluster-backend/assets")
    
    datasets = ["3차", "4차"]
    
    for ds in datasets:
        await process_dataset(ds, assets_dir)

if __name__ == "__main__":
    asyncio.run(main())
