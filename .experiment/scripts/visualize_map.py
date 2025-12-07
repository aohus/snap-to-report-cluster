import argparse
import asyncio
import base64
import colorsys
import glob
import math
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required. Install it via 'pip install Pillow'")
    sys.exit(1)

try:
    import folium
    from folium import plugins
except ImportError:
    print("Error: folium is required. Install it via 'pip install folium'")
    sys.exit(1)

def generate_distinct_colors(n: int) -> List[str]:
    """Generate n visually distinct hex colors using Golden Ratio for Hue."""
    colors = []
    hue = 0.5 # Fixed start for reproducibility
    golden_ratio_conjugate = 0.618033988749895
    
    for i in range(n):
        hue += golden_ratio_conjugate
        hue %= 1.0
        
        # Vary saturation and value to add dimension to distinction
        # Cycle through 3 levels of saturation and 2 levels of brightness
        saturation = 0.7 + (i % 3) * 0.1  # 0.7, 0.8, 0.9
        value = 0.8 + (i % 2) * 0.15      # 0.8, 0.95
        
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
        )
        colors.append(hex_color)
    return colors

def image_to_base64(image_path: str, size: Tuple[int, int] = (400, 400)) -> str:
    """Converts an image to a base64 encoded data URL string (thumbnail)."""
    try:
        img = Image.open(image_path)
        # Convert to RGB if RGBA to allow JPEG saving
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        img.thumbnail(size)
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=80)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return ""

def load_ground_truth(sql_dump_path: str) -> Dict[str, str]:
    """
    Parses the SQL dump to create a mapping from original_filename to cluster_id.
    """
    if not os.path.exists(sql_dump_path):
        print(f"Warning: SQL dump not found at {sql_dump_path}. Clustering info will be missing.")
        return {}

    mapping = {}
    try:
        with open(sql_dump_path, 'r', encoding='utf-8') as f:
            in_copy_block = False
            for line in f:
                line = line.strip()
                if line.startswith("COPY public.photos"):
                    in_copy_block = True
                    continue
                if line == r"\\.":
                    in_copy_block = False
                    continue

                if in_copy_block:
                    # format: id job_id cluster_id original_filename ...
                    parts = line.split('\t')
                    if len(parts) < 4:
                        continue
                    
                    cluster_id = parts[2]
                    original_filename = parts[3]
                    
                    if cluster_id != r'\N':
                        mapping[original_filename] = cluster_id
    except Exception as e:
        print(f"Error reading SQL dump: {e}")
    
    return mapping

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates distance in meters between two lat/lon points."""
    R = 6371000.0  # Earth radius in meters
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def format_timestamp(ts):
    """datetime 또는 pandas.Timestamp를 'MM/DD HH:MM:SS' 문자열로."""
    return ts.strftime("%m/%d %H:%M:%S")

def format_timedelta(delta):
    """timedelta를 'D, H:MM:SS' 문자열로."""
    days = delta.days
    seconds = delta.seconds
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{days}, {hours}:{minutes:02d}:{seconds:02d}"


def jitter_ring_for_duplicates(
    photos: List[Dict[str, any]],
    radius_m: float = 2.0,
    lat_key: str = "lat",
    lon_key: str = "lon",
    j_lat_key: str = "j_lat",
    j_lon_key: str = "j_lon",
) -> List[Dict[str, any]]:
    """
    B 방식 jittering:
    - 동일 (lat, lon)을 갖는 photo 들을 한 그룹으로 보고
    - 그룹 내 각 photo 를 반지름 radius_m (meter) 원 위에 균등 각도로 배치
    - 결과는 각 photo dict 에 j_lat, j_lon 키로 추가해서 반환

    photos: [{"lat": ..., "lon": ..., ...}, ...]
    """
    # 1) 같은 좌표 그룹 만들기 (float 오차 줄이려고 소수점 n자리 반올림)
    groups = defaultdict(list)  # (lat_round, lon_round) -> [index...]
    for idx, p in enumerate(photos):
        key = (round(p[lat_key], 7), round(p[lon_key], 7))
        groups[key].append(idx)

    # 2) 각 그룹에 대해 원 둘레 균등 배치
    for key, idx_list in groups.items():
        n = len(idx_list)
        # 사진이 1개뿐이면 jitter 없이 원래 좌표 그대로 복사
        if n == 1:
            i = idx_list[0]
            p = photos[i]
            p[j_lat_key] = p[lat_key]
            p[j_lon_key] = p[lon_key]
            continue

        base_lat = photos[idx_list[0]][lat_key]
        base_lon = photos[idx_list[0]][lon_key]
        lat_rad = math.radians(base_lat)

        meters_per_deg_lat = 111_320.0
        meters_per_deg_lon = meters_per_deg_lat * math.cos(lat_rad)
        if meters_per_deg_lon == 0:  # 극지방 보호용, 거의 안 쓰이겠지만 방어코드
            meters_per_deg_lon = 1e-9

        for k, i in enumerate(idx_list):
            theta = 2.0 * math.pi * k / n
            r = radius_m

            dlat = (r * math.cos(theta)) / meters_per_deg_lat
            dlon = (r * math.sin(theta)) / meters_per_deg_lon

            p = photos[i]
            p[j_lat_key] = base_lat + dlat
            p[j_lon_key] = base_lon + dlon

    return photos

def create_map(photos: List[Dict], output_file: str = "map.html", cluster_map: Dict[str, str] = {}):
    if not photos:
        print("No photos with GPS data found.")
        return

    # Calculate center
    avg_lat = sum(p['lat'] for p in photos) / len(photos)
    avg_lon = sum(p['lon'] for p in photos) / len(photos)

    # Increased zoom level and max_zoom
    m = folium.Map(location=[avg_lat, avg_lon], zoom_start=20, max_zoom=25, control_scale=True)

    # Add Google Satellite layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=y&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Satellite',
        overlay=False,
        control=True,
        maxNativeZoom=18,
        maxZoom=25
    ).add_to(m)
    
    # Add Google Maps (Standard) layer
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
        attr='Google',
        name='Google Maps',
        overlay=False,
        control=True,
        maxNativeZoom=18,
        maxZoom=25
    ).add_to(m)

    # Add Measure Control to allow measuring distance between points
    m.add_child(plugins.MeasureControl(
        position='topright',
        primary_length_unit='meters',
        secondary_length_unit='kilometers',
        primary_area_unit='sqmeters',
        active_color='#ff0000',
        completed_color='#00ff00'
    ))
    
    # Assign distinct colors to clusters
    unique_clusters = sorted(list(set(cluster_map.values())))
    distinct_colors = generate_distinct_colors(len(unique_clusters))
    cluster_color_map = {cid: distinct_colors[i] for i, cid in enumerate(unique_clusters)}

    photos = jitter_ring_for_duplicates(photos, radius_m=2.0)

    is_cluster_ver = True
    groups = defaultdict(list)
    if is_cluster_ver:
        for photo in photos:
            groups[cluster_map.get(photo['name'])].append(photo)
    else:
        groups = {'all': photos}
    print(f"Processing {len(groups), len(photos)} photos for map generation...")
    
    # Collect all coordinates to fit map bounds later
    all_lat_lons = []

    for c, group in groups.items():
        # print("=======================")
        # print([name for name, cls in cluster_map.items() if cls == c])
        # print([p['name'] for p in group])
        # Sort photos by time
        path_coords = []
        sorted_photos = sorted(group, key=lambda x: x['timestamp'])
        for i, p in enumerate(sorted_photos):
            lat, lon, timestamp, name = p['j_lat'], p['j_lon'], p['timestamp'], p['name']
            all_lat_lons.append([lat, lon])
            name = p['name']
            ts = p['timestamp']
            # if ts.date() == date(2023, 8, 19):
            #     continue
            image_path = p['path']
            
            # Embed image as Base64
            img_base64 = image_to_base64(image_path)
            
            cluster_id = cluster_map.get(name)
            cluster_color = cluster_color_map.get(cluster_id, '#808080') # Grey for unknown/noise
            cluster_label = cluster_id if cluster_id else "Unknown"

            # Calculate distance to next point
            next_str = "<br>Next: <b>None</b>"
            if i < len(sorted_photos) - 1:
                next_p = sorted_photos[i+1]
                is_same_cls = "Y" if cluster_map.get(next_p['name']) == cluster_id else "N"
                time_diff = format_timedelta(next_p['timestamp'] - timestamp)
                dist = haversine_distance(lat, lon, next_p['j_lat'], next_p['j_lon'])
                next_str = f"<br>Next: <b>{is_same_cls} | {time_diff} | {dist:.2f} m</b>"
            
            # Calculate distance from prev point
            prev_str = "<br>Prev: <b>None</b>"
            if i > 0:
                prev_p = sorted_photos[i-1]
                is_same_cls = "Y" if cluster_map.get(prev_p['name']) == cluster_id else "N"
                time_diff = format_timedelta(timestamp - prev_p['timestamp'])
                dist = haversine_distance(lat, lon, prev_p['j_lat'], prev_p['j_lon'])
                prev_str = f"<br>Prev: <b>{is_same_cls} | {time_diff} | {dist:.2f} m</b>"

            popup_html = f"""
            <div style="width:320px">
                <b>{name}</b><br>
                <img src="{img_base64}" width="100%" style="border:1px solid #ccc; margin: 5px 0;"><br>
                Cluster: <b>{cluster_label}</b><br>
                Time: {format_timestamp(ts)}<br>

                {prev_str}
                {next_str}
            </div>
            """
            
            # Use CircleMarker for better customization
            folium.CircleMarker(
                location=[lat, lon],
                radius=6,
                # popup=folium.Popup(popup_html, max_width=350),
                # tooltip=f"{i+1}. {name} ({cluster_label})",
                tooltip=folium.Tooltip(
                    popup_html,
                    max_width=350,
                    sticky=False  # 마우스 따라다니게 하려면 True 로
                ),
                color='black',      # Border color
                weight=1,
                fill=True,
                fill_color=cluster_color,
                fill_opacity=0.9
            ).add_to(m)
            
            path_coords.append((lat, lon))

        # Draw line connecting points
        if len(path_coords) >= 2:
            folium.PolyLine(
                path_coords,
                color="blue",
                weight=1.0,
                opacity=0.8,
                tooltip="Path"
            ).add_to(m)

    # Add AntPath for animated movement visualization
    # plugins.AntPath(path_coords, delay=1000, color='blue', weight=2, opacity=0.6).add_to(m)
    
    # Fit map to bounds if there are photos
    if all_lat_lons:
        m.fit_bounds(all_lat_lons)

    folium.LayerControl().add_to(m)
    
    m.save(output_file)
    print(f"Map saved to {output_file}")
    
    # Attempt to open
    try:
        import webbrowser
        webbrowser.open('file://' + os.path.realpath(output_file))
    except:
        pass

async def main():
    parser = argparse.ArgumentParser(description="Visualize photos on Google Map with distances.")
    parser.add_argument("dirs", nargs='+', help="List of directories containing photos")
    parser.add_argument("--output", "-o", default=None, help="Output HTML file")
    parser.add_argument("--sql-dump", default=None, help="Path to SQL dump for ground truth clusters")
    
    args = parser.parse_args()

    # Locate SQL dump if not provided
    sql_dump_path = args.sql_dump
    if not sql_dump_path:
        base_dir = Path(__file__).resolve().parent.parent.parent
        candidate = base_dir / ".experiment/dataset_sqldump/report_db_photos_2025-12-03_200313.sql"
        print(candidate)
        if candidate.exists():
            sql_dump_path = str(candidate)
        else:
            print("Note: SQL dump not found at default location. Clusters will be unknown.")
    
    cluster_mapping = {}
    if sql_dump_path:
        print(f"Loading ground truth from {sql_dump_path}...")
        cluster_mapping = load_ground_truth(sql_dump_path)
        print(f"Loaded {len(cluster_mapping)} file mappings.")

    all_photos = []
    missing_gps_files = [] # Initialize list for files without GPS
    
    # Initialize MetadataExtractor
    extractor = MetadataExtractor()

    output_file_name = "map"
    for d in args.dirs:
        if not os.path.isdir(d):
            print(f"Warning: {d} is not a directory. Skipping.")
            continue
        output_file_name += f"_{d.split('/')[-1]}"
            
        # Search for images
        exts = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(d, ext)))
            
        print(f"Found {len(files)} images in {d}")
        
        for f in files:
            # Use MetadataExtractor
            meta = await extractor.extract(f)
            # Check if metadata extraction was successful and GPS data is present
            if meta and meta.lat is not None and meta.lon is not None:
                # Convert timestamp (float) to datetime for compatibility
                ts = datetime.fromtimestamp(meta.timestamp) if meta.timestamp else None
                
                all_photos.append({
                    'path': f,
                    'name': meta.original_name,
                    'lat': meta.lat,
                    'lon': meta.lon,
                    'timestamp': ts or datetime.min
                })
            else:
                missing_gps_files.append(f) # Collect files without GPS

    # Determine output file path
    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / ".experiment/exp_results/map/"
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    if not args.output:
        output_file = output_dir / f"{output_file_name}.html"
    else:
        output_file = output_dir / args.output

    print(f"Total photos with GPS: {len(all_photos)}")
    
    if missing_gps_files:
        print("\n--- Photos without valid GPS data (will not be on map) ---")
        for f_no_gps in missing_gps_files:
            print(f"- {f_no_gps}")
        print("----------------------------------------------------------\n")

    create_map(all_photos, str(output_file), cluster_mapping)

if __name__ == "__main__":
    asyncio.run(main())