from dataclasses import dataclass
from typing import Optional


@dataclass
class PhotoMeta:
    id: str
    path: str
    original_name: str
    device: str
    exposure_time: Optional[float] = None
    iso_speed_rating: Optional[int] = None
    focal_length: Optional[float] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    timestamp: Optional[float] = None  # Unix timestamp
    orientation: Optional[int] = None
    digital_zoom: Optional[float] = None
    scene_capture_type: Optional[int] = None
    white_balance: Optional[int] = None
    flash: Optional[int] = None
    gps_img_direction: Optional[float] = None  # 방위각(도 단위)
