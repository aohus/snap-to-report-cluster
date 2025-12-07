from dataclasses import dataclass
from typing import Optional


@dataclass
class PhotoMeta:
    id: str
    path: str
    original_name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    alt: Optional[float] = None
    timestamp: Optional[float] = None  # Unix timestamp
    focal_35mm: Optional[float] = None
    orientation: Optional[int] = None
    digital_zoom: Optional[float] = None
    scene_capture_type: Optional[int] = None
    white_balance: Optional[int] = None
    exposure_mode: Optional[int] = None
    flash: Optional[int] = None
    gps_img_direction: Optional[float] = None  # 방위각(도 단위)
