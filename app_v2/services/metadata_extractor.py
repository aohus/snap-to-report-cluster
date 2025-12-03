import logging
import secrets
import string
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Tuple

import piexif
from app_v2.common.models import PhotoMeta

logger = logging.getLogger(__name__)


def generate_short_id(prefix: str, length: int = 10) -> str:
    chars = string.ascii_letters + string.digits
    random_str = ''.join(secrets.choice(chars) for _ in range(length))
    return f"{prefix}_{random_str}"


class MetadataExtractor:
    async def extract(self, image_path: str) -> PhotoMeta:
        """Asynchronously extracts metadata from an image file."""
        try:
            exif = self._export_exif_sync(image_path)
        except Exception as e:
            logger.error(f"Failed to run exif extraction for {image_path}: {e}")
            exif = None

        lat, lon, alt, direction = self._get_gps_from_exif(exif)
        timestamp = self._parse_datetime_from_exif(exif)

        focal_35mm = None
        orientation = None
        digital_zoom = None
        scene_capture_type = None
        white_balance = None
        exposure_mode = None
        flash = None

        if exif is not None:
            exif_0th = exif.get("0th", {})
            exif_exif = exif.get("Exif", {})

            orientation = exif_0th.get(piexif.ImageIFD.Orientation)
            focal_35mm = exif_exif.get(piexif.ExifIFD.FocalLengthIn35mmFilm)
            if isinstance(focal_35mm, (tuple, list)):
                focal_35mm = self._rational_to_float(focal_35mm)

            dz = exif_exif.get(piexif.ExifIFD.DigitalZoomRatio)
            digital_zoom = self._rational_to_float(dz)

            scene_capture_type = exif_exif.get(piexif.ExifIFD.SceneCaptureType)
            white_balance = exif_exif.get(piexif.ExifIFD.WhiteBalance)
            exposure_mode = exif_exif.get(piexif.ExifIFD.ExposureMode)
            flash = exif_exif.get(piexif.ExifIFD.Flash)

        return PhotoMeta(
            id=generate_short_id('meta'),
            path=image_path,
            original_name=image_path.split("/")[-1],
            lat=lat,
            lon=lon,
            alt=alt,
            timestamp=timestamp,
            focal_35mm=float(focal_35mm) if focal_35mm is not None else None,
            orientation=orientation,
            digital_zoom=digital_zoom,
            scene_capture_type=scene_capture_type,
            white_balance=white_balance,
            exposure_mode=exposure_mode,
            flash=flash,
            gps_img_direction=direction,
        )

    def _export_exif_sync(self, img_path: str) -> Optional[dict]:
        """Synchronous helper for loading EXIF data."""
        try:
            return piexif.load(img_path)
        except Exception as e:
            logger.warning(f"Failed to load EXIF from {img_path}: {e}")
            return None

    def _rational_to_float(self, value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            if isinstance(value, tuple) and len(value) == 2:
                num, den = value
                if den == 0: return None
                return float(num) / float(den)
            if isinstance(value, (list, tuple)) and len(value) > 0:
                v0 = value[0]
                if isinstance(v0, tuple) and len(v0) == 2:
                    num, den = v0
                    if den == 0: return None
                    return float(num) / float(den)
        except (ValueError, TypeError, ZeroDivisionError):
            return None
        return None

    def _parse_datetime_from_exif(self, exif: Optional[dict]) -> Optional[float]:
        if exif is None: return None
        exif_exif = exif.get("Exif", {})
        dt_bytes = exif_exif.get(piexif.ExifIFD.DateTimeOriginal)
        if not dt_bytes: return None

        try:
            dt_str = dt_bytes.decode()
            base_dt = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
        except (UnicodeDecodeError, ValueError):
            return None

        # Microseconds
        subsec_bytes = exif_exif.get(piexif.ExifIFD.SubSecTimeOriginal)
        microseconds = 0
        if subsec_bytes:
            try:
                subsec_str = subsec_bytes.decode().ljust(6, '0')
                microseconds = int(subsec_str[:6])
            except (UnicodeDecodeError, ValueError):
                microseconds = 0
        
        # Timezone
        offset_bytes = exif_exif.get(piexif.ExifIFD.OffsetTimeOriginal)
        tz = timezone.utc
        if offset_bytes:
            try:
                offset_str = offset_bytes.decode()
                sign = -1 if offset_str[0] == "-" else 1
                h = int(offset_str[1:3])
                m = int(offset_str[4:6])
                tz = timezone(sign * timedelta(hours=h, minutes=m))
            except (UnicodeDecodeError, ValueError):
                pass # Fallback to UTC

        dt = base_dt.replace(microsecond=microseconds, tzinfo=tz)
        return dt.timestamp()

    def _get_gps_from_exif(self, exif: Optional[dict]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        if exif is None or "GPS" not in exif:
            return None, None, None, None

        gps = exif["GPS"]
        def convert_coord(coord, ref):
            try:
                degrees, minutes, seconds = [x[0] / x[1] for x in coord]
                result = degrees + (minutes / 60.0) + (seconds / 3600.0)
                return result if ref in [b"N", b"E"] else -result
            except (TypeError, ValueError, ZeroDivisionError, IndexError):
                return None

        lat = convert_coord(gps.get(piexif.GPSIFD.GPSLatitude), gps.get(piexif.GPSIFD.GPSLatitudeRef))
        lon = convert_coord(gps.get(piexif.GPSIFD.GPSLongitude), gps.get(piexif.GPSIFD.GPSLongitudeRef))
        
        alt = None
        if piexif.GPSIFD.GPSAltitude in gps:
            alt_val = gps[piexif.GPSIFD.GPSAltitude]
            if alt_val[1] != 0:
                alt = alt_val[0] / alt_val[1]

        direction = self._rational_to_float(gps.get(piexif.GPSIFD.GPSImgDirection))

        return lat, lon, alt, direction