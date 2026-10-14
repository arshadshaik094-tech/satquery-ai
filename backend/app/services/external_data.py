"""
Live public data-source integrations — all real, keyless HTTP APIs.
No API key required for any of these (Esri World Imagery, NASA EONET v3,
NASA GIBS, Nominatim/OSM). Network access to these hosts is required at
runtime on the machine serving requests.
"""
from __future__ import annotations

import math
import httpx
from app.config import get_settings

TIMEOUT = 15.0


def latlon_to_tile(lat: float, lon: float, zoom: int) -> tuple[int, int]:
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


async def fetch_esri_tile_url(lat: float, lon: float, zoom: int = 15) -> str:
    """Returns a real, directly fetchable Esri World Imagery XYZ tile URL
    for the given coordinates (no API key required)."""
    settings = get_settings()
    x, y = latlon_to_tile(lat, lon, zoom)
    return settings.ESRI_WORLD_IMAGERY_TILE_TEMPLATE.format(z=zoom, x=x, y=y)


async def fetch_esri_tile_bytes(lat: float, lon: float, zoom: int = 15) -> bytes:
    url = await fetch_esri_tile_url(lat, lon, zoom)
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content


async def fetch_eonet_events(limit: int = 20, status: str = "open", category: str | None = None):
    """Real NASA EONET v3 natural-event feed (fires, floods, storms, etc)."""
    settings = get_settings()
    params = {"limit": limit, "status": status}
    if category:
        params["category"] = category
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(settings.NASA_EONET_URL, params=params)
        r.raise_for_status()
        return r.json()


def gibs_ndvi_tile_url(lat: float, lon: float, date: str, zoom: int = 6) -> str:
    """Real NASA GIBS WMTS tile URL for MODIS NDVI on a given date
    (YYYY-MM-DD). GIBS uses a different tiling scheme (EPSG:4326); we
    return the templated URL for the caller to resolve exact row/col via
    the standard WMTS GetCapabilities if pixel-exact tiles are needed."""
    settings = get_settings()
    x, y = latlon_to_tile(lat, lon, zoom)
    return settings.NASA_GIBS_WMTS_TEMPLATE.format(
        layer="MODIS_Terra_NDVI_8Day",
        date=date,
        tile_matrix_set="2km",
        z=zoom, y=y, x=x,
    )


async def geocode_place(place_name: str) -> list[dict]:
    """
    Geocoding with two-stage fallback:
    1. Photon (OpenStreetMap-based, finds specific buildings/colleges/local places)
    2. Nominatim fallback if Photon fails
    Returns normalized list with lat, lon, display_name keys.
    """
    headers = {"User-Agent": "SatQueryAI/1.0"}

    # Stage 1 — Photon (better for specific local places, buildings, colleges)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as client:
            r = await client.get(
                "https://photon.komoot.io/api/",
                params={"q": place_name, "limit": 5},
            )
            r.raise_for_status()
            data = r.json()
            features = data.get("features", [])
            if features:
                results = []
                for f in features:
                    props = f.get("properties", {})
                    coords = f.get("geometry", {}).get("coordinates", [])
                    if len(coords) >= 2:
                        # Build a readable display name
                        parts = [
                            props.get("name", ""),
                            props.get("street", ""),
                            props.get("city", ""),
                            props.get("state", ""),
                            props.get("country", ""),
                        ]
                        display = ", ".join(p for p in parts if p)
                        results.append({
                            "lat": str(coords[1]),
                            "lon": str(coords[0]),
                            "display_name": display or place_name,
                        })
                if results:
                    return results
    except Exception:
        pass  # Fall through to Nominatim

    # Stage 2 — Nominatim fallback
    settings = get_settings()
    params = {"q": place_name, "format": "json", "limit": 5}
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=headers) as client:
        r = await client.get(settings.NOMINATIM_URL, params=params)
        r.raise_for_status()
        return r.json()


async def fetch_gdacs_events(days_back: int = 30) -> dict:
    """
    Real GDACS (UN/EC Global Disaster Alert and Coordination System) events.
    Covers floods, earthquakes, tropical cyclones, volcanoes, wildfires, droughts.
    No API key required. Returns GeoJSON FeatureCollection.
    """
    from datetime import datetime, timedelta
    today = datetime.utcnow()
    from_date = (today - timedelta(days=days_back)).strftime("%Y-%m-%d")
    to_date = today.strftime("%Y-%m-%d")
    url = "https://www.gdacs.org/gdacsapi/api/events/geteventlist/SEARCH"
    params = {
        "eventtypes": "FL,EQ,TC,VO,WF",
        "alertlevels": "Green,Orange,Red",
        "fromdate": from_date,
        "todate": to_date,
        "limit": 50,
    }
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        r = await client.get(url, params=params)
        r.raise_for_status()
        return r.json()


def gibs_rgb_tile_url(date: str, zoom: int, x: int, y: int) -> str:
    """
    NASA GIBS RGB imagery tile URL for a specific date.
    Uses MODIS Terra Corrected Reflectance (TrueColor) - daily global coverage.
    Date format: YYYY-MM-DD
    """
    layer = "MODIS_Terra_CorrectedReflectance_TrueColor"
    tile_matrix_set = "GoogleMapsCompatible_Level9"
    base = "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best"
    return f"{base}/{layer}/default/{date}/{tile_matrix_set}/{zoom}/{y}/{x}.jpg"


async def capture_satellite_region(date: str, lat: float, lon: float, bbox_size_km: float = 10.0) -> str:
    """
    Captures a satellite image region using EOX Sentinel-2 cloudless WMS.
    - Free, no API key, 10m resolution, covers 2018-2023 annually
    - 'date' param accepts a year string like "2018", "2019" ... "2023"
      or a full date "YYYY-MM-DD" (year is extracted)
    Saves to uploads/ and returns the relative path.
    """
    import os
    from pathlib import Path
    from PIL import Image
    import io

    os.makedirs("uploads", exist_ok=True)

    # Extract year from date string
    year = str(date)[:4]
    # Clamp to available EOX years
    available = [2018, 2019, 2020, 2021, 2022, 2023]
    year_int = int(year)
    if year_int < 2018:
        year_int = 2018
    elif year_int > 2023:
        year_int = 2023
    year_str = str(year_int)

    # Calculate bounding box from center lat/lon
    # 1 degree lat ≈ 111km, 1 degree lon ≈ 111*cos(lat) km
    import math
    lat_offset = (bbox_size_km / 2) / 111.0
    lon_offset = (bbox_size_km / 2) / (111.0 * math.cos(math.radians(lat)))

    min_lat = lat - lat_offset
    max_lat = lat + lat_offset
    min_lon = lon - lon_offset
    max_lon = lon + lon_offset

    bbox = f"{min_lon},{min_lat},{max_lon},{max_lat}"
    layer = f"s2cloudless-{year_str}"

    url = (
        f"https://tiles.maps.eox.at/wms"
        f"?service=WMS&request=GetMap"
        f"&layers={layer}"
        f"&styles=&format=image/jpeg"
        f"&transparent=false&version=1.1.1"
        f"&width=768&height=768"
        f"&srs=EPSG:4326"
        f"&bbox={bbox}"
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        img = Image.open(io.BytesIO(r.content)).convert("RGB")

    filename = f"satellite_{year_str}_{lat:.4f}_{lon:.4f}.jpg"
    filepath = Path("uploads") / filename
    img.save(filepath, quality=92)
    return str(filepath)


async def _stitch_esri_tiles(lat: float, lon: float, zoom: int) -> "Image.Image":
    """Download and stitch 3x3 Esri World Imagery tiles around the center."""
    from PIL import Image
    import io

    cx, cy = latlon_to_tile(lat, lon, zoom)
    rows = []
    for dy in range(-1, 2):
        row = []
        for dx in range(-1, 2):
            x, y = cx + dx, cy + dy
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{zoom}/{y}/{x}"
            async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                r = await client.get(url)
                r.raise_for_status()
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
                row.append(img)
        rows.append(row)
    return _stitch_grid(rows)


async def _stitch_gibs_tiles(date: str, lat: float, lon: float, zoom: int = 9) -> "Image.Image":
    """Download and stitch NASA GIBS MODIS TrueColor tiles."""
    from PIL import Image
    import io

    cx, cy = latlon_to_tile(lat, lon, zoom)
    rows = []
    for dy in range(-1, 2):
        row = []
        for dx in range(-1, 2):
            x, y = cx + dx, cy + dy
            url = gibs_rgb_tile_url(date, zoom, x, y)
            try:
                async with httpx.AsyncClient(timeout=TIMEOUT) as client:
                    r = await client.get(url)
                    r.raise_for_status()
                    img = Image.open(io.BytesIO(r.content)).convert("RGB")
                    row.append(img)
            except Exception:
                row.append(Image.new("RGB", (256, 256), (30, 30, 30)))
        rows.append(row)
    return _stitch_grid(rows)


def _stitch_grid(rows: list) -> "Image.Image":
    """Stitch a 2D grid of PIL Images into one image."""
    from PIL import Image
    tw = rows[0][0].width
    th = rows[0][0].height
    out = Image.new("RGB", (tw * len(rows[0]), th * len(rows)))
    for iy, row in enumerate(rows):
        for ix, img in enumerate(row):
            out.paste(img, (ix * tw, iy * th))
    return out
