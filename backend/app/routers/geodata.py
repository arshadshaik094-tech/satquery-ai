from fastapi import APIRouter, HTTPException, Query

from app.services import external_data

router = APIRouter(prefix="/api/geodata", tags=["geodata"])


@router.get("/imagery-tile-url")
async def imagery_tile_url(lat: float, lon: float, zoom: int = 15):
    try:
        url = await external_data.fetch_esri_tile_url(lat, lon, zoom)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Esri tile lookup failed: {e}")
    return {"tile_url": url, "source": "Esri World Imagery"}


@router.get("/events")
async def events(limit: int = 20, status: str = "open", category: str | None = None):
    try:
        data = await external_data.fetch_eonet_events(limit, status, category)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"NASA EONET request failed: {e}")
    return data


@router.get("/ndvi-tile-url")
def ndvi_tile_url(lat: float, lon: float, date: str = Query(..., description="YYYY-MM-DD"), zoom: int = 6):
    url = external_data.gibs_ndvi_tile_url(lat, lon, date, zoom)
    return {"tile_url": url, "source": "NASA GIBS (MODIS Terra NDVI 8-Day)"}


@router.get("/geocode")
async def geocode(place: str):
    try:
        results = await external_data.geocode_place(place)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Nominatim request failed: {e}")
    return {"results": results}


@router.get("/gdacs-events")
async def gdacs_events(days_back: int = 30):
    """Real GDACS disaster events — floods, earthquakes, cyclones, wildfires."""
    try:
        data = await external_data.fetch_gdacs_events(days_back)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"GDACS request failed: {e}")
    return data


@router.get("/bhuvan-wms-url")
def bhuvan_wms_url():
    """Returns Bhuvan ISRO WMS endpoint info for frontend layer control."""
    return {
        "wms_url": "https://bhuvan-vec2.nrsc.gov.in/bhuvan/wms",
        "layers": {
            "lulc50k": "Land Use / Land Cover 50K",
            "india_outline": "India Administrative Boundary",
        },
        "source": "Bhuvan — ISRO/NRSC Geoportal",
        "attribution": "© ISRO/NRSC — National Remote Sensing Centre",
    }


@router.get("/isro-sources")
def isro_sources():
    """Returns all ISRO data sources used in SatQuery AI."""
    return {
        "sources": [
            {
                "name": "Bhuvan",
                "org": "ISRO/NRSC",
                "description": "India's national geoportal — Land Use/Land Cover thematic layers",
                "url": "https://bhuvan.nrsc.gov.in",
                "type": "WMS",
                "keyless": True,
            },
            {
                "name": "MOSDAC",
                "org": "ISRO/SAC",
                "description": "Meteorological & Oceanographic Satellite Data — rainfall, ocean, atmosphere",
                "url": "https://mosdac.gov.in",
                "type": "Open Data Portal",
                "keyless": True,
            },
            {
                "name": "NRSC",
                "org": "ISRO",
                "description": "National Remote Sensing Centre — satellite data acquisition & processing",
                "url": "https://nrsc.gov.in",
                "type": "Data Archive",
                "keyless": True,
            },
        ]
    }


@router.get("/satellite-tile-url")
def satellite_tile_url(date: str, zoom: int, x: int, y: int):
    """
    NASA GIBS satellite imagery tile URL for a specific date.
    Uses MODIS Terra TrueColor (daily global coverage, no API key).
    Date format: YYYY-MM-DD
    """
    url = external_data.gibs_rgb_tile_url(date, zoom, x, y)
    return {"tile_url": url, "source": "NASA GIBS MODIS Terra", "date": date}


@router.post("/capture-region")
async def capture_region(
    date: str = Query(..., description="YYYY-MM-DD"),
    lat: float = Query(...),
    lon: float = Query(...),
    bbox_size_km: float = Query(5.0, description="Bounding box size in km"),
):
    """
    Captures a satellite image for a specific date and location.
    Returns the image saved to uploads/ and returns the path.
    Uses NASA GIBS MODIS Terra TrueColor.
    """
    try:
        image_path = await external_data.capture_satellite_region(date, lat, lon, bbox_size_km)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(502, f"Failed to capture region: {e}")
    return {"image_path": image_path, "date": date, "location": {"lat": lat, "lon": lon}}
