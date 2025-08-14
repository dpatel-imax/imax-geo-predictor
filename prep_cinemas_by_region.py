# prep_cinemas_by_region.py
import json, time, os, math
from pathlib import Path
import requests
import numpy as np
from geopy.geocoders import Nominatim

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
geolocator = Nominatim(user_agent="imax_site_selector_prep")

def geocode_bbox(query: str):
    """Return (south, west, north, east) for a place via Nominatim."""
    loc = geolocator.geocode(query, exactly_one=True, addressdetails=True)
    time.sleep(1.0)  # be polite
    if not loc or "boundingbox" not in loc.raw:
        raise ValueError(f"No bbox for '{query}'")
    # Nominatim order is [south, north, west, east] as strings
    s, n, w, e = loc.raw["boundingbox"]
    return float(s), float(w), float(n), float(e)

def fetch_cinemas_in_bbox(south, west, north, east):
    """One Overpass call to fetch ALL cinemas in the bbox."""
    query = f"""
    [out:json][timeout:120];
    (
      node["amenity"="cinema"]({south},{west},{north},{east});
      way["amenity"="cinema"]({south},{west},{north},{east});
      relation["amenity"="cinema"]({south},{west},{north},{east});
    );
    out center;
    """
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=180)
    r.raise_for_status()
    data = r.json().get("elements", [])
    pts = []
    for el in data:
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None: 
            continue
        pts.append((float(lat), float(lon)))
    return pts

def save_region(scope: str, region_name: str, pts: list[tuple[float,float]]):
    slug = region_name.lower().strip().replace(" ", "_") if region_name else "global"
    base = DATA_DIR / f"cinemas_{scope}_{slug}"
    arr = np.array(pts, dtype=np.float64)
    np.save(base.with_suffix(".npy"), arr)
    with open(base.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump({"scope": scope, "region": region_name, "count": len(pts)}, f, ensure_ascii=False)
    print(f"Saved {len(pts)} cinemas to {base.with_suffix('.npy')}")

def main():
    # EXAMPLES — run what you need, one at a time:
    # scope, region = "country", "Japan"
    # scope, region = "state", "California, United States"
    # scope, region = "city", "Toronto, Canada"
    scope, region = "country", "Japan"  # <-- change me

    # Build geocode query for bbox
    query = region if scope != "global" else "World"
    s, w, n, e = geocode_bbox(query)
    pts = fetch_cinemas_in_bbox(s, w, n, e)
    save_region(scope, region, pts)

if __name__ == "__main__":
    main()
