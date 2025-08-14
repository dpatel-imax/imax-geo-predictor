import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim
from thefuzz import process
import math
import os, json, time
from pathlib import Path
from sklearn.neighbors import BallTree
# =========================
# Load data
# =========================
imax_theaters_df = pd.read_csv('list_of_IMAX.csv')
worldcities_df = pd.read_csv('worldcities.csv')

# Normalize common column name variants in worldcities
if 'pop' in worldcities_df.columns and 'population' not in worldcities_df.columns:
    worldcities_df = worldcities_df.rename(columns={'pop': 'population'})
if 'lon' in worldcities_df.columns and 'lng' not in worldcities_df.columns:
    worldcities_df = worldcities_df.rename(columns={'lon': 'lng'})

# =========================
# Build a fast lookup index for (city, country) -> (lat, lon, population)
# =========================
def build_worldcities_index(df):
    df = df.copy()
    pop_col = 'population' if 'population' in df.columns else 'pop'
    lat_col = 'lat' if 'lat' in df.columns else 'latitude'
    lng_col = 'lng' if 'lng' in df.columns else ('longitude' if 'longitude' in df.columns else 'lon')

    # normalize keys once
    df['city_l'] = df['city'].astype(str).str.lower().str.strip()
    df['country_l'] = df['country'].astype(str).str.lower().str.strip()

    # keep the largest-pop row for duplicate (city,country) pairs
    df = df.sort_values(by=pop_col, ascending=False).drop_duplicates(subset=['city_l', 'country_l'])

    # build dict
    index = {
        (row['city_l'], row['country_l']): (row[lat_col], row[lng_col], row[pop_col])
        for _, row in df.iterrows()
    }
    return index

WORLD_INDEX = build_worldcities_index(worldcities_df)

def lookup_city_geo_pop_fast(city, country, world_index=WORLD_INDEX):
    key = (str(city).lower().strip(), str(country).lower().strip())
    return world_index.get(key, (None, None, np.nan))

# =========================
# Overpass (OSM) — with simple on-disk caching
# =========================
OSM_CACHE_PATH = "osm_cache.json"
_osm_cache = {}
if os.path.exists(OSM_CACHE_PATH):
    try:
        with open(OSM_CACHE_PATH, "r", encoding="utf-8") as f:
            _osm_cache = json.load(f)
    except Exception:
        _osm_cache = {}

def _save_osm_cache():
    try:
        with open(OSM_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(_osm_cache, f)
    except Exception:
        pass

def get_cinema_data_osm(lat, lon, radius_km=20):
    """
    Query OpenStreetMap Overpass API for cinemas near a given lat/lon.
    Returns list of dicts with minimal fields.
    (Kept for compatibility; now called via cached wrapper.)
    """
    delta = radius_km / 111.0
    south, north = lat - delta, lat + delta
    west, east  = lon - delta, lon + delta

    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="cinema"]({south},{west},{north},{east});
      way["amenity"="cinema"]({south},{west},{north},{east});
      relation["amenity"="cinema"]({south},{west},{north},{east});
    );
    out;
    """

    try:
        response = requests.post("https://overpass-api.de/api/interpreter", data={'data': query}, timeout=30)
        response.raise_for_status()
        result = response.json()
        cinemas = []
        for elem in result.get("elements", []):
            elem_type = elem.get("type")
            tags = elem.get("tags", {}) or {}
            name = tags.get("name")
            brand = tags.get("brand")
            addr_city = tags.get("addr:city")
            lat_result = elem.get("lat") or (elem.get("center") or {}).get("lat")
            lon_result = elem.get("lon") or (elem.get("center") or {}).get("lon")
            cinemas.append({
                "type": elem_type,
                "name": name,
                "brand": brand,
                "city": addr_city,
                "lat": lat_result,
                "lon": lon_result
            })
        return cinemas
    except Exception:
        return []

def get_cinema_data_osm_cached(lat, lon, radius_km=20, sleep_s=1.05):
    # Round lat/lon to reduce cache key explosion
    key = f"{round(float(lat),4)}|{round(float(lon),4)}|{int(radius_km)}"
    if key in _osm_cache:
        return _osm_cache[key]
    data = get_cinema_data_osm(lat, lon, radius_km)
    _osm_cache[key] = data
    _save_osm_cache()
    # be polite to Overpass (public instance ~1 req/sec)
    time.sleep(sleep_s)
    return data

# =========================
# Geolocator (not used heavily now, but kept available)
# =========================
geolocator = Nominatim(user_agent="imax_site_selector")

# =========================
# Region filter over IMAX list (your function unchanged)
# =========================
def region_input(scope, region_name):
    if scope == 'global':
        return imax_theaters_df
    elif scope == 'country':
        return imax_theaters_df[imax_theaters_df['Country'].str.lower() == region_name.lower()]
    elif scope == 'state':
        return imax_theaters_df[imax_theaters_df['State'].str.lower() == region_name.lower()]
    elif scope == 'city':
        return imax_theaters_df[imax_theaters_df['City'].str.lower() == region_name.lower()]
    else:
        return pd.DataFrame()

# =========================
# Build scoped, mapped dataframe (fast lookup + cached OSM)
# =========================
def cinema_stats_for_scope_params(worldcities_df, scope, region_name="", radius_km=20, use_osm=True, max_rows=None):
    filtered_df = region_input(scope, region_name)
    if filtered_df.empty:
        return pd.DataFrame(columns=[
            "City","State/Province","Country","lat","lon",
            "population","cinema_count_radius_km","cinemas_per_100k"
        ])
    if max_rows:
        filtered_df = filtered_df.head(int(max_rows))

    rows = []
    for _, r in filtered_df.iterrows():
        city    = r.get("City")
        country = r.get("Country")
        state   = r.get("State/Province", None)

        lat, lon, pop = lookup_city_geo_pop_fast(city, country)
        if lat is None or lon is None:
            continue

        if use_osm:
            cinemas = get_cinema_data_osm_cached(lat, lon, radius_km=radius_km)
            count = len(cinemas) if isinstance(cinemas, list) else np.nan
        else:
            count = np.nan  # skip network during dev

        per_100k = np.nan
        if pd.notna(pop) and pop > 0 and pd.notna(count):
            per_100k = 100000.0 * count / pop

        rows.append({
            "City": city,
            "State/Province": state,
            "Country": country,
            "lat": lat,
            "lon": lon,
            "population": pop,
            "cinema_count_radius_km": count,
            "cinemas_per_100k": per_100k
        })
    return pd.DataFrame(rows)

# =========================
# Distance + scoring
# =========================

# === Precompute & BallTree helpers ===
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def _prepped_path(scope: str, region_name: str):
    slug = region_name.lower().strip().replace(" ", "_") if region_name else "global"
    return DATA_DIR / f"cinemas_{scope}_{slug}.npy"

def geocode_bbox(query: str):
    """Return (south, west, north, east) via Nominatim for a region name."""
    loc = geolocator.geocode(query, exactly_one=True, addressdetails=True, timeout=15)
    time.sleep(1.0)  # be polite to Nominatim
    if not loc or "boundingbox" not in loc.raw:
        raise ValueError(f"No bounding box found for '{query}'")
    s, n, w, e = loc.raw["boundingbox"]  # strings
    return float(s), float(w), float(n), float(e)

def fetch_cinemas_in_bbox(south, west, north, east):
    """ONE Overpass call for ALL cinemas within a bbox. Returns [(lat,lon), ...]."""
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
    elements = r.json().get("elements", [])
    pts = []
    for el in elements:
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        pts.append((float(lat), float(lon)))
    return pts

def ensure_region_precomputed(scope: str, region_name: str):
    """
    If data/cinemas_{scope}_{slug}.npy doesn't exist, fetch once and save.
    Return a BallTree over those points (haversine metric), or None if not available.
    """
    p = _prepped_path(scope, region_name)
    if not p.exists():
        # Build a geocode query (state/city need country context if ambiguous — pass 'California, United States')
        query = region_name if scope.lower() != "global" else None
        if query is None:
            return None  # we don't precompute for global
        s, w, n, e = geocode_bbox(query)
        pts = fetch_cinemas_in_bbox(s, w, n, e)
        np.save(p, np.array(pts, dtype=np.float64))
        time.sleep(0.5)  # small pause for politeness

    arr = np.load(p)  # shape (N,2) in degrees
    if arr.size == 0:
        return None
    radians = np.radians(arr)  # BallTree expects radians
    return BallTree(radians, metric="haversine")

def count_cinemas_with_tree(tree: BallTree, lat: float, lon: float, radius_km: float):
    """Fast radius count using the precomputed BallTree."""
    earth_km = 6371.0088
    r = radius_km / earth_km
    q = np.radians([[lat, lon]])
    return int(tree.query_radius(q, r=r, count_only=True)[0])

def seed_osm_cache_for_candidates(candidates_df: pd.DataFrame, tree: BallTree, radius_km: int):
    changed = False
    for _, row in candidates_df.iterrows():
        lat = float(row["lat"]); lon = float(row["lon"])
        key = f"{round(lat,4)}|{round(lon,4)}|{int(radius_km)}"
        if key in _osm_cache:
            continue
        cnt = count_cinemas_with_tree(tree, lat, lon, radius_km)
        # store a stub list with 'cnt' items so len(...) works with your existing code
        _osm_cache[key] = [{}] * int(cnt)
        changed = True
    if changed:
        _save_osm_cache()





def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = math.radians(lat2 - lat1); dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def imax_points_from_csv_fast(imax_df):
    pts = []
    # de-duplicate city/country to reduce lookups
    dedup = imax_df.drop_duplicates(subset=['City','Country'])
    for _, r in dedup.iterrows():
        lat, lon, _ = lookup_city_geo_pop_fast(r["City"], r["Country"])
        if lat is not None and lon is not None:
            pts.append((lat, lon))
    return pts

IMAX_POINTS = imax_points_from_csv_fast(imax_theaters_df)

def nearest_imax_km(lat, lon):
    if not IMAX_POINTS:
        return np.nan
    # compute min distance
    dmin = np.inf
    for ilat, ilon in IMAX_POINTS:
        d = haversine_km(lat, lon, ilat, ilon)
        if d < dmin:
            dmin = d
    return dmin

def add_scoring(df):
    if df.empty:
        return df
    df = df.copy()
    df["nearest_imax_km"] = df.apply(lambda r: nearest_imax_km(r["lat"], r["lon"]), axis=1)

    # Feature transforms
    lp = np.log1p(df["population"].clip(lower=0).fillna(0))
    lc = np.log1p(df["cinema_count_radius_km"].clip(lower=0).fillna(0))
    nd = df["nearest_imax_km"].fillna(0) / 100.0

    # Tunable linear score
    df["score"] = (0.6 * lp) + (0.3 * lc) + (0.1 * nd)

    # Optional: remove cities already very close to an IMAX
    df = df[(df["nearest_imax_km"].isna()) | (df["nearest_imax_km"] >= 20)]
    return df.sort_values("score", ascending=False)

# --- NEW: worldcities-based scope filter for candidate cities (not IMAX list) ---
def filter_candidates_worldcities(worldcities_df, scope, region_name=""):
    df = worldcities_df.copy()

    # normalize keys once
    df["city_l"]    = df["city"].astype(str).str.lower().str.strip()
    df["country_l"] = df["country"].astype(str).str.lower().str.strip()
    if "admin_name" in df.columns:
        df["admin_l"] = df["admin_name"].astype(str).str.lower().str.strip()
    if "continent" in df.columns:
        df["continent_l"] = df["continent"].astype(str).str.lower().str.strip()

    # pick coord/pop cols
    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lng" if "lng" in df.columns else ("lon" if "lon" in df.columns else "longitude")
    pop_col = "population" if "population" in df.columns else "pop"

    # keep valid coords
    df = df.dropna(subset=[lat_col, lon_col])
    df[pop_col] = pd.to_numeric(df[pop_col], errors="coerce")

    rn = str(region_name).strip().lower()

    if scope.lower() == "country" and rn:
        df = df[df["country_l"] == rn]
    elif scope.lower() == "state" and rn:
        if "admin_l" in df.columns:
            df = df[df["admin_l"] == rn]
        else:
            df = df.iloc[0:0]  # no admin_name available
    elif scope.lower() == "city" and rn:
        df = df[df["city_l"] == rn]
    # "global" returns all

    # return minimal columns
    out = df[["city","country",lat_col,lon_col,pop_col] + (["admin_name"] if "admin_name" in df.columns else [])]
    return out.rename(columns={lat_col:"lat", lon_col:"lon", pop_col:"population"})



def top_locations(scope, region_name="", top_n=5, radius_km=20, use_osm=None, max_rows=None):
    """
    Smart defaults:
      - Global: use_osm=False (fast) and max_rows=300
      - Country/State/City: precompute once and use BallTree (instant); also seed osm_cache.json
    """
    scope_l = scope.lower()
    if use_osm is None:
        use_osm = (scope_l != "global")
    if max_rows is None and scope_l == "global":
        max_rows = 300

    # 1) Candidates from worldcities (not IMAX list)
    candidates = filter_candidates_worldcities(worldcities_df, scope, region_name)
    if candidates.empty:
        print("No candidate cities for this scope/region.")
        return candidates
    if max_rows:
        candidates = candidates.head(int(max_rows))

    # 2) Precompute region cinemas once (for non-global scopes) and build BallTree
    tree = None
    if scope_l != "global":
        try:
            tree = ensure_region_precomputed(scope, region_name)
        except Exception as e:
            print(f"[warn] Precompute failed for {scope}:{region_name} ({e}). Falling back to per-city OSM.")
            tree = None

    # 3) Compute features (prefer BallTree; else cached Overpass; else skip)
    rows = []
    for _, row in candidates.iterrows():
        city, country = row["city"], row["country"]
        lat, lon = float(row["lat"]), float(row["lon"])
        pop = row.get("population", np.nan)

        if tree is not None:
            ccount = count_cinemas_with_tree(tree, lat, lon, radius_km)
        elif use_osm:
            cinemas = get_cinema_data_osm_cached(lat, lon, radius_km=radius_km)
            ccount  = len(cinemas) if isinstance(cinemas, list) else np.nan
        else:
            ccount = np.nan

        per_100k = np.nan if not (pd.notna(pop) and pop > 0 and pd.notna(ccount)) \
                   else 100000.0 * ccount / pop

        rows.append({
            "City": city, "Country": country,
            "lat": lat, "lon": lon, "population": pop,
            "cinema_count_radius_km": ccount, "cinemas_per_100k": per_100k
        })

    base = pd.DataFrame(rows)
    if base.empty:
        return base

    # 4) If we had a tree, seed osm_cache.json so later per-city calls are instant too
    if tree is not None:
        seed_osm_cache_for_candidates(base, tree, radius_km)

    # 5) Score + drop near-IMAX (your existing scorer)
    ranked = add_scoring(base)

    # 6) Remove exact IMAX cities (optional dedupe vs existing IMAX list)
    if {"City","Country"}.issubset(imax_theaters_df.columns):
        imax_keys = set(zip(
            imax_theaters_df["City"].astype(str).str.lower().str.strip(),
            imax_theaters_df["Country"].astype(str).str.lower().str.strip()
        ))
        ranked = ranked[~ranked.apply(
            lambda r: (str(r["City"]).lower().strip(), str(r["Country"]).lower().strip()) in imax_keys, axis=1
        )]

    return ranked.head(top_n)





print(top_locations("global", top_n=5, use_osm=False))  # fast, no network
#print(top_locations("country", "United States", top_n=5))             # uses OSM by default
#print(top_locations("state", "California", top_n=5))
#print(top_locations("city", "Toronto", top_n=5))


