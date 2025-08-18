# main.py
import os, json, time, math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from geopy.geocoders import Nominatim
from sklearn.neighbors import BallTree

# =========================
# Load data
# =========================
imax_theaters_df = pd.read_csv('list_of_IMAX.csv')
worldcities_df   = pd.read_csv('worldcities.csv')

# Normalize common column name variants in worldcities
if 'pop' in worldcities_df.columns and 'population' not in worldcities_df.columns:
    worldcities_df = worldcities_df.rename(columns={'pop': 'population'})
if 'lon' in worldcities_df.columns and 'lng' not in worldcities_df.columns:
    worldcities_df = worldcities_df.rename(columns={'lon': 'lng'})

# =========================
# Fast (city,country) -> (lat, lon, population) index
# =========================
def build_worldcities_index(df: pd.DataFrame):
    df = df.copy()
    pop_col = 'population' if 'population' in df.columns else 'pop'
    lat_col = 'lat' if 'lat' in df.columns else 'latitude'
    lng_col = 'lng' if 'lng' in df.columns else ('longitude' if 'longitude' in df.columns else 'lon')

    df['city_l']    = df['city'].astype(str).str.lower().str.strip()
    df['country_l'] = df['country'].astype(str).str.lower().str.strip()

    df = df.sort_values(by=pop_col, ascending=False).drop_duplicates(subset=['city_l', 'country_l'])

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
# Overpass (OSM) — with simple JSON cache
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
    """
    delta = radius_km / 111.0
    south, north = lat - delta, lat + delta
    west,  east  = lon - delta, lon + delta

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
            tags = elem.get("tags", {}) or {}
            lat_result = elem.get("lat") or (elem.get("center") or {}).get("lat")
            lon_result = elem.get("lon") or (elem.get("center") or {}).get("lon")
            if lat_result is None or lon_result is None:
                continue
            cinemas.append({
                "type": elem.get("type"),
                "name": tags.get("name"),
                "brand": tags.get("brand"),
                "city": tags.get("addr:city"),
                "lat": lat_result,
                "lon": lon_result
            })
        return cinemas
    except Exception:
        return []

def get_cinema_data_osm_cached(lat, lon, radius_km=20, sleep_s=1.05):
    key = f"{round(float(lat),4)}|{round(float(lon),4)}|{int(radius_km)}"
    if key in _osm_cache:
        return _osm_cache[key]
    data = get_cinema_data_osm(lat, lon, radius_km)
    _osm_cache[key] = data
    _save_osm_cache()
    time.sleep(sleep_s)  # respect Overpass rate limits
    return data

# =========================
# Geocoder (for bbox precompute)
# =========================
geolocator = Nominatim(user_agent="imax_site_selector")

# =========================
# Precompute & BallTree helpers (one Overpass call per region)
# =========================
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

def _prepped_path(scope: str, region_name: str):
    slug = region_name.lower().strip().replace(" ", "_") if region_name else "global"
    return DATA_DIR / f"cinemas_{scope}_{slug}.npy"

def geocode_bbox(query: str):
    """Return (south, west, north, east) via Nominatim for a region name."""
    loc = geolocator.geocode(query, exactly_one=True, addressdetails=True, timeout=15)
    time.sleep(1.0)  # be polite
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
    if scope.lower() == "global":
        return None
    p = _prepped_path(scope, region_name)
    if not p.exists():
        query = region_name  # e.g., "California, United States", "Toronto, Canada"
        s, w, n, e = geocode_bbox(query)
        pts = fetch_cinemas_in_bbox(s, w, n, e)
        np.save(p, np.array(pts, dtype=np.float64))
        time.sleep(0.5)

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
    """
    Seed osm_cache.json so later calls to get_cinema_data_osm_cached() are instant.
    We store a stub list with 'count' items so len(list) == count keeps your code unchanged.
    """
    changed = False
    for _, row in candidates_df.iterrows():
        lat = float(row["lat"]); lon = float(row["lon"])
        key = f"{round(lat,4)}|{round(lon,4)}|{int(radius_km)}"
        if key in _osm_cache:
            continue
        cnt = count_cinemas_with_tree(tree, lat, lon, radius_km)
        _osm_cache[key] = [{}] * int(cnt)
        changed = True
    if changed:
        _save_osm_cache()

# =========================
# Candidate pool from worldcities (NOT the IMAX list)
# =========================
def filter_candidates_worldcities(worldcities_df, scope, region_name=""):
    df = worldcities_df.copy()

    df["city_l"]    = df["city"].astype(str).str.lower().str.strip()
    df["country_l"] = df["country"].astype(str).str.lower().str.strip()
    if "admin_name" in df.columns:
        df["admin_l"] = df["admin_name"].astype(str).str.lower().str.strip()
    if "continent" in df.columns:
        df["continent_l"] = df["continent"].astype(str).str.lower().str.strip()

    lat_col = "lat" if "lat" in df.columns else "latitude"
    lon_col = "lng" if "lng" in df.columns else ("lon" if "lon" in df.columns else "longitude")
    pop_col = "population" if "population" in df.columns else "pop"

    df = df.dropna(subset=[lat_col, lon_col])
    df[pop_col] = pd.to_numeric(df[pop_col], errors="coerce")

    rn = str(region_name).strip().lower()
    s = scope.lower()

    if s == "country" and rn:
        df = df[df["country_l"] == rn]
    elif s == "state" and rn:
        if "admin_l" in df.columns:
            df = df[df["admin_l"] == rn]
        else:
            df = df.iloc[0:0]
    elif s == "city" and rn:
        df = df[df["city_l"] == rn]
    # "global" returns all

    out = df[["city", "country", lat_col, lon_col, pop_col] + (["admin_name"] if "admin_name" in df.columns else [])]
    return out.rename(columns={lat_col: "lat", lon_col: "lon", pop_col: "population"})

# =========================
# Distances and heuristics
# =========================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def imax_points_from_csv_fast(imax_df):
    pts = []
    dedup = imax_df.drop_duplicates(subset=['City', 'Country'])
    for _, r in dedup.iterrows():
        lat, lon, _ = lookup_city_geo_pop_fast(r["City"], r["Country"])
        if lat is not None and lon is not None:
            pts.append((lat, lon))
    return pts

IMAX_POINTS = imax_points_from_csv_fast(imax_theaters_df)

def nearest_imax_km(lat, lon):
    if not IMAX_POINTS:
        return np.nan
    dmin = np.inf
    for ilat, ilon in IMAX_POINTS:
        d = haversine_km(lat, lon, ilat, ilon)
        if d < dmin:
            dmin = d
    return dmin

def add_scoring(df: pd.DataFrame):
    if df.empty:
        return df
    df = df.copy()
    df["nearest_imax_km"] = df.apply(lambda r: nearest_imax_km(r["lat"], r["lon"]), axis=1)

    lp = np.log1p(df["population"].clip(lower=0).fillna(0))
    lc = np.log1p(df["cinema_count_radius_km"].clip(lower=0).fillna(0))
    nd = df["nearest_imax_km"].fillna(0) / 100.0

    df["score"] = (0.6 * lp) + (0.3 * lc) + (0.1 * nd)

    # Exclude cities very close to an existing IMAX (tunable)
    df = df[(df["nearest_imax_km"].isna()) | (df["nearest_imax_km"] >= 20)]
    return df.sort_values("score", ascending=False)

# =========================
# Heuristic ranker (fast) — uses precompute when available
# =========================
def top_locations(scope, region_name="", top_n=5, radius_km=20, use_osm=None, max_rows=None):
    """
    Heuristic version.
      - Global: use_osm=False (fast) and max_rows=300 (by default)
      - Country/State/City: precompute once and use BallTree (instant); also seed osm_cache.json
    """
    scope_l = scope.lower()
    if use_osm is None:
        use_osm = (scope_l != "global")
    if max_rows is None and scope_l == "global":
        max_rows = 300

    candidates = filter_candidates_worldcities(worldcities_df, scope, region_name)
    if candidates.empty:
        print("No candidate cities for this scope/region.")
        return candidates
    if max_rows:
        candidates = candidates.head(int(max_rows))

    tree = None
    if scope_l != "global":
        try:
            tree = ensure_region_precomputed(scope, region_name)
        except Exception as e:
            print(f"[warn] Precompute failed for {scope}:{region_name} ({e}). Falling back to per-city OSM).")
            tree = None

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

        per_100k = np.nan if not (pd.notna(pop) and pop > 0 and pd.notna(ccount)) else 100000.0 * ccount / pop

        rows.append({
            "City": city, "Country": country, "lat": lat, "lon": lon, "population": pop,
            "cinema_count_radius_km": ccount, "cinemas_per_100k": per_100k
        })

    base = pd.DataFrame(rows)
    if base.empty:
        return base

    if tree is not None:
        seed_osm_cache_for_candidates(base, tree, radius_km)

    ranked = add_scoring(base)

    # Remove exact IMAX cities (optional dedupe)
    if {"City", "Country"}.issubset(imax_theaters_df.columns):
        imax_keys = set(zip(
            imax_theaters_df["City"].astype(str).str.lower().str.strip(),
            imax_theaters_df["Country"].astype(str).str.lower().str.strip()
        ))
        ranked = ranked[~ranked.apply(
            lambda r: (str(r["City"]).lower().strip(), str(r["Country"]).lower().strip()) in imax_keys, axis=1
        )]

    return ranked.head(top_n)

# =========================
# ML ranker (scikit-learn) — learns a propensity from data
# =========================
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

FEATURES = ["log_population", "cinema_count_20km", "log_cinema_count", "cinemas_per_100k"]

def feature_row(lat, lon, population, radius_km=20, tree=None, use_osm=False):
    if tree is not None:
        ccount = count_cinemas_with_tree(tree, lat, lon, radius_km)
    elif use_osm:
        cinemas = get_cinema_data_osm_cached(lat, lon, radius_km=radius_km)
        ccount = len(cinemas) if isinstance(cinemas, list) else np.nan
    else:
        ccount = np.nan

    return {
        "population": population,
        "log_population": np.log1p(population) if pd.notna(population) and population > 0 else 0.0,
        "cinema_count_20km": ccount,
        "log_cinema_count": np.log1p(ccount) if pd.notna(ccount) and ccount >= 0 else 0.0,
        "cinemas_per_100k": (100000.0 * ccount / population) if pd.notna(ccount) and pd.notna(population) and population > 0 else 0.0
    }

def build_training_table(worldcities_df, imax_df, sample_neg_per_country=150, radius_km=20, tree=None, use_osm=False):
    # Positives = IMAX cities
    pos = []
    seen = set()
    for _, r in imax_df.drop_duplicates(subset=["City", "Country"]).iterrows():
        city, country = r["City"], r["Country"]
        key = (str(city).lower().strip(), str(country).lower().strip())
        if key in seen: 
            continue
        seen.add(key)
        lat, lon, pop = lookup_city_geo_pop_fast(city, country)
        if lat is None or lon is None or pd.isna(pop) or pop <= 0: 
            continue
        f = feature_row(lat, lon, pop, radius_km=radius_km, tree=tree, use_osm=use_osm)
        f.update({"city": city, "country": country, "lat": lat, "lon": lon, "label": 1})
        pos.append(f)

    # Negatives = non-IMAX cities (sampled by country)
    wc = worldcities_df.copy()
    wc["country_l"] = wc["country"].astype(str).str.lower().str.strip()
    imax_keys = set(zip(imax_df["City"].str.lower().str.strip(), imax_df["Country"].str.lower().str.strip()))

    neg = []
    for country, g in wc.groupby("country_l"):
        g2 = g.sort_values(by=("population" if "population" in g.columns else "pop"), ascending=False)
        picked = 0
        for _, row in g2.iterrows():
            key = (str(row["city"]).lower().strip(), str(row["country"]).lower().strip())
            if key in imax_keys:
                continue
            lat = float(row["lat"] if "lat" in row else row["latitude"])
            lon = float(row["lng"] if "lng" in row else (row.get("lon") if "lon" in row else row["longitude"]))
            pop = row.get("population", row.get("pop", np.nan))
            if pd.isna(pop) or pop <= 0:
                continue
            f = feature_row(lat, lon, pop, radius_km=radius_km, tree=tree, use_osm=use_osm)
            f.update({"city": row["city"], "country": row["country"], "lat": lat, "lon": lon, "label": 0})
            neg.append(f)
            picked += 1
            if picked >= sample_neg_per_country:
                break

    df = pd.DataFrame(pos + neg)
    return df

def train_model(train_df: pd.DataFrame, algo="rf"):
    X = train_df[FEATURES].fillna(0.0).to_numpy()
    y = train_df["label"].astype(int).to_numpy()
    groups = train_df["country"].astype(str)

    if algo == "lr":
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=300, class_weight="balanced"))
        ])
    else:
        model = RandomForestClassifier(
            n_estimators=400, max_depth=None, n_jobs=-1,
            class_weight="balanced_subsample", random_state=42
        )

    gkf = GroupKFold(n_splits=5)
    aucs, aps = [], []
    for tr, te in gkf.split(X, y, groups):
        model.fit(X[tr], y[tr])
        ps = model.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], ps))
        aps.append(average_precision_score(y[te], ps))
    print(f"CV AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f} | AP: {np.mean(aps):.3f} ± {np.std(aps):.3f}")

    model.fit(X, y)  # final fit on all data
    return model

def score_candidates(model, candidates_df, radius_km=20, tree=None, use_osm=False):
    rows = []
    for _, row in candidates_df.iterrows():
        city, country = row["city"], row["country"]
        lat, lon = float(row["lat"]), float(row["lon"])
        pop = row.get("population", np.nan)

        f = feature_row(lat, lon, pop, radius_km=radius_km, tree=tree, use_osm=use_osm)
        X = pd.DataFrame([f])[FEATURES].fillna(0.0).to_numpy()  # shape (1, n_features)

        # Predict probability; take the single scalar without casting an array
        p = model.predict_proba(X)[0, 1]          # <- no float(...), no deprecation warning

        rows.append({
            "City": city, "Country": country, "lat": lat, "lon": lon,
            "population": pop, "ml_prob": float(p)  # optional cast here is fine; p is a Python float already
        })
    return pd.DataFrame(rows).sort_values("ml_prob", ascending=False)


def filter_near_imax(df, min_km=20):
    df = df.copy()
    df["nearest_imax_km"] = df.apply(lambda r: nearest_imax_km(r["lat"], r["lon"]), axis=1)
    return df[df["nearest_imax_km"] >= min_km]

def top_locations_ml(scope, region_name="", top_n=5, radius_km=20,
                     use_osm_for_train=False, use_osm_for_score=False, algo="rf",
                     sample_neg_per_country=150):
    """
    ML version (scikit-learn).
      - Builds a labeled dataset (IMAX vs non-IMAX), trains a model, scores candidates.
      - Uses BallTree for fast cinema counts when available (non-global scopes).
    """
    tree = None
    if scope.lower() != "global":
        try:
            tree = ensure_region_precomputed(scope, region_name)
        except Exception as e:
            print(f"[warn] Precompute failed for {scope}:{region_name} ({e}). Proceeding without BallTree.")

    train_df = build_training_table(worldcities_df, imax_theaters_df,
                                    sample_neg_per_country=sample_neg_per_country,
                                    radius_km=radius_km, tree=tree, use_osm=use_osm_for_train)
    model = train_model(train_df, algo=algo)

    candidates = filter_candidates_worldcities(worldcities_df, scope, region_name)
    if candidates.empty:
        print("No candidate cities for this scope/region.")
        return candidates

    scored = score_candidates(model, candidates, radius_km=radius_km, tree=tree, use_osm=use_osm_for_score)
    scored = filter_near_imax(scored, min_km=20)

    imax_keys = set(zip(imax_theaters_df["City"].str.lower().str.strip(),
                        imax_theaters_df["Country"].str.lower().str.strip()))
    scored = scored[~scored.apply(
        lambda r: (str(r["City"]).lower().strip(), str(r["Country"]).lower().strip()) in imax_keys, axis=1
    )]

    return scored.head(top_n)


if __name__ == "__main__":
    # Heuristic (fast dev path)
    #print("Heuristic / Global:")
    #print(top_locations("global", top_n=5, use_osm=False))   # fast, no network

    # Country/State/City: first run does one Overpass bbox fetch; subsequent runs are instant (BallTree)
    #print("Heuristic / Country=Japan:")
    #print(top_locations("country", "Japan", top_n=5))

    # print("Heuristic / State=California, United States:")
    # print(top_locations("state", "California, United States", top_n=5))

    # print("Heuristic / City=Toronto, Canada:")
    # print(top_locations("city", "Toronto, Canada", top_n=5))

    # ML (train + score). Start with OSM disabled for speed; enable on small scopes when ready.
    print("ML / Country=Japan (RF):")
    print(top_locations_ml("country", "Canada", top_n=5, use_osm_for_train=False, use_osm_for_score=False, algo="rf"))
