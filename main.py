import os, json, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from geopy.geocoders import Nominatim
from sklearn.neighbors import BallTree
from thefuzz import fuzz
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score



# Load IMAX theaters and world cities data
imax_theaters_df = pd.read_csv('list_of_IMAX.csv')
worldcities_df = pd.read_csv('worldcities.csv')

US_STATE_FULL_SET = {
    "Alabama","Alaska","Arizona","Arkansas","California","Colorado","Connecticut","Delaware",
    "District of Columbia","Florida","Georgia","Hawaii","Idaho","Illinois","Indiana","Iowa","Kansas",
    "Kentucky","Louisiana","Maine","Maryland","Massachusetts","Michigan","Minnesota","Mississippi",
    "Missouri","Montana","Nebraska","Nevada","New Hampshire","New Jersey","New Mexico","New York",
    "North Carolina","North Dakota","Ohio","Oklahoma","Oregon","Pennsylvania","Rhode Island",
    "South Carolina","South Dakota","Tennessee","Texas","Utah","Vermont","Virginia","Washington",
    "West Virginia","Wisconsin","Wyoming"
}

def normalize_scope_inputs(scope: str, region_name: str):
    s = (scope or "").strip().lower()
    rn = (region_name or "").strip()

    if s == "state":
        
        if "," in rn:
            state_part = rn.split(",")[0].strip()
        else:
            state_part = rn
        
        if state_part in US_STATE_FULL_SET:
            return state_part, f"{state_part}, United States"
        
        return state_part, rn

    return rn, rn



# fix column names for consistency
if 'pop' in worldcities_df.columns and 'population' not in worldcities_df.columns:
    worldcities_df = worldcities_df.rename(columns={'pop': 'population'})
if 'lon' in worldcities_df.columns and 'lng' not in worldcities_df.columns:
    worldcities_df = worldcities_df.rename(columns={'lon': 'lng'})

#Creates a fast lookup index for world cities population and lat/lon coordinates
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

# builds index for every world city in worldcities.csv to make data training efficient
WORLD_INDEX = build_worldcities_index(worldcities_df)

def lookup_city_geo_pop_fast(city, country, world_index=WORLD_INDEX):
    key = (str(city).lower().strip(), str(country).lower().strip())
    return world_index.get(key, (None, None, np.nan))

# caches all OpenStreetMap queries to avoid hitting the API too often
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

# Function to get cinema data from OpenStreetMap using Overpass API
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
                "operator": tags.get("operator"),
                "city": tags.get("addr:city"),
                "lat": lat_result,
                "lon": lon_result,
                "tags": tags
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

# Geocoding setup
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

# --------------------------------------------
# PATCHED: Heuristic ranker
# --------------------------------------------
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

    
    region_filter, geocode_query = normalize_scope_inputs(scope, region_name)

    # (1) Candidates from worldcities using the normalized FILTER value
    candidates = filter_candidates_worldcities(worldcities_df, scope, region_filter)
    if candidates.empty:
        print("No candidate cities for this scope/region.")
        return candidates
    if max_rows:
        candidates = candidates.head(int(max_rows))

    # (2) Precompute region cinemas using the normalized GEOCODE value
    tree = None
    if scope_l != "global" and use_osm:
        try:
            tree = ensure_region_precomputed(scope, geocode_query)
        except Exception as e:
            print(f"[warn] Precompute failed for {scope}:{geocode_query} ({e}). Falling back to per-city OSM).")
            tree = None

    # (3) Features
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
    if {"City","Country"}.issubset(imax_theaters_df.columns):
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
        p = model.predict_proba(X)[0, 1]

        rows.append({
            "City": city, "Country": country, "lat": lat, "lon": lon,
            "population": pop, "ml_prob": float(p)
        })
    return pd.DataFrame(rows).sort_values("ml_prob", ascending=False)


def filter_near_imax(df, min_km=20):
    df = df.copy()
    df["nearest_imax_km"] = df.apply(lambda r: nearest_imax_km(r["lat"], r["lon"]), axis=1)
    return df[df["nearest_imax_km"] >= min_km]


# caching helplers to improve efficiency with the ML training and scoring

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

def _slug(s: str) -> str:
    s = (s or "").strip().lower().replace(" ", "_")
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in s)

def _ml_config_key(radius_km: int, algo: str, sample_neg_per_country: int, imax_len: int) -> str:
    return f"r{int(radius_km)}_{algo}_neg{int(sample_neg_per_country)}_imax{int(imax_len)}"

def _train_cache_path(radius_km: int, algo: str, sample_neg_per_country: int, imax_len: int) -> Path:
    return CACHE_DIR / f"train_{_ml_config_key(radius_km, algo, sample_neg_per_country, imax_len)}.pkl"

def _model_cache_path(radius_km: int, algo: str, sample_neg_per_country: int, imax_len: int) -> Path:
    return CACHE_DIR / f"model_{_ml_config_key(radius_km, algo, sample_neg_per_country, imax_len)}.joblib"

def _cand_cache_path(scope: str, region_name: str, radius_km: int, algo: str, sample_neg_per_country: int, imax_len: int) -> Path:
    slug = _slug(region_name) if region_name else "global"
    return CACHE_DIR / f"cand_{scope.lower()}_{slug}_{_ml_config_key(radius_km, algo, sample_neg_per_country, imax_len)}.pkl"


# =========================
# Cached builders
# =========================
def build_training_table_cached(worldcities_df, imax_df,
                                sample_neg_per_country=150, radius_km=20, tree=None, use_osm=False,
                                algo="rf", rebuild=False):
    """
    Returns training DataFrame, using cache unless rebuild=True.
    Cache key depends on: radius_km, algo, sample_neg_per_country, len(IMAX list).
    """
    imax_len = int(imax_df.drop_duplicates(subset=["City", "Country"]).shape[0])
    train_path = _train_cache_path(radius_km, algo, sample_neg_per_country, imax_len)

    if (not rebuild) and train_path.exists():
        try:
            df = pd.read_pickle(train_path)
            # quick sanity check
            if {"label", "city", "country"}.issubset(df.columns):
                return df
        except Exception:
            pass  # fall through to rebuild

    df = build_training_table(worldcities_df, imax_df,
                              sample_neg_per_country=sample_neg_per_country,
                              radius_km=radius_km, tree=tree, use_osm=use_osm)
    try:
        df.to_pickle(train_path)
    except Exception:
        pass
    return df


def train_or_load_model(train_df: pd.DataFrame, algo="rf",
                        radius_km=20, sample_neg_per_country=150, imax_len=None,
                        retrain=False):
    """
    Returns fitted model, using cache unless retrain=True.
    """
    if imax_len is None:
        # derive from positives in train_df
        imax_len = int((train_df["label"] == 1).sum())

    model_path = _model_cache_path(radius_km, algo, sample_neg_per_country, imax_len)

    if (not retrain) and model_path.exists():
        try:
            return joblib.load(model_path)
        except Exception:
            pass  # fall through to retrain

    model = train_model(train_df, algo=algo)
    try:
        joblib.dump(model, model_path)
    except Exception:
        pass
    return model


def build_candidate_features_cached(candidates_df: pd.DataFrame,
                                    radius_km=20, tree=None, use_osm=False,
                                    scope="global", region_name="",
                                    algo="rf", sample_neg_per_country=150, imax_len=0,
                                    rebuild=False):
    """
    Precompute feature rows for candidate cities and cache them.
    """
    cand_path = _cand_cache_path(scope, region_name, radius_km, algo, sample_neg_per_country, imax_len)

    if (not rebuild) and cand_path.exists():
        try:
            return pd.read_pickle(cand_path)
        except Exception:
            pass  # rebuild

    rows = []
    for _, row in candidates_df.iterrows():
        city, country = row["city"], row["country"]
        lat, lon = float(row["lat"]), float(row["lon"])
        pop = row.get("population", np.nan)

        f = feature_row(lat, lon, pop, radius_km=radius_km, tree=tree, use_osm=use_osm)
        f.update({"City": city, "Country": country, "lat": lat, "lon": lon, "population": pop})
        rows.append(f)

    feats = pd.DataFrame(rows)
    try:
        feats.to_pickle(cand_path)
    except Exception:
        pass
    return feats

# --------------------------------------------
# PATCHED: ML ranker (with your caching flags/signature)
# --------------------------------------------
def top_locations_ml(scope, region_name="", top_n=5, radius_km=20,
                     use_osm_for_train=False, use_osm_for_score=False, algo="rf",
                     sample_neg_per_country=150,
                     rebuild_train=False, retrain_model=False, rebuild_features=False):
    """
    ML version with caching.
      - Caches training table, fitted model, and candidate features.
      - First run is heavier; subsequent runs are fast (load + predict).
    """
    scope_l = scope.lower()

    # ✅ NEW: normalize once for filtering vs geocoding
    region_filter, geocode_query = normalize_scope_inputs(scope, region_name)

    # BallTree for the scope (uses geocode_query so “California” resolves reliably)
    tree = None
    if scope_l != "global" and (use_osm_for_train or use_osm_for_score):
        try:
            tree = ensure_region_precomputed(scope, geocode_query)
        except Exception as e:
            print(f"[warn] Precompute failed for {scope}:{geocode_query} ({e}). Proceeding without BallTree.")

    # 1) Training table (cached)
    train_df = build_training_table_cached(
        worldcities_df, imax_theaters_df,
        sample_neg_per_country=sample_neg_per_country,
        radius_km=radius_km,
        tree=tree, use_osm=use_osm_for_train,
        algo=algo,
        rebuild=rebuild_train
    )

    # 2) Model (cached)
    imax_len = int(imax_theaters_df.drop_duplicates(subset=["City", "Country"]).shape[0])
    model = train_or_load_model(
        train_df, algo=algo,
        radius_km=radius_km, sample_neg_per_country=sample_neg_per_country, imax_len=imax_len,
        retrain=retrain_model
    )

    # 3) Candidate cities in scope (uses region_filter so it matches worldcities admin_name)
    candidates = filter_candidates_worldcities(worldcities_df, scope, region_filter)
    if candidates.empty:
        print("No candidate cities for this scope/region.")
        return candidates

    # 4) Candidate features (cached)
    feats = build_candidate_features_cached(
        candidates,
        radius_km=radius_km, tree=tree, use_osm=use_osm_for_score,
        scope=scope, region_name=region_filter,
        algo=algo, sample_neg_per_country=sample_neg_per_country, imax_len=imax_len,
        rebuild=rebuild_features
    )

    # 5) Predict probabilities
    X = feats[FEATURES].fillna(0.0).to_numpy()
    ml_prob = model.predict_proba(X)[:, 1]
    out = feats.copy()
    out["ml_prob"] = ml_prob

    # 6) Filter by nearest IMAX + remove exact IMAX cities
    out = filter_near_imax(out.rename(columns={"City":"City","Country":"Country","lat":"lat","lon":"lon"}), min_km=20)
    imax_keys = set(zip(imax_theaters_df["City"].str.lower().str.strip(),
                        imax_theaters_df["Country"].str.lower().str.strip()))
    out = out[~out.apply(
        lambda r: (str(r["City"]).lower().strip(), str(r["Country"]).lower().strip()) in imax_keys, axis=1
    )]

    cols = ["City", "Country", "lat", "lon", "population", "ml_prob"]
    return out.sort_values("ml_prob", ascending=False)[cols].head(top_n)



# =========================
# NEW: City-scope non-IMAX venue recommender
# =========================

def build_imax_index(imax_df: pd.DataFrame):
    """
    Build {(city_l, country_l): [(lat, lon, name), ...]} for fast in-city IMAX exclusion.
    Uses WORLD_INDEX to geocode IMAX city to a centroid; if you have per-venue coords, swap in those instead.
    """
    idx = {}
    for _, r in imax_df.iterrows():
        city = str(r.get("City","")).strip()
        country = str(r.get("Country","")).strip()
        if not city or not country:
            continue
        lat, lon, _ = lookup_city_geo_pop_fast(city, country)
        if lat is None or lon is None:
            continue
        key = (city.lower(), country.lower())
        name = str(r.get("Theater Name", r.get("Cinema", ""))).strip()  # optional name column fallback
        idx.setdefault(key, []).append((float(lat), float(lon), name))
    return idx

def slugify(s: str):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in s.strip().lower().replace(" ", "_"))

def nominatim_osm_area_id(city: str, country: str):
    """
    Use Nominatim to resolve the OSM object (relation/way) for the city, then convert to Overpass area id.
    area id formula: relation -> 3600000000 + rel_id; way -> 2400000000 + way_id
    """
    q = f"{city}, {country}"
    loc = geolocator.geocode(q, exactly_one=True, addressdetails=True, timeout=20)
    time.sleep(1.0)
    if not loc or "osm_type" not in loc.raw or "osm_id" not in loc.raw:
        raise ValueError(f"Could not resolve OSM object for '{q}'")
    osm_type = loc.raw["osm_type"]
    osm_id   = int(loc.raw["osm_id"])
    if osm_type == "relation":
        return 3600000000 + osm_id
    elif osm_type == "way":
        return 2400000000 + osm_id
    else:
        # Fallback: use bbox if it's a node; less precise but still useful
        raise ValueError(f"City resolved to unsupported osm_type '{osm_type}' for '{q}'")

def fetch_city_cinemas(city: str, country: str):
    """
    One-shot Overpass query to pull all cinemas inside the city administrative area.
    Caches to data/cinemas_city_{city}_{country}.json and .npy
    """
    slug = f"{slugify(city)}_{slugify(country)}"
    jpath = DATA_DIR / f"cinemas_city_{slug}.json"
    nppath = DATA_DIR / f"cinemas_city_{slug}.npy"

    if jpath.exists() and nppath.exists():
        try:
            with open(jpath, "r", encoding="utf-8") as f:
                raw = json.load(f)
            arr = np.load(nppath)
            return raw, arr
        except Exception:
            pass

    # Prefer polygon/area query
    try:
        area_id = nominatim_osm_area_id(city, country)
        query = f"""
        [out:json][timeout:120];
        area({area_id})->.searchArea;
        (
          node["amenity"="cinema"](area.searchArea);
          way["amenity"="cinema"](area.searchArea);
          relation["amenity"="cinema"](area.searchArea);
        );
        out center tags;
        """
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=180)
        r.raise_for_status()
        data = r.json()
    except Exception:
        # Fallback: tight bbox around city centroid
        lat, lon, _ = lookup_city_geo_pop_fast(city, country)
        if lat is None or lon is None:
            raise
        delta = 0.15  # ~15–20km box (tunable)
        south, north = lat - delta, lat + delta
        west,  east  = lon - delta, lon + delta
        query = f"""
        [out:json][timeout:120];
        (
          node["amenity"="cinema"]({south},{west},{north},{east});
          way["amenity"="cinema"]({south},{west},{north},{east});
          relation["amenity"="cinema"]({south},{west},{north},{east});
        );
        out center tags;
        """
        r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query}, timeout=180)
        r.raise_for_status()
        data = r.json()

    elements = data.get("elements", [])
    cinemas = []
    pts = []
    for el in elements:
        tags = el.get("tags", {}) or {}
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        cinemas.append({
            "type": el.get("type"),
            "name": tags.get("name"),
            "brand": tags.get("brand"),
            "operator": tags.get("operator"),
            "addr:city": tags.get("addr:city"),
            "lat": float(lat),
            "lon": float(lon),
            "tags": tags
        })
        pts.append((float(lat), float(lon)))

    # cache
    try:
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump({"elements": cinemas}, f)
        np.save(nppath, np.array(pts, dtype=np.float64))
    except Exception:
        pass

    return {"elements": cinemas}, np.array(pts, dtype=np.float64)

def ensure_city_balltree(city: str, country: str):
    """
    Load (or build) a BallTree for all cinemas in a given city.
    """
    _, arr = fetch_city_cinemas(city, country)
    if arr.size == 0:
        return None
    return BallTree(np.radians(arr), metric="haversine")

def _normalize_name(s: str):
    if not s:
        return ""
    return "".join(ch for ch in s.lower() if ch.isalnum() or ch.isspace()).strip()

def exclude_imax_matches(osm_cinemas: list, imax_index: dict,
                         city: str, country: str,
                         fuzz_threshold: int = 92, prox_m: float = 700.0):
    """
    Remove OSM cinemas that appear to be the IMAX venue (by fuzzy name or geo proximity).
    """
    key = (city.lower().strip(), country.lower().strip())
    imax_list = imax_index.get(key, [])
    if not imax_list:
        return osm_cinemas  # nothing to exclude

    # Build BallTree over IMAX points in this city for fast proximity checks
    imax_pts = np.array([(lat, lon) for (lat, lon, _) in imax_list], dtype=np.float64)
    imax_tree = BallTree(np.radians(imax_pts), metric="haversine") if len(imax_pts) else None
    earth_km = 6371.0088
    prox_radians = (prox_m / 1000.0) / earth_km

    survivors = []
    for c in osm_cinemas:
        cname = _normalize_name(c.get("name") or "") or _normalize_name(c.get("brand") or "") or _normalize_name(c.get("operator") or "")
        lat, lon = float(c["lat"]), float(c["lon"])

        # 1) name fuzzy match
        is_imax_name = False
        for _, _, iname in imax_list:
            if not iname:
                continue
            score = fuzz.token_set_ratio(cname, _normalize_name(iname))
            if score >= fuzz_threshold:
                is_imax_name = True
                break
        if is_imax_name:
            continue  # exclude

        # 2) proximity match
        if imax_tree is not None:
            cnt = imax_tree.query_radius(np.radians([[lat, lon]]), r=prox_radians, count_only=True)[0]
            if cnt > 0:
                continue  # exclude

        survivors.append(c)

    return survivors

def score_city_cinemas(cinemas: list, city_lat: float, city_lon: float,
                       city_population: float,
                       tree_city: BallTree,
                       local_km: float = 3.0):
    """
    Light-weight, explainable scoring for venues within a single city.
    Features:
      - local_cinema_density: # other cinemas within local_km
      - dist_to_center_km: distance to city centroid (closer slightly better)
      - whitespace: distance to nearest IMAX (km)
    """
    rows = []
    earth_km = 6371.0088
    r = local_km / earth_km

    # Array of ALL city cinema points in radians for density calc
    pts = np.radians(np.array([(c["lat"], c["lon"]) for c in cinemas], dtype=np.float64))

    for i, c in enumerate(cinemas):
        lat, lon = float(c["lat"]), float(c["lon"])

        # Local density excluding itself
        q = np.radians([[lat, lon]])
        idxs = tree_city.query_radius(q, r=r, return_distance=False)[0]
        local_density = max(0, len(idxs) - 1)

        # Distance to city center
        dist_center = haversine_km(lat, lon, city_lat, city_lon)

        # White-space: distance to nearest IMAX (uses global IMAX list)
        whitespace_km = nearest_imax_km(lat, lon)

        # Simple score (tunable): prefer some local ecosystem, closer to center, far from IMAX
        score = (0.4 * np.log1p(local_density)) + (0.2 * (1.0 / (1.0 + dist_center))) + (0.4 * (whitespace_km / 50.0))

        rows.append({
            "name": c.get("name"),
            "brand": c.get("brand"),
            "operator": c.get("operator"),
            "lat": lat, "lon": lon,
            "local_cinemas_within_km": local_km,
            "local_cinema_density": local_density,
            "dist_to_center_km": dist_center,
            "nearest_imax_km": whitespace_km,
            "score": float(score)
        })

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    return df

def recommend_non_imax_cinemas(city: str, country: str, top_n: int = 5,
                               fuzz_threshold: int = 92, prox_m: float = 700.0,
                               local_km: float = 3.0):
    """
    City-scope entry point:
      1) fetch all cinemas within city polygon (cached)
      2) exclude IMAX venues by name/proximity
      3) score remaining venues and return top N
    """
    # Resolve city centroid & population (for context/scoring)
    city_lat, city_lon, city_pop = lookup_city_geo_pop_fast(city, country)
    if city_lat is None or city_lon is None:
        print(f"Could not resolve city centroid for {city}, {country}")
        return pd.DataFrame()

    # Pull raw cinemas for the city
    raw, _arr = fetch_city_cinemas(city, country)
    cinemas = raw.get("elements", [])
    if not cinemas:
        print(f"No cinemas found inside {city}, {country}")
        return pd.DataFrame()

    # Build IMAX index and exclude IMAX matches
    imax_idx = build_imax_index(imax_theaters_df)
    candidates = exclude_imax_matches(cinemas, imax_idx, city, country,
                                      fuzz_threshold=fuzz_threshold, prox_m=prox_m)
    if not candidates:
        print("All cinemas appear to be IMAX or no non-IMAX candidates remain.")
        return pd.DataFrame()

    # Score within-city venues
    tree_city = ensure_city_balltree(city, country)
    if tree_city is None:
        print("City cinema BallTree could not be built.")
        return pd.DataFrame()

    ranked = score_city_cinemas(candidates, city_lat, city_lon, city_pop, tree_city, local_km=local_km)
    return ranked.head(top_n)
