# app.py
import time
import unicodedata
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
import pydeck as pdk

# ---------------- Backend hooks ----------------
from main import (
    top_locations,
    top_locations_ml,
    recommend_non_imax_cinemas,
    worldcities_df,
    imax_theaters_df,
    lookup_city_geo_pop_fast,
)

# optional: fetch all city cinemas for the grey overlay in City Mode
try:
    from main import fetch_city_cinemas
    HAVE_FETCH_ALL_CITY = True
except Exception:
    HAVE_FETCH_ALL_CITY = False

# ---------------- Page config ----------------
st.set_page_config(page_title="IMAX Geo Site Selector", layout="wide")

# Mapbox key not needed when we use built-in provider styles
pdk.settings.mapbox_api_key = ""

st.title("IMAX Site Selection – Explorer")

# ---------------- Utilities ----------------
def _norm(s: str) -> str:
    """Lowercase, trim, and strip accents/diacritics."""
    s = str(s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def fuzzy_eq(a: str, b: str, threshold: float = 0.82) -> bool:
    """
    Accent-insensitive, case-insensitive fuzzy comparison.
    Returns True if:
      - exact after normalization,
      - one contains the other (helps with variants),
      - or SequenceMatcher ratio >= threshold.
    """
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    if na in nb or nb in na:
        return True
    return SequenceMatcher(None, na, nb).ratio() >= threshold

def overpass_elements_to_points(elements: list[dict]) -> pd.DataFrame:
    """
    Flatten Overpass JSON elements (node/way/relation) into a points dataframe.
    Uses element.lat/lon or center.lat/center.lon; keeps a few useful tags.
    """
    rows = []
    for e in elements or []:
        center = e.get("center") or {}
        lat = e.get("lat") or center.get("lat")
        lon = e.get("lon") or center.get("lon")
        if lat is None or lon is None:
            continue

        tags = e.get("tags") or {}
        rows.append({
            "latitude": float(lat),
            "longitude": float(lon),
            "name": tags.get("name", ""),
            "brand": tags.get("brand", ""),
            "operator": tags.get("operator", ""),
            "source_type": e.get("type", ""),
            "id": e.get("id", None),
        })
    return pd.DataFrame(rows)

def build_imax_points_for_scope(scope: str, region: str | None) -> pd.DataFrame:
    """
    For Rank Cities view: build a point layer of existing IMAX theaters in the selected scope.
    Fuzzy matches on Country or State (admin).
    Falls back to geocoding city centroids if theater lat/lon aren't in the dataset.
    """
    try:
        df = imax_theaters_df.copy()
        reg = (region or "").strip()
        if scope == "country" and reg:
            df = df[df["Country"].astype(str).apply(lambda x: fuzzy_eq(x, reg))]
        elif scope == "state" and reg and "State" in df.columns:
            df = df[df["State"].astype(str).apply(lambda x: fuzzy_eq(x, reg))]

        rows = []
        for _, r in df.iterrows():
            city = str(r.get("City", "")).strip()
            country = str(r.get("Country", "")).strip()
            lat = r.get("Latitude")
            lon = r.get("Longitude")

            if pd.isna(lat) or pd.isna(lon):
                # fallback to city centroid if no theater coords available
                lat, lon, _ = lookup_city_geo_pop_fast(city, country)

            if lat is not None and lon is not None:
                rows.append({"City": city, "Country": country, "lat": float(lat), "lon": float(lon)})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

def reset_rank_view():
    # wipe previous ranking results + scope memory
    st.session_state.pop("last_results", None)
    st.session_state.pop("last_scope", None)
    st.session_state.pop("last_region", None)
    st.session_state.pop("last_engine", None)
    st.session_state.pop("last_radius_km", None)
    st.session_state.pop("last_top_n", None)
    # also clear jump picker selection if present
    st.session_state.pop("tab1_city_picker", None)
    # stop any pending handoff
    st.session_state["autostart_city_query"] = False
    # immediately re-render to show the default/info state
    st.rerun()

def reset_city_view():
    # nothing is persisted for venues table/map, so just ensure we won’t auto-run
    st.session_state["autostart_city_query"] = False
    # clear any prior prefill (optional)
    st.session_state["city_prefill"] = ""
    st.session_state["country_prefill"] = ""
    st.rerun()



def build_imax_points_for_city(city: str, country: str) -> pd.DataFrame:
    """
    For City Venue view: build a point layer of existing IMAX theaters in/near the city.
    Fuzzy matches on City + Country. Uses theater coordinates when present, otherwise city centroid.
    """
    try:
        df = imax_theaters_df.copy()
        df = df[
            df["Country"].astype(str).apply(lambda x: fuzzy_eq(x, country))
            & df["City"].astype(str).apply(lambda x: fuzzy_eq(x, city))
        ]

        rows = []
        for _, r in df.iterrows():
            t_city = str(r.get("City", "")).strip()
            t_country = str(r.get("Country", "")).strip()
            lat = r.get("Latitude")
            lon = r.get("Longitude")

            if pd.isna(lat) or pd.isna(lon):
                lat, lon, _ = lookup_city_geo_pop_fast(t_city, t_country)

            if lat is not None and lon is not None:
                rows.append({"name": f"IMAX – {t_city}", "lat": float(lat), "lon": float(lon)})
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# ---------------- Session/init helpers ----------------
def _init_state():
    st.session_state.setdefault("view", "rank")        # 'rank' or 'venues'
    st.session_state.setdefault("pending_view", None)
    st.session_state.setdefault("last_scope", None)
    st.session_state.setdefault("last_region", None)
    st.session_state.setdefault("last_engine", None)
    st.session_state.setdefault("last_radius_km", None)
    st.session_state.setdefault("last_top_n", None)

    st.session_state.setdefault("city_prefill", "")
    st.session_state.setdefault("country_prefill", "")
    st.session_state.setdefault("autostart_city_query", False)

_init_state()
# Apply pending programmatic navigation BEFORE widgets are created
if st.session_state.get("pending_view"):
    st.session_state["view"] = st.session_state.pop("pending_view")

# ---------------- Validation helpers ----------------
def region_exists(scope: str, region: str) -> bool:
    """Lightweight existence check against worldcities_df."""
    if scope == "global":
        return True
    r = (region or "").strip().lower()
    if not r:
        return False
    df = worldcities_df.copy()
    df["country_l"] = df["country"].astype(str).str.lower().str.strip()
    if "admin_name" in df.columns:
        df["admin_l"] = df["admin_name"].astype(str).str.lower().str.strip()

    if scope == "country":
        return (df["country_l"] == r).any()
    if scope == "state" and "admin_l" in df.columns:
        return (df["admin_l"] == r).any()
    return False

def city_country_exists(city: str, country: str) -> bool:
    if not city or not country:
        return False
    df = worldcities_df.copy()
    df["city_l"] = df["city"].astype(str).str.lower().str.strip()
    df["country_l"] = df["country"].astype(str).str.lower().str.strip()
    return ((df["city_l"] == city.strip().lower()) & (df["country_l"] == country.strip().lower())).any()

# ---------------- Map helper ----------------
def show_map(results_df: pd.DataFrame,
             imax_layer_df: pd.DataFrame | None = None,
             zoom_guess: float = 3,
             tooltip_fields: list[str] | None = None):
    """Scatter map of result cities + optional IMAX overlay. Uses carto provider (no OSM tile URL)."""
    if results_df is None or results_df.empty:
        return

    # Ensure needed cols exist
    if not {"lat", "lon"}.issubset(results_df.columns):
        return

    view_state = pdk.ViewState(
        latitude=float(results_df["lat"].mean()),
        longitude=float(results_df["lon"].mean()),
        zoom=zoom_guess,
        pitch=0,
        bearing=0,
    )

    # Candidates layer (red)
    cand_layer = pdk.Layer(
        "ScatterplotLayer",
        data=results_df,
        get_position="[lon, lat]",
        get_radius=1500,
        radius_min_pixels=2,
        radius_max_pixels=12,
        get_fill_color=[255, 64, 64, 180],
        pickable=True,
    )
    layers = [cand_layer]

    # Existing IMAX (blue)
    if imax_layer_df is not None and not imax_layer_df.empty and {"lat","lon"}.issubset(imax_layer_df.columns):
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=imax_layer_df,
                get_position="[lon, lat]",
                get_radius=1200,
                radius_min_pixels=1,
                radius_max_pixels=8,
                get_fill_color=[0, 122, 255, 160],
                pickable=True,
            )
        )

    # HTML tooltip
    tooltip = None
    if tooltip_fields:
        # build a simple HTML block
        lines = []
        for c in tooltip_fields:
            lines.append(f"<b>{c}:</b> {{{{{c}}}}}")
        tooltip = {
            "html": "<br/>".join(lines),
            "style": {"backgroundColor": "#1f2937", "color": "white"}
        }

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            map_provider="carto",   # fixes OSM tile errors
            map_style="light",
            tooltip=tooltip,
        ),
        use_container_width=True
    )

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Navigation")
    st.radio(
        "Mode",
        options=("rank", "venues"),
        format_func=lambda v: "Rank Cities" if v == "rank" else "City Venue Mode",
        key="view",
    )

    st.divider()
    st.header("Options")

    if st.session_state["view"] == "rank":
        # ---- Rank Cities–only controls ----
        st.slider("Top N", 1, 50, 5, 1, key="rank_top_n")
        st.slider("Radius (km) for cinema density features", 5, 40, 20, 1, key="rank_radius_km")
        st.radio("Ranking Engine", ["Heuristic (fast)", "ML (scikit-learn, cached)"],
                 index=0, key="rank_engine")
        st.caption("Tip: Heuristic is quick for broad sweeps; ML gives nuanced ranking.")
        st.divider()
        run_btn = st.button("Run / Refresh", key="run_rank")

    elif st.session_state["view"] == "venues":
        # ---- City Venue Mode–only controls ----
        st.slider("Top candidate venues to show", 1, 30, 5, 1, key="city_top_n")
        st.checkbox("Also show all cinemas in the city (grey)", value=True, key="city_show_all_cinemas")
        st.divider()
        run_btn = st.button("Run / Refresh", key="run_city")

# ---------------- Views ----------------
# ---------- TAB 1: Rank Cities ----------
if st.session_state["view"] == "rank":
    st.subheader("Rank Candidate Cities")

    # On-page scope UI (dropdown + optional region text box)
    st.markdown("**Ranking scope**")
    scope = st.selectbox("Scope", ["global", "country", "state"], index=1, key="scope_main", on_change=reset_rank_view,)
    region = "" if scope == "global" else st.text_input(f"{scope.title()} name", value="", key="region_main", on_change=reset_rank_view)

    # Read sidebar options for this mode
    top_n = st.session_state.get("rank_top_n", 5)
    radius_km = st.session_state.get("rank_radius_km", 20)
    engine = st.session_state.get("rank_engine", "Heuristic (fast)")
    run_btn = st.session_state.get("run_rank", False)

    # 1) RUN
    if run_btn:
        # Validate inputs
        if scope != "global" and not region.strip():
            st.warning("Please enter a region name.")
            st.stop()
        if scope != "global" and not region_exists(scope, region):
            st.error(f"'{region}' doesn't look like a valid {scope}. Check spelling (and for states, use admin name).")
            st.stop()

        # Execute ranking
        with st.spinner("Ranking candidate cities… this can take a moment if OSM/cache is cold."):
            t0 = time.time()
            try:
                HEAVY_COUNTRIES = {"united states", "china", "india", "russia", "brazil"}

                # decide whether to allow OSM
                wants_osm = True
                if scope == "country" and region.strip().lower() in HEAVY_COUNTRIES:
                    wants_osm = False
                if str(engine).startswith("Heuristic"):
                    results = top_locations(scope, region, top_n=top_n, radius_km=radius_km, use_osm=wants_osm)
                else:
                    results = top_locations_ml(
                        scope, region, top_n=top_n, radius_km=radius_km,
                        use_osm_for_train=wants_osm, use_osm_for_score=wants_osm,
                        algo="rf", sample_neg_per_country=150,
                        rebuild_train=False, retrain_model=False, rebuild_features=False)
            except Exception as e:
                st.error(f"Error while ranking: {e}")
                results = pd.DataFrame()
            elapsed = time.time() - t0

        st.caption(f"Completed in {elapsed:.2f}s")

        # Persist for rerender / jump
        st.session_state["last_results"] = results
        st.session_state["last_scope"] = scope
        st.session_state["last_region"] = region
        st.session_state["last_engine"] = engine
        st.session_state["last_radius_km"] = radius_km
        st.session_state["last_top_n"] = top_n

    # 2) RENDER FROM SESSION
    results = st.session_state.get("last_results")
    scope_used = st.session_state.get("last_scope") or scope
    region_used = st.session_state.get("last_region") or region

    if results is None or results.empty:
        st.info("Click **Run / Refresh** to rank candidate cities.")
        st.stop()

    # Table
    st.dataframe(results, use_container_width=True)

    # Export CSV
    csv_bytes = results.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"imax_candidates_{scope_used or scope}.csv",
        mime="text/csv",
        key="rank_download_csv"
    )

    # IMAX overlay (blue) using fuzzy scope filtering
    imax_layer_df = build_imax_points_for_scope(scope_used, region_used)

    # Map of results + IMAX overlay
    show_map(
        results,
        imax_layer_df,
        zoom_guess=3 if scope_used in ("global", "country") else 5,
        tooltip_fields=[
            c for c in ["City", "Country", "population", "cinema_count_radius_km",
                        "cinemas_per_100k", "score", "ml_prob"]
            if c in results.columns
        ],
    )

    # Hand-off → City Venue Mode
    selectable = (
        results[["City", "Country"]]
        .dropna()
        .astype(str)
        .assign(label=lambda d: d["City"].str.strip() + ", " + d["Country"].str.strip())
    )

    st.markdown("**Jump to City Venue Mode**")
    chosen = st.selectbox(
        "Choose a city from these results:",
        options=[""] + selectable["label"].tolist(),
        index=0,
        key="tab1_city_picker",
    )

    go_city = st.button(
        "Open in City Venue Mode",
        type="primary",
        use_container_width=False,
        key="go_to_city_mode",
    )
    if go_city:
        if not chosen:
            st.warning("Pick a city first.")
        else:
            city, country = [p.strip() for p in chosen.split(",", 1)]
            # Defer navigation to next run (avoids Streamlit mutation error)
            st.session_state["pending_view"] = "venues"
            st.session_state["city_prefill"] = city
            st.session_state["country_prefill"] = country
            st.session_state["autostart_city_query"] = True
            st.rerun()

# ---------- TAB 2: City Venue Candidates ----------
if st.session_state["view"] == "venues":
    st.subheader("Non-IMAX Venue Candidates (City Scope)")
    st.caption("Find specific cinema venues (not currently IMAX) within a city polygon, score them, and explore on the map.")

    # Prefill from jump (if any)
    city_name = st.session_state.get("city_prefill") or ""
    country_name = st.session_state.get("country_prefill") or ""

    city_col, country_col = st.columns(2)
    with city_col:
        city_name = st.text_input("City", value=city_name, key="city_input_live", on_change=reset_city_view,)
    with country_col:
        country_name = st.text_input("Country", value=country_name, key="country_input_live", on_change=reset_city_view,)

    # Decide if we auto-run (after jump) or wait for button
    auto = bool(st.session_state.get("autostart_city_query"))
    run_city = st.session_state.get("run_city") or auto

    if run_city:
        if not city_name or not country_name:
            st.warning("Enter both City and Country to continue.")
            st.session_state["autostart_city_query"] = False
            st.stop()

        if not city_country_exists(city_name, country_name):
            st.error(f"'{city_name}, {country_name}' was not found in the world cities data. Check spelling.")
            st.session_state["autostart_city_query"] = False
            st.stop()

        with st.spinner(f"Loading venue mode for {city_name}, {country_name}…"):
            try:
                venues = recommend_non_imax_cinemas(
                    city_name, country_name,
                    top_n=st.session_state.get("city_top_n", 5)
                )
            except Exception as e:
                st.error(f"Error fetching/processing venues: {e}")
                venues = pd.DataFrame()

        st.session_state["autostart_city_query"] = False

        if venues is None or venues.empty:
            st.warning("No non-IMAX candidate venues found.")
        else:
            st.dataframe(venues, use_container_width=True)

            # -------- Map layers for City Mode --------
            layers = []

            # (1) Optional overlay: all cinemas in city (grey), robust to nodes/ways/relations
            if HAVE_FETCH_ALL_CITY and st.session_state.get("city_show_all_cinemas", True):
                try:
                    raw, _ = fetch_city_cinemas(city_name, country_name)
                    pts = overpass_elements_to_points(raw.get("elements", []))
                    if not pts.empty:
                        pts["color_r"] = 140; pts["color_g"] = 140; pts["color_b"] = 140
                        pts["radius_m"] = 80
                        layers.append(
                            pdk.Layer(
                                "ScatterplotLayer",
                                data=pts,
                                get_position="[longitude, latitude]",
                                get_radius="radius_m",
                                get_fill_color="[color_r, color_g, color_b]",
                                pickable=True,
                            )
                        )
                    else:
                        st.info("OSM returned no cinema points for the grey overlay (none with coords).")
                except Exception as e:
                    st.info(f"Couldn’t fetch all-city cinemas overlay: {e}")

            # (2) Candidate venues (red)
            v = venues.rename(columns={"lat":"latitude","lon":"longitude"}).copy()
            v["color_r"] = 230; v["color_g"] = 30; v["color_b"] = 30
            v["radius_m"] = 90
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=v,
                    get_position="[longitude, latitude]",
                    get_radius="radius_m",
                    get_fill_color="[color_r, color_g, color_b]",
                    pickable=True,
                )
            )

            # (3) Existing IMAX in this city (blue, fuzzy matching)
            imax_city_df = build_imax_points_for_city(city_name, country_name)
            if not imax_city_df.empty:
                imax_city_df = imax_city_df.rename(columns={"lat": "latitude", "lon": "longitude"})
                layers.append(
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=imax_city_df,
                        get_position="[longitude, latitude]",
                        get_radius=1200,
                        radius_min_pixels=1,
                        radius_max_pixels=8,
                        get_fill_color=[0, 122, 255, 160],
                        pickable=True,
                    )
                )

            # View + render
            init_lat = float(v["latitude"].mean())
            init_lon = float(v["longitude"].mean())
            view_state = pdk.ViewState(latitude=init_lat, longitude=init_lon, zoom=12)

            tooltip = {
                "html": (
                    "<b>{name}</b><br/>"
                    "Local dens: {local_cinema_density}<br/>"
                    "Dist center: {dist_to_center_km} km<br/>"
                    "Nearest IMAX: {nearest_imax_km} km<br/>"
                    "<b>Score:</b> {score}"
                ),
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
            st.pydeck_chart(
                pdk.Deck(
                    layers=layers,
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_provider="carto",
                    map_style="light"
                ),
                use_container_width=True
            )

            # Quick links for top pick
            top_row = v.iloc[0]
            lat, lon = top_row["latitude"], top_row["longitude"]
            st.markdown(
                f"[Open in Google Maps](https://www.google.com/maps/search/?api=1&query={lat},{lon})  |  "
                f"[Street View](https://www.google.com/maps/@?api=1&map_action=pano&viewpoint={lat},{lon})"
            )
    else:
        st.info("Enter a city & country, then click **Run / Refresh**.")
