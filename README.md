IMAX Geo Site Selector

A Streamlit web application for geospatial site selection and venue analysis, built to help identify high-potential cities and specific cinema venues for IMAX expansion. The tool integrates global city data, IMAX theater locations, and OpenStreetMap (Overpass API) to provide both macro-level city ranking and micro-level venue recommendations.

🚀 Features
1. Rank Cities Mode

Rank candidate cities by population, cinema density, and distance to nearest IMAX.

Two ranking engines:

Heuristic model (fast, interpretable).

Machine Learning model (Random Forest / Logistic Regression with GroupKFold CV).

CSV export of results for offline analysis.

Interactive map (PyDeck) showing candidate cities (red) and existing IMAX (blue).

2. City Venue Mode

Fetch all cinemas in a given city via OpenStreetMap (Overpass API).

Exclude existing IMAX venues using fuzzy string matching (thefuzz) and geospatial proximity checks.

Score non-IMAX venues based on:

Local cinema density,

Distance to city center,

Distance from nearest IMAX.

Interactive map layers with candidate venues, all cinemas (optional), and existing IMAX.

Quick links to Google Maps and Street View for top venues.

3. Performance & Robustness

Caching with joblib and pickled DataFrames for training sets, models, candidate features, and OSM queries.

Efficient geospatial algorithms using haversine distance and BallTree (scikit-learn).

Built-in input validation, error handling, and session state management for smooth navigation.

🛠️ Tech Stack

Frontend: Streamlit
, PyDeck

Data Processing: pandas, numpy

Geospatial Tools: geopy, scikit-learn BallTree (haversine metric), haversine formulas

Machine Learning: scikit-learn (Random Forest, Logistic Regression, GroupKFold CV)

Fuzzy Matching: thefuzz (string similarity)

Caching & Serialization: joblib, pickle, JSON

External Data Sources:

OpenStreetMap
 (Overpass API)

World Cities dataset (worldcities.csv)

IMAX theater list (list_of_IMAX.csv)

📦 Installation

Clone the repo and install dependencies:

git clone https://github.com/yourusername/imax-geo-selector.git
cd imax-geo-selector
pip install -r requirements.txt

▶️ Usage

Run the Streamlit app locally:

streamlit run app.py


The app will open in your browser at http://localhost:8501
.

📂 Data Requirements

worldcities.csv — global city data (population, coordinates, country, admin regions).

list_of_IMAX.csv — list of existing IMAX theaters with city/country (latitude/longitude optional).

Place these CSVs in the project root before running.

⚡ Example Workflows

Rank Cities

Choose scope: Global, Country, or State.

Select ranking engine: Heuristic or ML.

View ranked list of candidate cities and map overlay of IMAX vs. candidates.

Export results as CSV.

City Venue Analysis

Enter a City + Country.

Fetch all cinemas from OSM.

Exclude IMAX venues.

Get top non-IMAX candidates with scores and maps.

Jump directly from “Rank Cities” mode to “City Venue Mode.”

🔮 Roadmap

Add support for alternative ML algorithms (XGBoost, LightGBM).

Expand scoring features (income levels, tourism data, regional cinema chains).

Integrate real IMAX per-venue coordinates for higher precision.

Deploy as a hosted app (Streamlit Cloud / internal server).