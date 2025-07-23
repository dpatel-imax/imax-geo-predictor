import pandas as pd
import requests
from geopy.geocoders import Nominatim
from thefuzz import process

imax_theaters_df = pd.read_csv('list_of_IMAX.csv')

def get_cinema_data_osm(lat, lon, radius_km=20):
    """
    Query OpenStreetMap Overpass API for cinemas near a given lat/lon.
    Returns full cinema data as a list of dictionaries.
    """
    delta = radius_km / 111  # Approximate degrees conversion
    south, north = lat - delta, lat + delta
    west, east = lon - delta, lon + delta

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
        response = requests.post("https://overpass-api.de/api/interpreter", data={'data': query})
        response.raise_for_status()
        result = response.json()
        cinemas = []

        for elem in result.get("elements", []):
            elem_type = elem.get("type")
            tags = elem.get("tags", {})
            name = tags.get("name", None)
            brand = tags.get("brand", None)
            addr_city = tags.get("addr:city", None)
            lat_result = elem.get("lat") or elem.get("center", {}).get("lat")
            lon_result = elem.get("lon") or elem.get("center", {}).get("lon")

            cinemas.append({
                "type": elem_type,
                "name": name,
                "brand": brand,
                "city": addr_city,
                "lat": lat_result,
                "lon": lon_result
            })

        return cinemas

    except Exception as e:
        # Return None or empty list so future code can handle gracefully
        return []


geolocator = Nominatim(user_agent="imax_site_selector")

def get_lat_lon_from_city(city_name):
    location = geolocator.geocode(city_name)
    if location:
        return location.latitude, location.longitude
    return None, None

worldcities_df = pd.read_csv('worldcities.csv')

# Standardize city names for comparison (lowercase, strip spaces)
imax_cities = imax_theaters_df['City'].str.lower().str.strip().unique()
worldcities = worldcities_df['city'].str.lower().str.strip().unique()

# Set a similarity threshold (e.g., 85 out of 100)
threshold = 95

not_found = []
approx_matches = {}

# For each IMAX location, check if city matches a worldcities city, and if so, check if country matches.
# If no city match, or city matches but country does not, write to names.txt in the requested format.

imax_cities = imax_theaters_df['City'].str.lower().str.strip()
imax_countries = imax_theaters_df['Country'].str.lower().str.strip()
worldcities_cities = worldcities_df['city'].str.lower().str.strip()
country_columns = [col for col in worldcities_df.columns if 'country' in col.lower()]

with open("names.txt", "w", encoding="utf-8") as f:
    for idx, (imax_city, imax_country) in enumerate(zip(imax_cities, imax_countries)):
        # Find all rows in worldcities_df where city matches
        matches = worldcities_df[worldcities_df['city'].str.lower().str.strip() == imax_city]
        found = False
        for _, row in matches.iterrows():
            # Check if any country column matches
            country_match = False
            for col in country_columns:
                if pd.notna(row[col]) and row[col].lower().strip() == imax_country:
                    country_match = True
                    break
            if not country_match:
                # City matches but country does not
                f.write(f"{imax_theaters_df.iloc[idx]['City']}, {imax_theaters_df.iloc[idx]['Country']} -> {row['city']}, {row[country_columns[0]]}\n")
                found = True
        if matches.empty:
            # No city match at all
            f.write(f"{imax_theaters_df.iloc[idx]['City']}, {imax_theaters_df.iloc[idx]['Country']} -> None, None\n")
    f.write("done\n")
