import pandas as pd
import requests
imax_theaters_df = pd.read_csv('list_of_IMAX.csv')

import requests

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


