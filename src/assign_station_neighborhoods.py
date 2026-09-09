import geopandas as gpd
import pandas as pd
from sqlalchemy import text

from database import get_engine

STATIONS_TABLE = "stations"
OUTPUT_TABLE = "station_neighborhoods"
NEIGHBORHOOD_URL = (
    "https://data.cityofnewyork.us/resource/9nt8-h7nd.geojson?$limit=500"
    )

def load_stations():
    engine = get_engine()

    with engine.connect() as connection:
        data = pd.read_sql(text(
            f"""Select * From {STATIONS_TABLE}"""
        ), con=connection)
    engine.dispose()

    return data

def load_neighborhoods():
    neighborhoods = gpd.read_file(NEIGHBORHOOD_URL)

    neighborhoods = neighborhoods[["nta2020", "ntaname", "boroname", "ntatype", "geometry"]].copy()

    neighborhoods.columns = ["nta_code","nta_name","borough","nta_type","geometry"]
    return neighborhoods.to_crs("EPSG:4326")

def assign_neighborhoods(stations, neighborhoods):
    station_points = gpd.GeoDataFrame(stations.copy(),
                                    geometry=gpd.points_from_xy(stations["station_lng"],
                                                                stations["station_lat"]),
                                                                crs="EPSG:4326")

    matched = gpd.sjoin(station_points, neighborhoods,how="left", predicate="within")

    matched["matched_neighborhood"] = matched["nta_code"].notna()

    return pd.DataFrame(matched.drop(columns=["geometry", "index_right"]))

def main():
    print("Loading stations")
    stations = load_stations()

    print("Downloading NYC neighborhood boundaries")
    neighborhoods = load_neighborhoods()

    print("Assigning stations to neighborhoods")
    station_neighborhoods = assign_neighborhoods(stations, neighborhoods)

    total = len(station_neighborhoods)
    matched = int(station_neighborhoods["matched_neighborhood"].sum())
    print(f"Total stations: {total:,}")
    print(f"Matched stations: {matched:,}")
    print(f"Unmatched stations: {total - matched:,}")
    print(
        "Neighborhoods containing stations: "
        f"{station_neighborhoods['nta_code'].nunique():,}"
    )

    engine = get_engine()

    station_neighborhoods.to_sql(

        OUTPUT_TABLE,
        engine,
        if_exists="replace",
        index=False,
        chunksize=5000
        )
    engine.dispose()

    print(f"Saved station-neighborhood mapping to PostgreSQL table: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
