from pathlib import Path
import pandas as pd
from database import get_engine

PROJECT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_DIR/"data"/"raw"

OUTPUT_TABLE ="stations"

def extract_stations(data, endpoint):
    stations = data[[
        f"{endpoint}_station_id",
        f"{endpoint}_station_name",
        f"{endpoint}_lat",
        f"{endpoint}_lng"
    ]].copy()

    stations.columns = ["station_id", "station_name", "station_lat", "station_lng"]
    stations = stations.dropna(subset=["station_id", "station_lat", "station_lng"])

    stations["station_name"] = stations["station_name"].fillna(stations["station_id"])

    stations["observed_as_start"] = endpoint == "start"
    stations["observed_as_end"] = endpoint == "end"

    return stations

def load_stations(file_path):
    cols = ["start_station_id","start_station_name","start_lat","start_lng",
            "end_station_id","end_station_name","end_lat","end_lng"]
    csv_chunks =pd.read_csv(file_path,usecols=cols,
                    dtype={
                        "start_station_id":"string",
                        "start_station_name": "string",
                        "end_station_id": "string",
                        "end_station_name": "string"
                    }, low_memory=False, chunksize=250000)

    station_chunks =[]

    for data in csv_chunks:
        stations = pd.concat([extract_stations(data,"start"), extract_stations(data,"end")],
                            ignore_index=True)

        station_chunks.append(combine_stations(stations))

    return combine_stations(pd.concat(station_chunks, ignore_index=True))

def combine_stations(stations):
    combined_stations = stations.groupby("station_id", as_index=False).agg(
        station_name = ("station_name","last"),
        station_lat = ("station_lat","median"),
        station_lng = ("station_lng","median"),
        observed_as_start = ("observed_as_start", "any"),
        observed_as_end = ("observed_as_end","any")
    ).sort_values("station_id", ignore_index=True)

    return combined_stations

def main():
    csv_files = sorted(RAW_DATA_DIR.rglob("*.csv"))

    station_data = []
    for file_path in csv_files:
        print(f"Loading {file_path.name}")
        station_data.append(load_stations(file_path))

    stations = combine_stations(pd.concat(station_data, ignore_index=True))
    print(f"Found {len(stations):,} unique stations")
    
    engine = get_engine()
    stations.to_sql(OUTPUT_TABLE, engine, if_exists="replace", index=False)
    engine.dispose()

    print(f"Saved stations to PSQL table: {OUTPUT_TABLE}")

if __name__ == "__main__":
    main()