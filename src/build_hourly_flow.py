from pathlib import Path
import pandas as pd
from sqlalchemy import text
from database import get_engine

PROJECT_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_TABLE = "hourly_neighborhood_flow"

def load_station_neighborhoods():
    engine = get_engine()
    with engine.connect() as connection:
        mapping = pd.read_sql(text(
            """
            Select station_id, nta_code, nta_name, borough
            From station_neighborhoods
            Where matched_neighborhood Is TRUE
            """
        ), con=connection)
    engine.dispose()

    return mapping.drop_duplicates("station_id")


def aggregate_events(data, mapping, time_column, station_column, count_column):
    events = data[[time_column, station_column]].copy()

    events.columns = ["hour", "station_id"]
    events["hour"] = pd.to_datetime(events["hour"]).dt.floor("h")
    events = events.dropna(subset=["hour", "station_id"])
    events = events.merge(mapping, on="station_id", how="inner")

    events = events.groupby(["hour","nta_code","nta_name","borough"],
                            as_index=False).size().rename(columns={"size":count_column})
    
    return events


def process_file(file_path, mapping):
    data = pd.read_csv(
        file_path,
        usecols=["started_at", "ended_at", "start_station_id", "end_station_id"],
        dtype={
            "start_station_id": "string",
            "end_station_id": "string",
            },low_memory=False
            )
    
    departures = aggregate_events(data,mapping,"started_at","start_station_id","departures")

    arrivals = aggregate_events(data,mapping,"ended_at","end_station_id","arrivals")

    keys = ["hour", "nta_code", "nta_name", "borough"]
    flow = departures.merge(arrivals, on=keys, how="outer")

    flow[["departures", "arrivals"]] = flow[["departures", "arrivals"]].fillna(0)

    return flow


def combine_files(csv_files, mapping):
    file_results = []
    for file_path in csv_files:
        print(f"Loading {file_path.name}")
        file_results.append(process_file(file_path, mapping))

    keys = ["hour", "nta_code", "nta_name", "borough"]

    flow = pd.concat(file_results, ignore_index=True)
    flow = flow.groupby(keys, as_index=False)[["departures", "arrivals"]].sum()
    flow["net_flow"] = flow["arrivals"] - flow["departures"]

    return flow.sort_values(["hour", "nta_code"], ignore_index=True)


def main():
    csv_files = sorted(RAW_DATA_DIR.rglob("*.csv"))

    print("Loading station-neighborhood mapping")
    mapping = load_station_neighborhoods()
    print(f"Found {len(mapping):,} matched NYC stations")

    flow = combine_files(csv_files, mapping)
    print(f"Created {len(flow):,} neighborhood-hour rows")
    print(f"Neighborhoods: {flow['nta_code'].nunique():,}")
    print(f"Hours: {flow['hour'].nunique():,}")

    engine = get_engine()
    flow.to_sql(
        OUTPUT_TABLE,
        engine,
        if_exists="replace",
        index=False,
        chunksize=10_000,
        )
    engine.dispose()

    print(f"Saved data to PostgreSQL table: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
