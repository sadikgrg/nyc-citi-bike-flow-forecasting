import pandas as pd
from sqlalchemy import text
from database import get_engine

FLOW_TABLE = "hourly_neighborhood_flow"
WEATHER_TABLE = "hourly_weather_history"
OUTPUT_TABLE = "neighborhood_hourly_model_data"

def main():
    engine = get_engine()
    with engine.connect() as connection:
        flow = pd.read_sql(text(
            f"""Select * From {FLOW_TABLE}"""
        ), con=connection)
        weather = pd.read_sql(text(
            f"""Select * From {WEATHER_TABLE}"""
        ), con=connection)

    training_data = flow.merge(weather, on="hour", how="left")
    training_data.to_sql(OUTPUT_TABLE, engine,if_exists="replace", index=False)
    engine.dispose()
    print(f"Saved {len(training_data):,} rows to PostgreSQL table: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
