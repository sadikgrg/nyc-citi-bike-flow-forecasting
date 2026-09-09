from pathlib import Path
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from sqlalchemy import text
from database import get_engine

PROJECT_DIR = Path(__file__).parent.parent
FLOW_TABLE = "hourly_neighborhood_flow"
OUTPUT_TABLE = "hourly_weather_history"


def get_flow_date_range():
    engine = get_engine()
    with engine.connect() as connection:
        date_range = pd.read_sql(text(
            f"""Select min(hour) as start_hour, max(hour) as end_hour
            From {FLOW_TABLE}"""
        ), con=connection)
    engine.dispose()

    start_hour  = date_range["start_hour"][0].strftime("%Y-%m-%d")
    end_hour = date_range["end_hour"][0].strftime("%Y-%m-%d")

    start_date = pd.Timestamp(start_hour).date().isoformat()
    end_date = pd.Timestamp(end_hour).date().isoformat()
    return start_date, end_date


def download_weather(start_date, end_date):
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
	"latitude": 40.71,
	"longitude": -74,
	"start_date": start_date,
	"end_date": end_date,
	"hourly": ["temperature_2m", "dew_point_2m", "precipitation", "snowfall", "wind_speed_10m", "weather_code"],
	"timezone": "America/New_York",
    }
    responses = openmeteo.weather_api(url, params = params)
    response = responses[0]
    hourly = response.Hourly()

    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
    hourly_snowfall = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(4).ValuesAsNumpy()
    hourly_weather_code = hourly.Variables(5).ValuesAsNumpy()

    hours = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left"
    )


    hours = hours.tz_convert("America/New_York").tz_localize(None)

    weather = pd.DataFrame(
        {
            "hour": hours,
            "temperature": hourly_temperature_2m,
            "dew_point": hourly_dew_point_2m,
            "precipitation": hourly_precipitation,
            "snowfall": hourly_snowfall,
            "wind_speed": hourly_wind_speed_10m,
            "weather_code": hourly_weather_code
        }
    )

    weather = weather.groupby("hour", as_index=False).agg(
        temperature=("temperature", "mean"),
        dew_point=("dew_point", "mean"),
        precipitation=("precipitation", "sum"),
        snowfall=("snowfall", "sum"),
        weather_code=("weather_code", "max"),
        wind_speed=("wind_speed", "mean")
        )
    

    return weather



def main():
    start_date, end_date = get_flow_date_range()
    print(f"Downloading historical weather from {start_date} to {end_date}...")

    weather = download_weather(start_date, end_date)

    engine = get_engine()
    with engine.connect() as conn:
        weather.to_sql(OUTPUT_TABLE, con=conn, if_exists="replace",
                    index=False, chunksize=5000)
    engine.dispose()

    print(f"Saved {len(weather):,} rows to PostgreSQL table: {OUTPUT_TABLE}")

if __name__ == "__main__":
    main()