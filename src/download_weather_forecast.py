from pathlib import Path
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry
from database import get_engine

PROJECT_DIR = Path(__file__).parent.parent
OUTPUT_TABLE = "hourly_weather_forecast"

FORECAST_DAYS = 7

def download_weather():
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
	"latitude": 40.71,
	"longitude": -74,
	"hourly": ["temperature_2m", "dew_point_2m", "precipitation", "snowfall", "wind_speed_10m", "weather_code"],
	"timezone": "America/New_York",
    "forecast_hours": FORECAST_DAYS * 24,
    "past_hours":1
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
    
    weather["forecast_generated_at"] = pd.Timestamp.now(tz="UTC")

    return weather


def main():
    weather = download_weather()

    engine = get_engine()
    weather.to_sql(
        OUTPUT_TABLE,
        engine,
        if_exists="replace",
        index=False
        )

    engine.dispose()

    print(f"Saved {len(weather):,} rows to PostgreSQL table: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()
