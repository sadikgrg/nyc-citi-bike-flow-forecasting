from pathlib import Path
import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sqlalchemy import text
from xgboost import XGBRegressor

from database import get_engine
from model_columns import CATEGORICAL_FEATURES, MODEL_FEATURES, TARGETS, apply_categorical_dtypes


PROJECT_DIR = Path(__file__).parent.parent
MODEL_DIR = PROJECT_DIR / "models"
HISTORY_TABLE = "neighborhood_hourly_model_data"
WEATHER_TABLE = "hourly_weather_forecast"
FEATURE_TABLE = "neighborhood_hourly_features"
OUTPUT_TABLE = "hourly_neighborhood_forecast"
FORECAST_HOURS = 7 * 24
RECENT_WEEKS = [1, 2, 3, 4]
LAST_YEAR_WEEKS = [51, 52, 53, 54]


def load_inputs():
    engine = get_engine()
    with engine.connect() as connection:
        weather = pd.read_sql(text(
            f"""Select * From {WEATHER_TABLE} Order by hour"""
        ), con=connection)

        neighborhoods = pd.read_sql(text(
            f"""Select Distinct nta_code, nta_name, borough
            From {FEATURE_TABLE}
            Order By nta_code"""
        ), con=connection)

        history_range = pd.read_sql(text(
            f"""Select MIN(hour) as first_hour, MAX(hour) as last_hour
            From {HISTORY_TABLE}"""
        ), con=connection).iloc[0]

    engine.dispose()

    return weather, neighborhoods, history_range


def prepare_weather(weather):
    weather = weather.copy()
    weather["hour"] = pd.to_datetime(weather["hour"])
    weather = weather.drop_duplicates("hour").sort_values("hour", ignore_index = True)

    weather["temperature_lag_1hr"] = weather["temperature"].shift(1)
    weather["precipitation_lag_1hr"] = weather["precipitation"].shift(1)
    weather["snowfall_lag_1hr"] = weather["snowfall"].shift(1)

    weather = weather.dropna(subset=["temperature_lag_1hr","precipitation_lag_1hr","snowfall_lag_1hr"])

    return weather


def choose_history_weeks(weather, history_range):
    first_history = pd.Timestamp(history_range["first_hour"])
    last_history = pd.Timestamp(history_range["last_hour"])
    first_forecast = weather["hour"].min()
    last_forecast = weather["hour"].max()

    recent_start = first_forecast - pd.Timedelta(weeks=max(RECENT_WEEKS))
    recent_end = last_forecast - pd.Timedelta(weeks=min(RECENT_WEEKS))

    if first_history <= recent_start and last_history >= recent_end:
        print("Using the recent four-week average.")
        return RECENT_WEEKS
    
    last_year_start = first_forecast - pd.Timedelta(weeks=max(LAST_YEAR_WEEKS))
    last_year_end = last_forecast - pd.Timedelta(weeks=min(LAST_YEAR_WEEKS))

    if first_history <= last_year_start and last_history >= last_year_end:
        print("Recent demand is unavailable; using last year's four-week average.")
        return LAST_YEAR_WEEKS


def load_history(weather, weeks):

    start_hour = weather["hour"].min() - pd.Timedelta(weeks=max(weeks))
    end_hour = weather["hour"].max() - pd.Timedelta(weeks=min(weeks))

    engine = get_engine()

    with engine.connect() as connection:
        history = pd.read_sql(text(
            f"""Select hour, nta_code,arrivals,departures
            From {HISTORY_TABLE}
            Where hour>=:start_hour And hour<=:end_hour"""
        ), con=connection,params={"start_hour": start_hour, "end_hour": end_hour})

    engine.dispose()

    history["hour"] = pd.to_datetime(history["hour"])
    history["nta_code"] = history["nta_code"].astype("str")

    return history


def build_forecast_features(weather,neighborhoods, history, weeks):
    neighborhoods = neighborhoods.copy()
    neighborhoods["nta_code"] = neighborhoods["nta_code"].astype("str")

    grid = pd.MultiIndex.from_product([weather["hour"],neighborhoods["nta_code"]],
                                    names=["hour","nta_code"]).to_frame(index=False)

    grid = grid.merge(neighborhoods,on="nta_code",how="left")
    grid = grid.merge(weather, on="hour", how="left")

    grid["year"] = grid["hour"].dt.year
    grid["month"] = grid["hour"].dt.month
    grid["hour_of_day"] = grid["hour"].dt.hour
    grid["day_of_week"] = grid["hour"].dt.dayofweek

    holidays = USFederalHolidayCalendar().holidays(start=grid["hour"].min(),
                                                end=grid["hour"].max())

    grid["is_Holiday"] = grid["hour"].isin(holidays)

    for target in TARGETS:
        week_columns = []
        for week in weeks:
            column = f"{target}_week_{week}"
            week_columns.append(column)
            weekly_history = history[["hour", "nta_code", target]].copy()
            weekly_history["hour"] += pd.Timedelta(weeks=week)
            weekly_history = weekly_history.rename(columns={target: column})
            grid = grid.merge(weekly_history,on=["hour", "nta_code"],how="left")

        grid[f"{target}_historical_average"] = grid[week_columns].mean(axis=1)
        grid = grid.drop(columns=week_columns)

        grid = apply_categorical_dtypes(grid)    
        grid = grid.sort_values(["hour", "nta_code"], ignore_index=True)

    return grid


def make_predictions(features):
    forecast = features[["hour", "nta_code", "nta_name", "borough"]].copy()

    for target in TARGETS:
        model_path = MODEL_DIR / f"xgboost_{target}.json"
        model = XGBRegressor(enable_categorical=True)
        model.load_model(model_path)
        forecast[f"predicted_{target}"] = model.predict(features[MODEL_FEATURES])

    forecast["predicted_net_flow"] = forecast["predicted_arrivals"]-forecast["predicted_departures"]

    forecast.insert(0, "forecast_generated_at", pd.Timestamp.now(tz="UTC"))

    return forecast


def main():
    print("Loading forecast inputs...")
    weather, neighborhoods, history_range = load_inputs()
    weather = prepare_weather(weather)
    weeks = choose_history_weeks(weather, history_range)
    history = load_history(weather, weeks)

    print("Building forecast features...")
    features = build_forecast_features(weather, neighborhoods, history, weeks)
    print("Making predictions...")
    forecast = make_predictions(features)

    engine =get_engine()
    forecast.to_sql(OUTPUT_TABLE, engine, if_exists="replace", index=False)
    engine.dispose()

    print(f"Saved {len(forecast):,} rows to PostgreSQL table: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()