import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from sqlalchemy import text

from database import get_engine

INPUT_TABLE = "neighborhood_hourly_model_data"
OUTPUT_TABLE = "neighborhood_hourly_features"
HISTORICAL_WEEKS = 4


def add_calendar_features(data):
    data = data.copy()
    data["hour"] = pd.to_datetime(data["hour"])
    data["year"] = data["hour"].dt.year
    data["month"] = data["hour"].dt.month
    data["hour_of_day"] = data["hour"].dt.hour
    data["day_of_week"] = data["hour"].dt.dayofweek

    holidays = USFederalHolidayCalendar().holidays(start=data["hour"].min(),
                                                end=data["hour"].max())

    data["is_Holiday"] = data["hour"].isin(holidays)

    return data


def add_weather_lag_features(data):
    weather = data[["hour","temperature","precipitation","snowfall"]].drop_duplicates("hour")
    weather = weather.sort_values("hour",ignore_index=True)

    weather["temperature_lag_1hr"] = weather["temperature"].shift(1)
    weather["precipitation_lag_1hr"] = weather["precipitation"].shift(1)
    weather["snowfall_lag_1hr"] = weather["snowfall"].shift(1)

    lag_columns = ["hour","temperature_lag_1hr",
                "precipitation_lag_1hr","snowfall_lag_1hr"]

    data = data.merge(weather[lag_columns], on="hour", how="left")

    return data


def add_historical_average_features(data, weeks=HISTORICAL_WEEKS):
    data = data.copy()
    data["hour"] = pd.to_datetime(data["hour"])
    data = data.sort_values(["nta_code", "hour"], ignore_index=True)

    for target in ["arrivals", "departures"]:
        grouped = data.groupby("nta_code", sort=False)[target]
        weekly_history = pd.concat(
            [grouped.shift(168 * week) for week in range(1, weeks + 1)],
            axis="columns")
        data[f"{target}_historical_average"] = weekly_history.mean(
            axis="columns",skipna=False)

    required_features = ["arrivals_historical_average","departures_historical_average"]

    data = data.dropna(subset=required_features).reset_index(drop=True)

    return data


def main():
    engine = get_engine()
    with engine.connect() as connection:
        data = pd.read_sql(text(
            f"""SELECT * FROM {INPUT_TABLE}"""),
            con=connection)

    data = add_calendar_features(data)
    data = add_weather_lag_features(data)
    data = add_historical_average_features(data)
    data.to_sql(OUTPUT_TABLE, engine, if_exists="replace",index=False)

    engine.dispose()

    print(f"Saved {len(data):,} rows to PostgreSQL table: {OUTPUT_TABLE}")


if __name__ == "__main__":
    main()