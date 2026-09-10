import pandas as pd
NUMERIC_FEATURES = ["year","temperature","dew_point","precipitation","snowfall",
                    "wind_speed", "temperature_lag_1hr", "precipitation_lag_1hr",
                    "snowfall_lag_1hr","arrivals_historical_average","departures_historical_average",
                    "is_Holiday"]

CATEGORICAL_FEATURES = ["nta_code","borough","month","hour_of_day","day_of_week","weather_code"]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = ["arrivals", "departures"]

WEATHER_CODES = [
    0, 1, 2, 3, 45, 48, 51, 53, 55, 56, 57,
    61, 63, 65, 66, 67, 71, 73, 75, 77,
    80, 81, 82, 85, 86, 95, 96, 99,
]

def apply_categorical_dtypes(data):
    data = data.copy()

    for column in CATEGORICAL_FEATURES:
        if column != "weather_code":
            data[column] = data[column].astype(str).astype("category")

    codes = pd.to_numeric(
        data["weather_code"], errors="coerce"
    ).round().astype("Int64")

    data["weather_code"] = pd.Categorical(
        codes,
        categories=WEATHER_CODES,
    )

    return data