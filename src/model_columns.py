NUMERIC_FEATURES = ["year","temperature","dew_point","precipitation","snowfall",
                    "wind_speed", "temperature_lag_1hr", "precipitation_lag_1hr",
                    "snowfall_lag_1hr","arrivals_historical_average","departures_historical_average",
                    "is_Holiday"]

CATEGORICAL_FEATURES = ["nta_code","borough","month","hour_of_day","day_of_week","weather_code"]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGETS = ["arrivals", "departures"]