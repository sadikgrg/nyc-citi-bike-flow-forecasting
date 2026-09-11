# NYC Citi Bike Flow Forecasting
**[Open the live dashboard](https://nyc-citi-bike-flow-forecast.streamlit.app/)**

#### Dashboard Demo
[nyc-citi-bike-forecast-dashboard.webm](https://github.com/user-attachments/assets/3333f37e-0079-4b56-800f-82292da86e5c)


This project predicts hourly Citi Bike arrivals, departures, and net bike flow
for New York City neighborhoods. It uses trip history, NYC neighborhood
boundaries, Open-Meteo weather data, PostgreSQL, and XGBoost.

## Project structure

```text
data/raw/       Citi Bike trip CSV files
models/         Trained XGBoost models
notebooks/      Exploratory analysis
src/            Data pipeline, modeling, forecasting, and dashboard scripts
```

### Exploratory notebooks

The notebooks follow the project workflow:

1. `01_trip_data_exploration.ipynb` — inspect raw Citi Bike trips and missing station data
2. `02_station_neighborhood_validation.ipynb` — inspect NTA boundaries and unmatched stations
3. `03_weather_data_exploration.ipynb` — explore historical and forecast Open-Meteo data
4. `04_flow_weather_data_merge.ipynb` — combine neighborhood flow with hourly weather
5. `05_feature_engineering.ipynb` — inspect calendar, lag, and historical-average features
6. `06_model_training_evaluation.ipynb` — train and evaluate the XGBoost arrival and departure models
7. `07_forecast_visualizations.ipynb` — prototype the dashboard map and supporting charts

## Setup

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The code uses the standard PostgreSQL environment variables. `PGUSER` defaults
to your computer username, `PGHOST` to `localhost`, `PGPORT` to `5432`, and
`PGDATABASE` to `citibike`.

```bash
export PGUSER="your_postgres_user"
export PGPASSWORD="your_postgres_password"
export PGDATABASE="citibike"
```


## Build the training data

Run these scripts in order:

```bash
python src/build_stations.py
python src/assign_station_neighborhoods.py
python src/build_hourly_flow.py
python src/download_historical_weather.py
python src/merge_training_data.py
python src/build_features.py
python src/train_models.py
```

## Generate a forecast

```bash
python src/download_weather_forecast.py
python src/generate_forecast.py
```

## Open the dashboard

```bash
streamlit run src/dashboard.py
```
