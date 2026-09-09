from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy import text
from xgboost import XGBRegressor
from database import get_engine

from model_columns import NUMERIC_FEATURES, CATEGORICAL_FEATURES, MODEL_FEATURES, TARGETS


PROJECT_DIR = Path(__file__).parent.parent
MODEL_DIR = PROJECT_DIR / "models"
FEATURE_TABLE = "neighborhood_hourly_features"
FORECAST_DAYS = 7
BACKTEST_FOLDS = 4


def load_data():
    engine = get_engine()
    with engine.connect() as connection:
        data = pd.read_sql(text(
            f"""Select *
            From {FEATURE_TABLE}"""
        ), con=connection)
    engine.dispose()

    return data


def create_model():
    model = XGBRegressor(
        objective="count:poisson",
        enable_categorical =True,
        n_estimators = 500,
        tree_method = "hist",
        random_state =67
    )
    return model

def calculate_metrics(actual, predicted):
    metrics ={
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted))
    }

    return metrics


def build_evaluation_windows(data):
    horizon = pd.Timedelta(days=FORECAST_DAYS)
    data_end = data["hour"].max() + pd.Timedelta(hours=1)
    final_start = data_end - horizon

    development_windows = []

    for fold in range(BACKTEST_FOLDS):
        weeks_before_final = BACKTEST_FOLDS - fold
        start = final_start - weeks_before_final * horizon
        development_windows.append((start, start + horizon))

    return development_windows, (final_start, data_end)


def metric_row(stage,fold,start,end,target,model_name,scores):
    row = {
        "stage": stage,
        "fold": fold,
        "validation_start": start,
        "validation_end": end,
        "target": target,
        "model": model_name,
        "mae":scores["mae"],
        "rmse":scores["rmse"]
    }
    return row


def evaluate_window(data,start,end,stage,fold):
    train = data.loc[data["hour"]<start]
    validation = data.loc[(data["hour"]>=start) & (data["hour"]< end)]

    predictions = {}
    baselines= {}
    rows= []

    for target in TARGETS:
        model = create_model()
        model.fit(train[MODEL_FEATURES], train[target])

        predictions[target] = model.predict(validation[MODEL_FEATURES])
        baselines[target] = validation[f"{target}_historical_average"]

        model_scores = calculate_metrics(validation[target], predictions[target])
        baseline_scores = calculate_metrics(validation[target], baselines[target])

        rows.extend([metric_row(stage,fold,start,end,target,"XGBoost", model_scores),
                    metric_row(stage,fold,start,end,target,"Historical avg", baseline_scores)])

        print(
            f"{target.title()}\n"
            f"XGBoost MAE: {model_scores['mae']:.3f}   "
            f"Baseline MAE: {baseline_scores['mae']:.3f}"
        )

    actual_net_flow = validation["net_flow"]
    predicted_net_flow = predictions["arrivals"]- predictions["departures"]
    baseline_net_flow = baselines["arrivals"]- baselines["departures"]
    model_scores = calculate_metrics(actual_net_flow, predicted_net_flow)
    baseline_scores = calculate_metrics(actual_net_flow, baseline_net_flow)

    rows.extend(
        [metric_row(stage,fold,start,end,"net_flow","XGBoost",model_scores),
        metric_row(stage,fold,start,end,"net_flow","Historical average",baseline_scores)
        ])
    
    print(
        "Net flow\n"
        f"XGBoost MAE: {model_scores['mae']:.3f}   "
        f"Baseline MAE: {baseline_scores['mae']:.3f}\n"
    )

    return rows


def print_development_summary(results):
    summary = (results.loc[results["stage"]=="development"]
            .groupby(["target", "model"], as_index=False)
            .agg(mean_mae = ("mae","mean"), mean_rmse=("rmse","mean"))
            .sort_values(["target", "model"]))
    
    print(f"\n{BACKTEST_FOLDS}-fold development summary:")
    print(summary.round(3).to_string(index=False))


def train_and_save_models(data):
    for target in TARGETS:
        print(f"\nTraining final {target} model")
        model = create_model()
        model.fit(data[MODEL_FEATURES], data[target])
        model_path = MODEL_DIR / f"xgboost_{target}.json"
        model.save_model(model_path)
        print(f"Saved model: {model_path}")


def main():
    print(f"Loading data from PostgreSQL table: {FEATURE_TABLE}")
    data = load_data()
    data["hour"] = pd.to_datetime(data["hour"])
    data[CATEGORICAL_FEATURES] =data[CATEGORICAL_FEATURES].astype(str).astype("category")
    
    print(f"Loaded {len(data):,} rows")

    development_windows, final_window = build_evaluation_windows(data)
    result_rows = []

    print(f"\nRunning {BACKTEST_FOLDS} rolling development folds")

    for fold, (start, end) in enumerate(development_windows, start=1):
        result_rows.extend(
            evaluate_window(data, start, end, "development", fold)
        )

    results = pd.DataFrame(result_rows)
    print_development_summary(results)

    print("\nRunning the untouched final seven-day test")
    result_rows.extend(
        evaluate_window(data, *final_window, stage="final_test", fold=1)
    )

    results = pd.DataFrame(result_rows)
    final_results = results.loc[
        results["stage"] == "final_test",
        ["target", "model", "mae", "rmse"],
    ]

    print("\nFinal seven-day results:")
    print(final_results.round(3).to_string(index=False))

    print("\nTraining and saving final models")
    train_and_save_models(data)


if __name__ == "__main__":
    main()
