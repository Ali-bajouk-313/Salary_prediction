from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import METADATA_PATH, METRICS_PATH, MODEL_PATH
from src.preprocess import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, prepare_training_data


def train_model(random_state: int = 42) -> dict:
    prepared = prepare_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        prepared.features,
        prepared.target,
        test_size=0.2,
        random_state=random_state,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('categorical', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_COLUMNS),
            ('numeric', 'passthrough', NUMERIC_COLUMNS),
        ]
    )

    model = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            (
                'regressor',
                RandomForestRegressor(
                    n_estimators=300,
                    max_depth=20,
                    min_samples_leaf=2,
                    n_jobs=1,
                    random_state=random_state,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    metrics = {
        'mae': float(mean_absolute_error(y_test, preds)),
        'rmse': float(mean_squared_error(y_test, preds) ** 0.5),
        'r2': float(r2_score(y_test, preds)),
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    Path(METADATA_PATH).write_text(json.dumps(prepared.metadata, indent=2), encoding='utf-8')
    Path(METRICS_PATH).write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics


if __name__ == '__main__':
    print(json.dumps(train_model(), indent=2))
