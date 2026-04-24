import os

import hydra
import joblib
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_data
from preprocessor import build_preprocessor, get_features_and_target


def train_model(
    model, model_name: str, X_train, X_test, y_train, y_test, output_dir: str
):
    preprocessor = build_preprocessor()
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(pipeline, f"{output_dir}/{model_name}.pkl")
    print(f"Model saved: {output_dir}/{model_name}.pkl")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    df = load_data(cfg.data.path)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.data.test_size, random_state=cfg.data.random_state
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=cfg.models.logistic_regression.max_iter
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=cfg.models.random_forest.n_estimators,
            random_state=cfg.models.random_forest.random_state,
        ),
    }

    for name, model in models.items():
        train_model(
            model, name, X_train, X_test, y_train, y_test, cfg.output.saved_models_dir
        )


if __name__ == "__main__":
    main()
