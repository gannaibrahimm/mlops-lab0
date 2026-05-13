import os

import hydra
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from omegaconf import DictConfig
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_data
from preprocessor import build_preprocessor, get_features_and_target


def setup_mlflow(cfg):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    client = MlflowClient(tracking_uri=cfg.mlflow.tracking_uri)
    return client


def train_model(model, model_name, X_train, X_test, y_train, y_test, output_dir):
    with mlflow.start_run(run_name=model_name) as run:
        preprocessor = build_preprocessor()

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"{model_name} Accuracy: {acc:.4f}")

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_metric("accuracy", acc)

        if model_name == "logistic_regression":
            mlflow.log_param("max_iter", model.max_iter)

        if model_name == "random_forest":
            mlflow.log_param("n_estimators", model.n_estimators)
            mlflow.log_param("random_state", model.random_state)

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
        )

        os.makedirs(output_dir, exist_ok=True)

        model_path = f"{output_dir}/{model_name}.pkl"
        joblib.dump(pipeline, model_path)

        print(f"Model saved: {model_path}")

        return {
            "model_name": model_name,
            "accuracy": acc,
            "run_id": run.info.run_id,
        }


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    client = setup_mlflow(cfg)

    df = load_data(cfg.data.path)
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state,
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

    results = []

    for name, model in models.items():
        result = train_model(
            model,
            name,
            X_train,
            X_test,
            y_train,
            y_test,
            cfg.output.saved_models_dir,
        )
        results.append(result)

    best_model = max(results, key=lambda item: item["accuracy"])

    print("Best model:")
    print(best_model)

    model_uri = f"runs:/{best_model['run_id']}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=cfg.mlflow.registered_model_name,
    )

    client.set_registered_model_alias(
        name=cfg.mlflow.registered_model_name,
        alias="Production",
        version=registered_model.version,
    )

    print(
        f"Best model registered as {cfg.mlflow.registered_model_name} "
        f"version {registered_model.version} and assigned alias Production."
    )


if __name__ == "__main__":
    main()