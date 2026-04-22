import os

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_data
from preprocessor import build_preprocessor, get_features_and_target


def train_model(model, model_name: str, X_train, X_test, y_train, y_test):
    preprocessor = build_preprocessor()

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"{model_name} Accuracy: {acc:.4f}")

    os.makedirs("saved_models", exist_ok=True)
    joblib.dump(pipeline, f"saved_models/{model_name}.pkl")
    print(f"Model saved: saved_models/{model_name}.pkl")


def main():
    df = load_data("data/train.csv")
    X, y = get_features_and_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    for name, model in models.items():
        train_model(model, name, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
