import hydra
import mlflow
import pandas as pd
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    model_uri = f"models:/{cfg.mlflow.registered_model_name}@Production"

    model = mlflow.pyfunc.load_model(model_uri)

    sample = pd.DataFrame(
        [
            {
                "Age": 22,
                "Fare": 7.25,
                "SibSp": 1,
                "Parch": 0,
                "Sex": "male",
                "Embarked": "S",
                "Pclass": 3,
            }
        ]
    )

    prediction = model.predict(sample)

    print("Sample passenger:")
    print(sample)

    print("Prediction:", prediction)

    if prediction[0] == 1:
        print("Result: Survived")
    else:
        print("Result: Did not survive")


if __name__ == "__main__":
    main()