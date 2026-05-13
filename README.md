# MLOps Labs - Titanic Training Pipeline

## Overview

A configurable and reproducible machine learning training pipeline for the Titanic dataset using scikit-learn models, Hydra for configuration management, DVC for pipeline tracking, and MLflow for experiment tracking and model registry.

This project trains multiple models, tracks their performance using MLflow, selects the best model based on accuracy, registers it, assigns it as the Production model, and loads it again for prediction.

> Note: This version uses local MLflow tracking and local MLflow Model Registry. It does not use DagsHub.

---

## Models Used

- Logistic Regression
- Random Forest

---

## Tools Used

- Python
- scikit-learn
- Hydra
- DVC
- MLflow
- Joblib

---

## Project Structure

```text
mlops-lab0/
├── .dvc/
├── configs/
│   └── config.yaml
├── data/
│   ├── train.csv
│   └── train.csv.dvc
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   ├── train.py
│   └── predict.py
├── saved_models/
├── mlruns/
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.md