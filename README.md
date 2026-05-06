# MLOps Labs - Titanic Training Pipeline

## Overview
A configurable and reproducible training pipeline for the Titanic dataset using scikit-learn models, Hydra for configuration management, and DVC for data versioning and pipeline tracking.

## Models Used
- Logistic Regression
- Random Forest

## Project Structure
mlops-lab0/
├── .dvc/
├── configs/
│   └── config.yaml
├── data/
│   └── train.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   └── train.py
├── saved_models/
├── dvc.yaml
├── dvc.lock
├── requirements.txt
└── README.md

## How to Run

Install dependencies:
pip install -r requirements.txt

Pull data and models from remote storage:
dvc pull

Train the models:
python src/train.py

Run the full DVC pipeline:
dvc repro

## Configuration
All pipeline settings are managed in configs/config.yaml
You can change data path, test size, and model parameters without touching the code.

## DVC Remote Storage
Data and models are stored on DagsHub.
To push updates to remote storage:
dvc push

## Code Formatting
python -m isort src/
python -m black src/
python -m ruff check src/