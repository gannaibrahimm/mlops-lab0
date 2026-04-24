# MLOps Labs - Titanic Training Pipeline

## Overview
A configurable training pipeline for the Titanic dataset using scikit-learn models and Hydra for configuration management.

## Models Used
- Logistic Regression
- Random Forest

## Project Structure
mlops-labs/
├── configs/
│   └── config.yaml
├── data/
│   └── train.csv
├── src/
│   ├── data_loader.py
│   ├── preprocessor.py
│   └── train.py
├── saved_models/
├── requirements.txt
└── README.md

## How to Run

Install dependencies:
pip install -r requirements.txt

Train the models:
python src/train.py

## Configuration
All pipeline settings are managed in configs/config.yaml
You can change data path, test size, and model parameters without touching the code.

## Code Formatting
python -m isort src/
python -m black src/
python -m ruff check src/