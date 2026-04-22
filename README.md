# MLOps Labs - Titanic Training Pipeline

## Overview
A training pipeline for the Titanic dataset using scikit-learn models, with automated preprocessing and model saving.

## Models Used
- Logistic Regression
- Random Forest

## Project Structure
mlops-labs/
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

## Code Formatting
python -m isort src/
python -m black src/
python -m ruff check src/