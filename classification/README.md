# Predictive Maintenance Classification with XAI

## Project Overview

This project builds and explains machine learning models for predicting machine failure using the AI4I 2020 Predictive Maintenance Dataset. The task is binary classification: predict whether `machine_failure` is 0 or 1 based on machine operating conditions.

Because machine failures are rare, the project focuses on metrics that are useful for imbalanced classification, especially PR-AUC, recall, and F2-score. Explainable AI methods are used to understand which engineered machine-condition features drive the model predictions.

## Dataset

- Source: AI4I 2020 Predictive Maintenance Dataset
- Raw file: `data/raw/ai4i2020.csv`
- Target: `machine_failure`
- Original rows: 10,000

The notebook removes detailed failure-mode columns (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) because they are directly connected to the target. Identifier columns are also removed before modelling.

## Feature Engineering

The final model uses engineered features created from the original machine measurements:

- `temp_difference`: process temperature minus air temperature
- `machine_power`: rotational speed multiplied by torque
- `wear_strain`: tool wear multiplied by torque
- `tool_wear_min`: original tool wear variable
- `quality`: product quality type

The processed train, validation, and test splits are saved in:

```text
data/processed/train.csv
data/processed/validation.csv
data/processed/test.csv
```

## Modelling

The notebook compares several classification models with two imbalance strategies:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Class-weight strategy
- SMOTE + ENN strategy

The final model is selected using validation PR-AUC, with validation F2-score as a secondary criterion. The selected model is retrained on train + validation data and evaluated once on the test set.

## XAI Methods

The project includes both global and local explanation methods:

- Permutation feature importance
- SHAP global importance
- SHAP beeswarm plot
- Partial dependence plots
- SHAP waterfall plots for individual predictions
- Simple what-if analysis for selected predicted-failure cases

Generated XAI charts are saved in:

```text
reports/figures/
```

## Project Structure

```text
classification/
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── notebook/
│   └── classification.ipynb
├── reports/
│   └── figures/
├── README.md
└── requirements.txt
```

## How To Run

From the repository root:

```bash
cd classification
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Then open:

```text
notebook/classification.ipynb
```

The notebook can recreate the processed CSV files and save the XAI figures when the relevant cells are run.

## Main Findings

The model shows that failure risk is connected to combined operating stress rather than one simple variable. `machine_power`, `wear_strain`, and `temp_difference` are especially important in the explanations. This suggests that predictive maintenance should monitor combinations of load, tool wear, and thermal behavior rather than only checking tool wear time.
