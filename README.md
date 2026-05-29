
# Insurance Premium Prediction & Predictive Maintenance Classification with Explainable AI (XAI)

## Project Overview

This project applies machine learning and Explainable AI (XAI) techniques to two distinct problems:

1. **Regression**: Predicting health insurance charges and analyzing the fairness of premium pricing decisions.
2. **Classification**: Predicting machine failure for predictive maintenance using the AI4I 2020 dataset.

Both pipelines emphasize the use of XAI methods to interpret black-box model predictions and communicate actionable insights to non-technical stakeholders.

**Key Business Questions:**
- *Regression*: "Is the AI model making fair insurance pricing decisions? Are premiums driven by actual health risk factors, or by uncontrollable characteristics like gender and region?"
- *Classification*: "Which machine operating conditions are the strongest predictors of failure, and how can maintenance teams use these insights to prevent downtime?"

---

## Presentation Material

[Canva Presentation Link](https://canva.link/6vt385ti6a06ewl)

---

## Repository Structure

```
XAI_Capstone_Hayeong_Pawel/
│
├── classification/                    
│   ├── data/
│   │   ├── raw/
│   │   │   └── ai4i2020.csv
│   │   └── processed/
│   │       ├── train.csv
│   │       ├── validation.csv
│   │       └── test.csv
│   ├── notebook/
│   │   └── classification.ipynb
│   ├── reports/
│   │   └── figures/
│   ├── README.md
│   └── requirements.txt
│
├── data/                              
│   └── insurance.csv
├── models/                            
│   ├── xgboost_model.joblib
│   └── feature_names.joblib
├── notebooks/                     
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_xai.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Datasets

| Pipeline | Source | Size | Target Variable |
| :--- | :--- | :--- | :--- |
| Regression | Medical Cost Personal Dataset | 1,338 rows, 7 features | `charges` (USD) |
| Classification | AI4I 2020 Predictive Maintenance | 10,000 rows, engineered features | `machine_failure` (0/1) |

**Regression features:** age, sex, bmi, children, smoker, region

**Classification engineered features:** temp_difference, machine_power, wear_strain, tool_wear_min, quality

---

## Models

### Regression (Insurance Premium Prediction)
- **Algorithm:** XGBoost Regressor
- **Target transformation:** log(charges)
- **R² Score:** 0.848
- **MAE:** $2,402
- **RMSE:** $4,858

### Classification (Predictive Maintenance)
- **Algorithms compared:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost
- **Imbalance strategies:** Class-weight, SMOTE + ENN
- **Final model selected by:** Validation PR-AUC with F2-score as secondary criterion

---

## XAI Methods Applied

### In-class Methods:
- **SHAP** (bar, beeswarm, waterfall) - Global and local feature importance
- **LIME** - Local approximation of black-box model
- **PDP** (Partial Dependence Plots) - Feature effect on predictions
- **Permutation Importance** - Feature importance via permutation

### Bonus Method (not covered in class):
- **Counterfactual Explanation** - What-if scenario analysis (e.g., "What would a smoker need to change to reduce their premium?")

---

## Key Findings

### Regression
1. Smoking is the most dominant factor, raising premiums by 3.8x
2. Age and BMI are significant health-based risk factors
3. Sex and Region have near-zero impact on premiums
4. A smoker could save up to $14,681/year by quitting smoking

### Classification
- Engineered features (machine_power, wear_strain) are the strongest predictors of machine failure
- XAI methods reveal which operating conditions push the model toward predicting failure

---

## How to Run

```bash
git clone https://github.com/hjae-cmyk/XAI_Capstone_Hayeong_Pawel.git
cd XAI_Capstone_Hayeong_Pawel

# For Regression pipeline
python -m venv venv
source venv/Scripts/activate    # Windows
pip install -r requirements.txt
jupyter lab

# For Classification pipeline
cd classification
pip install -r requirements.txt
jupyter lab
```

---

## Team Work Distribution

| Team Member | Primary Role | Specific Deliverables |
| :--- | :--- | :--- |
| **Hayeong Jae** | Regression & XAI Lead | EDA with visualizations (`01_eda.ipynb`), XGBoost Regressor development and tuning (`02_modeling.ipynb`), Full XAI analysis: SHAP, LIME, PDP, Permutation Importance, Counterfactual Explanations (`03_xai.ipynb`), Repository setup and structure |
| **Paweł Kazimiruk** | Classification & XAI Lead | Classification pipeline development (`classification.ipynb`), Model comparison (LR, RF, XGBoost, LightGBM, CatBoost) with imbalance handling, XAI analysis for classification: Permutation Importance, SHAP, PDP, Waterfall, What-if analysis, Presentation preparation |

Both team members explained their respective findings to each other to ensure mutual understanding and a cohesive final presentation.

---

## Use of AI Tools

In compliance with the course guidelines, we transparently disclose our use of AI tools during this project:

- **GitHub Copilot**: Used for code debugging and error message interpretation, suggesting fixes for library-specific issues.
- **Claude (Anthropic)**: Used for Git Bash command corrections and README structure formatting.

*All core modeling decisions, XAI interpretations, and business insights were conceptualized and validated entirely by the team members.*

---
