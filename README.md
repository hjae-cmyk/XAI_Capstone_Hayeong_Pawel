cat > README.md << 'EOF'
# Insurance Premium Prediction with Explainable AI (XAI)

## Project Overview
This project applies machine learning and Explainable AI (XAI) techniques to predict health insurance charges and analyze the fairness of premium pricing decisions.

Key Business Question:
"Is the AI model making fair insurance pricing decisions? Are premiums driven by actual health risk factors, or by uncontrollable characteristics like gender and region?"
---

## Presentation Material Canva Link
https://canva.link/6vt385ti6a06ewl
---

## Repository Structure

Capstone/
├── data/
│   └── insurance.csv          
├── models/
│   ├── xgboost_model.joblib   
│   └── feature_names.joblib   
├── notebooks/
│   ├── 01_eda.ipynb           
│   ├── 02_modeling.ipynb      
│   └── 03_xai.ipynb           
├── .gitignore
├── requirements.txt
└── README.md

---

## Dataset
- Source: Medical Cost Personal Dataset
- Size: 1,338 records, 7 features
- Target: charges (insurance premium in USD)

Features: age, sex, bmi, children, smoker, region, charges

---

## Model
- Algorithm: XGBoost Regressor
- Target transformation: log(charges)
- R2 Score: 0.848
- MAE: $2,402
- RMSE: $4,858

---

## XAI Methods Applied

In-class Methods:
- SHAP (bar, beeswarm, waterfall) - Global and local feature importance
- LIME - Local approximation of black-box model
- PDP - Feature effect on predictions
- Permutation Importance - Feature importance via permutation

Bonus Method:
- Counterfactual Explanation - What-if scenario analysis

---

## Key Findings
1. Smoking is the most dominant factor, raising premiums by 3.8x
2. Age and BMI are significant health-based risk factors
3. Sex and Region have near-zero impact on premiums
4. A smoker could save up to $14,681/year by quitting smoking

---

## How to Run

git clone https://github.com/hjae-cmyk/XAI_Capstone_Hayeong_Pawel.git
cd XAI_Capstone_Hayeong_Pawel
python -m venv venv
source venv/Scripts/activate
pip install -r requirements.txt
jupyter lab


---

## Team Work Distribution

### Hayeong Jae
- Regression modeling (XGBoost, hyperparameter tuning)
- Full XAI analysis (SHAP, LIME, PDP, Permutation Importance, Counterfactual)
- Repository setup and structure
- Explained regression findings and XAI results to Paweł

### Paweł
- Classification modeling and XAI analysis
- Full XAI analysis (SHAP, LIME, PDP, Permutation Importance, Counterfactual)
- Explained classification findings to Hayeong

---

## AI Tools Usage

The following AI tools were used as assistants during this project:
- **GitHub Copilot**: used for code debugging, error message interpretation and suggesting fixes for issues.
- **Claude (Anthropic)**: used for Git Bash command corrections and README structure

---
## References
- Lundberg et al. (2017) - SHAP
- Ribeiro et al. (2016) - LIME
- Friedman (2001) - PDP
- Wachter et al. (2017) - Counterfactual Explanations
EOF
