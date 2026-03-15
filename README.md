# Credit Risk Modeling

Regression-based models quantifying loan default probabilities, directly informing allowance for doubtful accounts and risk mitigation.

---

## Overview

This project develops and validates regression-based credit risk models to quantify loan default probabilities. The models provide financial institutions with predictive capabilities for loan portfolio risk assessment, enabling data-driven decisions on allowance for doubtful accounts (ALDA) and risk mitigation strategies.

### Problem Statement

Financial institutions face significant credit risk from loan defaults. Accurate prediction of default probabilities is essential for:
- **Provisioning & Reserves** - Determining allowance for doubtful accounts
- **Pricing Strategy** - Setting appropriate interest rates based on risk
- **Portfolio Management** - Identifying high-risk segments
- **Regulatory Compliance** - Meeting capital adequacy requirements

### Solution

Develop and compare multiple regression-based models to identify the strongest predictors of loan default and provide calibrated probability estimates for each loan.

---

## Dataset

**Source:** `Loan_default.csv` (255,349 loan records)

### Features (18 variables)

**Borrower Demographics:**
- Age (years)
- Education level (High School, Bachelor's, Master's, PhD)
- Marital Status (Single, Married, Divorced)
- Employment Type (Full-time, Part-time, Self-employed, Unemployed)
- Months Employed (tenure)

**Financial Profile:**
- Income (annual, USD)
- Debt-to-Income Ratio (DTI)
- Number of Credit Lines (current)
- Credit Score (FICO-like range)

**Loan Characteristics:**
- Loan Amount (USD)
- Loan Term (12, 24, 36, 48, 60 months)
- Interest Rate (%)
- Loan Purpose (Home, Auto, Business, Education, Other)

**Risk Indicators:**
- Has Mortgage (Yes/No)
- Has Dependents (Yes/No)
- Has Co-Signer (Yes/No)

**Target Variable:**
- **Default** (Binary: 0 = No Default, 1 = Default)

---

## Methodology

### 1. Exploratory Data Analysis (EDA)

**Visualization Techniques:**
- Histograms and density plots for numerical variables
- Boxplots to identify outliers and distribution by default status
- Bar charts for categorical variables
- Correlation matrix and heatmaps

**Key Findings:**
- Default rate patterns across demographic segments
- Relationship between credit score and default probability
- Income and DTI ratio as strong predictive features
- Seasonal and employment-type variations

### 2. Feature Engineering & Preprocessing

- Categorical encoding (dummy variables, one-hot encoding)
- Standardization of numerical features
- Handling missing values and outliers
- Variable selection and interaction terms

### 3. Model Development

**Multiple Regression Models:**

1. **Generalized Linear Model (GLM)** - Logistic Regression
   - Baseline model with interpretable coefficients
   - Provides probability estimates directly

2. **Stepwise Regression Model**
   - Forward and backward selection
   - Identifies optimal feature subset
   - Reduces multicollinearity

3. **Lasso Regression (L1 Regularization)**
   - Automatic feature selection
   - Prevents overfitting
   - Sparse coefficient estimates

4. **Ridge Regression (L2 Regularization)**
   - Handles multicollinearity
   - Stable coefficient estimates

### 4. Model Evaluation

**Performance Metrics:**
- **Confusion Matrix** - True Positives, False Positives, True Negatives, False Negatives
- **ROC Curve (Receiver Operating Characteristic)**
  - Plots True Positive Rate vs False Positive Rate
  - Model discrimination ability across thresholds
- **AUC (Area Under Curve)**
  - Probability that model ranks random default higher than random non-default
  - Target: AUC > 0.80 for acceptable model
- **Accuracy, Precision, Recall, F1-Score**
- **Calibration Analysis** - Probability predictions vs actual default rates

---

## Key Results & Insights

### Model Comparison

The analysis reveals strong predictors of default:

**Primary Risk Factors:**
1. **Credit Score** - Inverse relationship; lower scores indicate higher default risk
2. **Debt-to-Income Ratio** - Higher DTI correlates with increased default probability
3. **Interest Rate** - Higher rates reflect underlying risk assessment
4. **Employment Status** - Unemployment increases default risk significantly
5. **Loan Amount & Term** - Longer terms show higher cumulative default probability

**Demographic Insights:**
- Younger borrowers (Age < 25) show higher default rates
- Married individuals with dependents show lower default risk
- Educational attainment inversely correlates with default

### Model Performance

Best-performing model selected based on:
- Highest AUC score
- Balanced sensitivity and specificity
- Interpretability for business stakeholders
- Stability across validation sets

### Business Applications

1. **Allowance for Doubtful Accounts (ALDA)**
   - Calculate expected loss = Probability of Default × Exposure at Default × Loss Given Default
   - Aggregate across loan portfolio for total provision

2. **Pricing Recommendations**
   - Adjust interest rates based on predicted default probability
   - Ensure risk-appropriate compensation

3. **Portfolio Risk Segmentation**
   - Classify loans into risk tiers (Low/Medium/High)
   - Implement targeted risk management strategies

4. **Early Warning System**
   - Monitor loans approaching high-risk probability thresholds
   - Trigger proactive collection or restructuring

---

## Deliverables

### R Scripts
- `loan_default_prediction.R` - Main analysis and model development
- `final_project.R` - Final model selection and validation
- `load_data.R` - Data loading and preprocessing
- `main.R` - Orchestration script

### Python Implementation
- `credit_risk_model.py` - Python replication of regression models

### Report
- `loan_default_prediction_report.pdf` - Comprehensive analysis report

### Data
- `Loan_default.csv` - Full dataset (255,349 records)

---

## Implementation Guide

### Prerequisites
```R
# R packages required
install.packages(c("glmnet", "Metrics", "psych", "ggplot2", 
                   "GGally", "dplyr", "caret", "corrplot", "car"))
```

### Running the Analysis

**Option 1: Full R Workflow**
```bash
# Run complete analysis
Rscript main.R
```

**Option 2: Step-by-Step**
```R
# Load and explore data
source("load_data.R")

# Run main analysis
source("loan_default_prediction.R")

# Generate final model and recommendations
source("final_project.R")
```

### Python Alternative
```python
from credit_risk_model import CreditRiskModel
import pandas as pd

# Load data
df = pd.read_csv('Loan_default.csv')

# Initialize model
model = CreditRiskModel(df)

# Train and validate
model.train_test_split(test_size=0.2, random_state=42)
model.fit_logistic_regression()
model.fit_lasso_regression()
model.fit_ridge_regression()

# Evaluate
results = model.compare_models()
print(results)

# Predict default probabilities for new loans
new_loans = pd.read_csv('new_loans.csv')
probabilities = model.predict_probability(new_loans)
```

---

## Key Outputs

### Model Predictions
- Default probability (0-1) for each loan
- Risk classification (Low/Medium/High)
- Confidence intervals around predictions

### Diagnostic Plots
- ROC curves comparing model performance
- Calibration plots (predicted vs actual probability)
- Feature importance rankings
- Residual diagnostics

### Summary Statistics
- Overall default rate: Dataset-specific
- Model discrimination (AUC)
- Sensitivity and specificity at optimal threshold
- Expected loss calculations

---

## Supporting Figures

**Figure 1: Dataset Overview**
![image](https://github.com/user-attachments/assets/6ccfc996-579c-46f6-807e-c075a660965f)

**Figure 2: Exploratory Data Analysis**
![plot1](https://github.com/user-attachments/assets/c72e453b-d005-416c-85d4-fc6e5883670a)

**Figure 3: Model Performance & ROC Curves**
![image](https://github.com/user-attachments/assets/e981a422-2a8a-456f-8671-a57544a08609)

---

## References & Standards

- **Basel III** - Capital adequacy and credit risk framework
- **IFRS 9** - Expected credit loss model
- **CECL (Current Expected Credit Loss)** - US accounting standard
- Logistic Regression for Credit Risk: Standard industry practice
- Scikit-learn Documentation, Statsmodels
