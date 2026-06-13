# Telecom Customer Churn Prediction

A complete end-to-end machine learning project that analyzes customer behavior across three integrated datasets (subscriptions, support complaints, and promotional offers) to identify the key drivers of customer churn and predict which customers are at risk of leaving.

---

## Problem Statement

Customer churn is one of the most expensive problems for subscription-based businesses — acquiring a new customer typically costs far more than retaining an existing one. This project tackles two questions:

1. **Why** are customers leaving? (diagnostic analysis)
2. **Who** is likely to leave next, so the business can intervene proactively? (predictive modeling)

The analysis combines transactional data with operational data (complaints and retention offers) to build a fuller picture of the customer journey — not just demographics and billing, but also service quality and retention efforts.

---

## Datasets

Three datasets were merged on `CustomerID` to enrich the analysis:

| Dataset | Rows | Description |
|---|---|---|
| `Telco_customer_churn.xlsx` | 7,043 | Core customer data — demographics, services subscribed, contract type, billing, and churn status |
| `complaints.csv` | 2,226 | Support tickets — issue type, resolution time, status |
| `offers.csv` | 1,074 | Retention offers sent to customers — offer type, discount value, acceptance |

**Data Cleaning:**
- Removed irrelevant/leakage features: geographic columns (`Lat Long`, `Latitude`, `Longitude`, `Zip Code`, `Country`), and post-outcome fields (`Churn Score`, `CLTV`, `Churn Reason`) that directly encode the churn outcome
- Converted `Total Charges` to numeric and dropped invalid rows
- Verified no missing values across all three datasets
- Checked for outliers in `Monthly Charges` and `Tenure Months` — distributions were normal with no significant outliers

---

## Key Findings (Storytelling)

### 1. Contract type is the single strongest churn signal
55% of customers are on **month-to-month** contracts, and they account for the overwhelming majority of churn:

| Contract | Churn Rate |
|---|---|
| Month-to-month | ~42% |
| One year | ~11% |
| Two year | ~2% |

Customers without a long-term commitment have almost nothing keeping them — and it shows.

### 2. The price difference doesn't justify switching
Month-to-month customers pay **$66.40/month** on average, only **$5.60 more** than two-year contract customers ($60.87/month). This small gap is likely not enough incentive for customers to commit to longer contracts — despite the dramatic difference in churn rates between these groups.

### 3. Payment method reveals a "commitment" pattern
**Electronic check** users churn at the highest rate, while customers on **automatic payments** (bank transfer or credit card) are far more loyal:

| Payment Method | Share of Customers |
|---|---|
| Electronic check | 33.6% |
| Mailed check | 22.8% |
| Bank transfer (automatic) | 21.9% |
| Credit card (automatic) | 21.6% |

Customers who automate their payments appear to be more "set and forget" — and more loyal as a result.

### 4. Fiber optic customers are leaving in large numbers
44% of customers use **Fiber optic** internet, and this group shows the highest churn risk — likely tied to pricing and service quality issues, not the technology itself.

### 5. The highest-risk customer profile
Combining contract, internet service, and payment method reveals one segment that stands out dramatically:

> **Month-to-month + Fiber optic + Electronic check** customers alone account for **42% of all churned customers** — a single combination representing nearly half of total churn.

### 6. Complaints are concentrated — and so is churn
Month-to-month customers generate **68% of all complaints**, despite being only 55% of the customer base. They also wait roughly **9 days** for issue resolution, compared to **~3 days** for two-year contract customers. Poor support experience compounds the lack of contractual commitment.

### 7. Retention offers aren't reaching the right audience effectively
73% of customers who *received* a retention offer are on month-to-month contracts — yet this group still has the highest churn rate. Offers are being targeted at the right segment, but **aren't converting** — suggesting the offer type or value may need rethinking.

### 8. Revenue impact of churn
Churned customers paid **~$1,500** in total charges on average, versus **~$2,500** for retained customers — confirming that early churn represents significant lost lifetime value per customer.

---

## Business Recommendations

1. **Target the highest-risk segment directly**: Month-to-month + Fiber optic + Electronic check customers should be the #1 priority for retention campaigns — they represent 42% of all churn from a single combination.
2. **Incentivize contract upgrades more aggressively**: The current $5.60/month price gap between month-to-month and two-year contracts is too small to drive switching. Larger discounts for annual commitments could meaningfully shift this.
3. **Encourage automatic payment adoption**: Since automatic payment users churn far less, consider incentives (small discounts, loyalty points) for switching from electronic/mailed checks to automatic payments.
4. **Prioritize support for month-to-month customers**: Reducing their resolution time from ~9 days closer to the ~3 days seen with loyal customers could directly reduce one of their key churn drivers.
5. **Re-evaluate retention offer strategy**: Offers are reaching the right (high-risk) customers but aren't reducing churn in that segment — test different offer types, values, or timing (e.g., proactive offers before complaints occur, not after).
6. **Investigate Fiber optic service quality/pricing**: Given its size (44% of customers) and high churn, even small improvements here would have an outsized impact on overall churn.

---

## Modeling Approach

### Feature Selection
Based on the EDA findings above, 8 features were selected:

- **Tenure Months**, **Contract** — strongest churn predictors (month-to-month + low tenure = high risk)
- **Monthly Charges**, **Payment Method** — linked to churn rate (electronic check = high risk)
- **Internet Service** — fiber optic = high churn risk
- **Online Security**, **Online Backup**, **Tech Support** — service quality indicators

### Preprocessing
- **Target encoding**: `Churn Label` (Yes/No) → binary (1/0) via `LabelEncoder`
- **Train/Validation/Test split**: 80% / 10% / 10% (5,625 / 704 / 703 customers)
- **Categorical features**: One-Hot Encoded (fit on train only, to avoid data leakage)
- **Numerical features** (`Monthly Charges`, `Tenure Months`): Standardized via `StandardScaler` (fit on train only)
- **Class imbalance**: The dataset is imbalanced (73% No churn vs. 27% Yes churn), so all models were trained with `class_weight='balanced'`, and **Recall** was prioritized as the key metric — missing an actual churner (false negative) is more costly to the business than a false alarm (false positive)

### Models Trained
Three models were trained and tuned via `GridSearchCV` (5-fold cross-validation, optimized for Recall):

| Model | Notes |
|---|---|
| Logistic Regression | Baseline model |
| Decision Tree | Tuned `max_depth` ∈ [3-8] |
| Random Forest | Tuned `n_estimators`, `max_depth`, `max_features` |

---

## Results

### Performance Comparison

| Model | Recall (Train) | Recall (Val) | Recall (Test) | F1-Score (Test) |
|---|---|---|---|---|
| Logistic Regression | 0.801 | 0.810 | 0.735 | 0.599 |
| Decision Tree | 0.823 | 0.822 | 0.746 | 0.605 |
| **Random Forest** | **0.822** | **0.833** | **0.757** | 0.591 |

**Best Random Forest parameters**: `max_depth=3`, `max_features=0.5`, `n_estimators=500`, `class_weight='balanced'`

### Model Selection: Random Forest

Random Forest was selected as the final model, achieving the **highest Test Recall (75.7%)** — meaning it correctly identifies roughly **3 out of every 4 customers who will actually churn**. All three models showed consistent performance across Train/Validation/Test sets (no significant overfitting), and ROC-AUC scores were close (~0.81–0.82) across all models, confirming strong and stable discriminative ability regardless of decision threshold.

### Feature Importance

| Rank | Feature | Importance |
|---|---|---|
| 1 | Contract_Month-to-month | ~0.44 |
| 2 | Online Security_No | ~0.15 |
| 3 | Contract_Two year | ~0.10 |
| 4 | Tech Support_No | ~0.095 |
| 5 | Tenure Months | ~0.07 |
| 6 | Internet Service_Fiber optic | ~0.05 |
| 7 | Monthly Charges | ~0.03 |
| 8 | Contract_One year | ~0.025 |
| 9 | Payment Method_Electronic check | ~0.01 |
| 10 | Internet Service_DSL | ~0.01 |

**Contract type (especially Month-to-month) is by far the strongest predictor — nearly 3x more important than any other feature**, strongly validating the EDA finding that month-to-month customers are the highest churn risk segment.

The next strongest signals are the **absence of add-on services** — customers without Online Security or Tech Support churn significantly more, suggesting these services act as "stickiness" factors that increase switching costs and customer satisfaction.

Interestingly, **Tenure Months ranks lower than expected** (5th) given how prominently it featured in the EDA. This suggests that Contract type already captures most of the same underlying signal — month-to-month customers tend to have low tenure, so the model relies more heavily on the contract feature itself.

This alignment between exploratory analysis and model-driven feature importance — along with the new insight around add-on services — strengthens confidence that the model is learning genuine business patterns, not noise.

---

## Tech Stack

- **Data manipulation**: pandas
- **Visualization**: matplotlib, seaborn
- **Machine learning**: scikit-learn (Logistic Regression, Decision Tree, Random Forest, GridSearchCV)
- **Evaluation**: Recall, F1-Score, Confusion Matrix, ROC-AUC
- **Model persistence**: joblib

---

## Project Structure

```
├── Telco_customer_churn.xlsx     # Core customer dataset
├── complaints.csv                 # Support ticket data
├── offers.csv                     # Retention offer data
├── churn_prediction.ipynb         # Full analysis notebook (EDA → Modeling → Evaluation)
├── churn_model.pkl                # Trained Random Forest model
├── onehot_encoder.pkl             # Fitted OneHotEncoder for categorical features
├── scaler.pkl                     # Fitted StandardScaler for numerical features
├── label_encoder.pkl              # Fitted LabelEncoder for target variable
└── README.md
```

---

## How to Use the Saved Model

```python
import joblib
import pandas as pd
import numpy as np

# Load model and preprocessing objects
model = joblib.load('churn_model.pkl')
encoder = joblib.load('onehot_encoder.pkl')
scaler = joblib.load('scaler.pkl')

# New customer data (raw)
new_customer = pd.DataFrame({
    'Contract': ['Month-to-month'],
    'Payment Method': ['Electronic check'],
    'Internet Service': ['Fiber optic'],
    'Online Security': ['No'],
    'Online Backup': ['No'],
    'Tech Support': ['No'],
    'Monthly Charges': [70.5],
    'Tenure Months': [5]
})

# Apply the same preprocessing used during training
categorical_cols = ['Contract', 'Payment Method', 'Internet Service',
                     'Online Security', 'Online Backup', 'Tech Support']
encoded = encoder.transform(new_customer[categorical_cols])

numerical_cols = ['Monthly Charges', 'Tenure Months']
scaled = scaler.transform(new_customer[numerical_cols])

final_input = np.concatenate([encoded, scaled], axis=1)

# Predict
prediction = model.predict(final_input)
print("Churn" if prediction[0] == 1 else "No Churn")
```

---

## Future Improvements

- Deploy the model via a **FastAPI** endpoint for real-time predictions
- Build a simple dashboard (Streamlit) for business users to explore at-risk customers
- Experiment with additional features (e.g., service bundling, tenure-to-charges ratio)
- A/B test the recommended retention strategies against a control group to measure actual impact