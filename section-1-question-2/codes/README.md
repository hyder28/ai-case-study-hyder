# COE Price Prediction & Quota Elasticity

Predictive model to forecast Singapore COE prices (Categories A & B) and quantify quota elasticity.

## Requirements

```bash
pip install pandas scikit-learn statsmodels
```

## Data Files Required

Place these CSV files in the same directory:
- `MotorVehicleQuotaQuotaPremiumAndPrevailingQuotaPremiumMonthly.csv`
- `COEBiddingResultsPrices.csv`

## Running the Notebook

```bash
jupyter notebook question2.ipynb
```

Or open in any Jupyter-compatible environment (VS Code, JupyterLab, etc.)

## What It Does

1. **Data Preparation**: Merges COE quota and price data
2. **Price Prediction**: Gradient Boosting models for Categories A & B
3. **Elasticity Analysis**: Quantifies price sensitivity to quota changes using log-linear regression

## Key Results

- **Category A**: MAE ~5,669, R² = 0.85
- **Category B**: MAE ~17,156, R² = -0.47
- **Quota Elasticity**: -0.37 (Cat A), -0.51 (Cat B)

