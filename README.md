# Stock Price Prediction — AAPL (Course Project, Machine Learning)

## Overview
This project predicts the next-day closing price of Apple Inc. (AAPL) stock using
real historical daily OHLCV data (2015-01-02 to 2025-12-31, 2,766 trading days).

It compares three modeling approaches:
1. **Classical time series** — ARIMA
2. **Feature-based ML** — Linear Regression, Decision Tree, Random Forest, Gradient Boosting
   (trained on engineered technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, lags, volatility)
3. **Deep learning** — LSTM (sequence-based, no hand-crafted features)

All models are evaluated on the same chronological (time-ordered) train/test split
using RMSE, MAE, MAPE, and R².

## Folder structure
```
Stock_Price_Prediction_Project/
├── README.md
├── dataset/
│   └── AAPL_Stock_Price_Dataset.csv     # real AAPL OHLCV data, 2015-2025
├── notebook/
│   └── Stock_Price_Prediction_Final.ipynb
└── model/
    ├── stock_price_forecasting_model.pkl  # best feature-based model (by RMSE)
    ├── feature_columns.pkl                # ordered list of feature columns
    └── scaler.pkl                         # fitted MinMaxScaler for inference
```

## Requirements
- Already available: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`
- Needed only for the ARIMA and LSTM sections (install if you want to run those cells):
  ```
  pip install statsmodels tensorflow
  ```

## How to run
1. Open `notebook/Stock_Price_Prediction_Final.ipynb` in Jupyter / VS Code / Colab.
2. Run cells top to bottom. The data-loading, EDA, feature engineering, and
   Linear/Tree/Ensemble model cells are already pre-executed in this notebook (with
   real outputs and charts) so you can read through it without running anything.
3. The ARIMA and LSTM cells need `statsmodels` / `tensorflow` respectively and are
   left un-executed here — run them yourself if your environment has those packages.

## Key results (feature-based models, test period only)

| Model              | RMSE   | MAE    | MAPE   | R²      |
|---------------------|--------|--------|--------|---------|
| Naive Persistence   | 3.56   | 2.40   | 1.13%  | 0.985   |
| Linear Regression   | 3.63   | 2.45   | 1.15%  | 0.985   |
| Random Forest       | 39.43  | 30.07  | 12.62% | -0.821  |
| Gradient Boosting   | 40.58  | 31.13  | 13.08% | -0.928  |
| Decision Tree       | 46.55  | 36.79  | 15.58% | -1.537  |

(ARIMA and LSTM rows will populate once you run those cells in an environment with
`statsmodels`/`tensorflow` installed — see notebook section 11 and 13.)

**Best feature-based model saved to `model/`: Linear Regression.**

## Notable finding
The tree-based models (Decision Tree, Random Forest, Gradient Boosting) score *worse*
than a naive "predict no change" baseline on this dataset — not due to a bug, but
because AAPL's price kept **climbing well above the highest price seen during
training**, and trees cannot extrapolate beyond their training range. Linear
Regression doesn't have this limitation, which is why it (and the naive baseline)
wins here. This is discussed in detail in the notebook (Section 12, "Why do the tree
models look so much worse here?") and is a genuinely useful, real observation for
the report/discussion.

## Limitations & Future Work
See the "Conclusion & Discussion" section in the notebook — in short: predicting
price *level* one day ahead is close to a random walk problem; predicting *returns*
or *direction* instead would be a stronger next step and would likely change the
tree-model ranking.
