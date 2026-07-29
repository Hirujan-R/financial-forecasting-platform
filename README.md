# Volatility Regime Prediction Model

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

A machine learning project that predicts future volatility regimes in equity markets using historical OHLCV data, technical indicators, volatility estimators, and market context features.

The project builds an end-to-end reproducible machine learning workflow using **Kedro**, **DVC**, and **MLflow** for pipeline orchestration, data versioning, experiment tracking, and model management.

The objective is to classify whether a stock is likely to enter a period of **higher volatility (volatility expansion)** or **lower volatility (volatility contraction)**.

---

# Overview

Financial markets are not stationary and frequently transition between different volatility environments. Periods of low volatility can precede large market movements, while periods of elevated volatility can eventually revert back to calmer conditions.

This project attempts to identify these volatility regime transitions using machine learning.

The model predicts:

| Class | Description |
|---|---|
| 0 | Volatility contraction regime |
| 1 | Volatility expansion regime |

The model is trained on the following stocks:

- Google (GOOG)
- Apple (AAPL)
- Microsoft (MSFT)
- NVIDIA (NVDA)

---

# Machine Learning Objective

The target variable is created using the **Garman-Klass realised volatility estimator**.

The model predicts whether future realised volatility will exceed the current volatility baseline.

The target generation process:

```
Calculate historical volatility baseline
            |
            v
Calculate future realised volatility
            |
            v
Compare future volatility against historical volatility
            |
            v
Classify volatility regime
```

Target definition:

```
Future volatility > Historical volatility
                |
                v
        Volatility expansion (1)


Future volatility <= Historical volatility
                |
                v
        Volatility contraction (0)
```

This allows the model to learn market conditions associated with upcoming volatility changes.

---

# Project Architecture

```
                         Raw Market Data
                                |
                                v
                         Data Validation
                                |
                                v
                       Feature Engineering
                                |
                                v
                     Feature / Target Split
                                |
                                v
                  Chronological Train/Test Split
                                |
                 +--------------+--------------+
                 |                             |
                 v                             v
        Logistic Regression              XGBoost Classifier
                 |                             |
                 +--------------+--------------+
                                |
                                v
                         Model Evaluation
                                |
                                v
                           MLflow Tracking
                                |
                                v
                         Model Registry
```

---

# Technologies Used

## Programming Language

- Python


## Machine Learning

- Scikit-learn
- XGBoost
- NumPy
- Pandas


## Data Engineering

- Kedro
- Kedro Pipelines
- Parquet datasets


## Experiment Tracking

- MLflow

Used for:

- Experiment tracking
- Hyperparameter logging
- Metric comparison
- Model artifact storage
- Model versioning


## Data Version Control

- DVC

Used for:

- Dataset versioning
- Pipeline reproducibility
- Tracking feature dataset changes
- Reproducing experiments


---

# Project Structure

```
volatility-regime-prediction/
│
├── data/
│   ├── 01_raw/
│   ├── 02_intermediate/
│   ├── 03_primary/
│   └── 04_feature/
│
├── src/
│   └── volatility_regime_prediction/
│       │
│       ├── pipelines/
│       │   ├── data_processing/
│       │   ├── feature_engineering/
│       │   └── modelling/
│       │
│       ├── features/
│       ├── models/
│       └── utils/
│
├── conf/
│   └── base/
│       ├── catalog.yml
│       ├── parameters.yml
│       └── globals.yml
│
├── notebooks/
│
├── mlruns/
│
├── dvc.yaml
├── pyproject.toml
└── README.md
```

---

# Dataset

Historical daily OHLCV data is used for:

```
GOOG
AAPL
MSFT
NVDA
```

Each observation contains:

| Feature | Description |
|---|---|
| Date | Trading date |
| Ticker | Stock symbol |
| Open | Opening price |
| High | Daily high price |
| Low | Daily low price |
| Close | Closing price |
| Volume | Trading volume |

---

# Feature Engineering

The feature engineering pipeline creates features across several market dimensions:

- Price momentum
- Trend behaviour
- Volatility dynamics
- Market stress
- Volume activity
- Risk-adjusted performance
- Broader market conditions


---

# Price Features

## Returns

Lagged returns and log returns are created to capture recent price movements.

Examples:

```
return_lag_1
return_lag_5
return_lag_10

log_return_lag_1
log_return_lag_20
```

---

## Momentum

Momentum measures the speed and direction of price movement.

Example:

```
momentum_20
```

Formula:

```
Momentum = Close(t) - Close(t-n)
```

---

# Trend Features

## Moving Averages

Implemented:

- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)


Examples:

```
SMA_20
SMA_50

ema_12
ema_26
```

---

## Moving Average Distance

Measures how far the current price deviates from its historical average.

Examples:

```
distance_close_vs_SMA_20

log_distance_close_vs_SMA_20
```

---

## EMA Crossover

Captures potential trend changes.

Example:

```
ema5_minus_ema20
```

Formula:

```
Short EMA - Long EMA
```

---

# Volatility Features

Multiple volatility estimators are implemented to capture different aspects of market uncertainty.

---

## Garman-Klass Volatility

Uses:

- Open
- High
- Low
- Close

Features:

```
gk_variance_lag_n

gk_variance_mean_n

gk_regime_short_long
```

---

## Parkinson Volatility

Uses high-low price ranges.

Features:

```
parkinson_variance

parkinson_volatility

parkinson_vol_of_vol

parkinson_regime
```

---

## Rogers-Satchell Volatility

Accounts for directional price movement.

Features:

```
rogers_satchell_variance

rs_volatility

rs_vol_of_vol

rs_regime
```

---

## Yang-Zhang Volatility

Combines:

- Overnight price gaps
- Open-close movement
- Rogers-Satchell volatility


Features:

```
yz_variance

yz_variance_mean

yz_volatility

yz_vol_of_vol

yz_regime

yz_volatility_ratio
```

---

# Technical Indicators

## Bollinger Bands

Used to capture volatility compression and breakout conditions.

Features:

```
bollinger_upper_distance

bollinger_lower_distance

bollinger_bandwidth
```

---

## Relative Strength Index (RSI)

Measures momentum extremes.

Feature:

```
rsi_14
```

---

## Candle Features

Candlestick structure features:

```
candle_body

body_percentage

upper_shadow

lower_shadow

upper_shadow_pct

lower_shadow_pct
```

These capture:

- Buying pressure
- Selling pressure
- Market indecision

---

# Volume Features

Volume-based features:

```
volume_pct_change

volume_sma

relative_volume

return_x_volume
```

These measure:

- Trading activity
- Strength behind price movements
- Market participation

---

# Risk Features

Risk-related features:

```
drawdown

rolling_window_mdd

rolling_sharpe_ratio
```

These capture:

- Market stress
- Downside risk
- Risk-adjusted returns

---

# Market Context Features

Additional market information is incorporated using SPY and VIX features.

---

# SPY Features

Market index features:

```
spy_lag_return

spy_volatility

spy_drawdown

spy_sma_ratio

spy_return_zscore
```

These provide broader market context.

---

# VIX Features

VIX represents market fear and expected volatility.

Features:

```
vix_level

vix_return_lag

vix_SMA

vix_percentile
```

---

# Machine Learning Models

Two classification models are trained.

---

# Logistic Regression

A linear baseline model used to understand relationships between market features and volatility regimes.

Pipeline:

```
Feature Scaling
        |
        v
Logistic Regression
```

Advantages:

- Interpretable coefficients
- Fast training
- Strong baseline model


Hyperparameters are optimised using:

```
GridSearchCV
```

---

# XGBoost Classifier

A gradient boosted decision tree model designed to capture nonlinear relationships.

Pipeline:

```
Feature Dataset
        |
        v
XGBoost Classifier
```

Advantages:

- Captures feature interactions
- Handles nonlinear patterns
- Robust with financial features


Optimised hyperparameters:

- Learning rate
- Maximum tree depth
- Number of estimators
- Subsample ratio
- Feature sampling ratio

---

# Model Training Pipeline

```
Raw Data
    |
    v
Feature Engineering
    |
    v
Missing Value Handling
    |
    v
Feature/Target Split
    |
    v
Chronological Train-Test Split
    |
    v
Feature Scaling
    |
    v
Hyperparameter Optimisation
    |
    v
Model Training
    |
    v
Evaluation
    |
    v
MLflow Logging
```

---

# Evaluation Metrics

Models are evaluated using:

## Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score


## Additional Metrics

- ROC-AUC
- Confusion Matrix
- Feature Importance
- Model coefficients


---

# MLOps Workflow

## Kedro

Kedro manages the machine learning workflow by providing:

- Modular pipelines
- Data catalog management
- Parameter management
- Reproducible execution


Pipeline structure:

```
Data Processing
        |
        v
Feature Engineering
        |
        v
Model Training
        |
        v
Model Evaluation
```

---

# DVC

DVC provides dataset and pipeline version control.

Example workflow:

```
Change Dataset
        |
        v
dvc repro
        |
        v
Rebuild Pipeline
        |
        v
Generate New Experiment
```

DVC tracks:

- Raw market data
- Feature datasets
- Pipeline outputs

---

# MLflow

MLflow tracks experiments and manages model versions.

Logged information:

- Hyperparameters
- Training metrics
- Validation metrics
- Model artefacts


Example experiment structure:

```
Volatility Regime Prediction

|
├── Logistic Regression Experiment
|
└── XGBoost Experiment
```

---

# Reproducibility

The complete pipeline can be reproduced using:

```bash
dvc repro
```

This will:

1. Check data dependencies
2. Run Kedro pipelines
3. Generate features
4. Train models
5. Produce evaluation results

---

# Installation

Clone repository:

```bash
git clone <repository-url>

cd volatility-regime-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the Kedro pipeline:

```bash
kedro run
```

Run the DVC pipeline:

```bash
dvc repro
```

Launch MLflow UI:

```bash
mlflow ui
```

---

# Future Improvements

## Modelling Improvements

Potential improvements:

- LightGBM implementation
- Neural network models
- Temporal models:
  - LSTM
  - Temporal CNN
  - Transformers

- Probability calibration for regime confidence scores


---

## Feature Improvements

Potential additional features:

- Options implied volatility
- Market breadth indicators
- Sector performance
- Macroeconomic variables
- Interest rates
- Credit spreads
- Asset correlation features


---

## Deployment Architecture

Possible production architecture:

```
Scheduled Market Data Ingestion
              |
              v
Feature Engineering Pipeline
              |
              v
MLflow Model Registry
              |
              v
FastAPI Prediction API
              |
              v
Dashboard / Monitoring System
```

---

# Author

Hirujan Rangaraj

Computer Science MEng  
University of Birmingham

