# Financial Forecasting Platform — Volatility Regime Prediction

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

A machine learning platform that predicts whether a stock is about to enter a
**volatility expansion** or **volatility contraction** regime, built end-to-end
from experiment to a public HTTPS dashboard on AWS.

The project is a full MLOps stack: a Kedro experimentation pipeline, DVC data
versioning, MLflow experiment tracking + model registry, a FastAPI serving layer,
a Streamlit dashboard, and GitHub Actions CI/CD — all deployed on AWS (ECS
Fargate, RDS, S3, ALB, Route 53).

---

## Goal of the project

Financial markets are not stationary — they repeatedly transition between calm
and volatile environments. This project builds a binary classifier that answers:

> **"Is this stock likely to enter a higher-volatility regime in the near future?"**

| Class | Meaning |
|---|---|
| `0` | Volatility contraction regime |
| `1` | Volatility expansion regime |

The model is trained on daily OHLCV data for **GOOG, AAPL, MSFT, NVDA**, with
**SPY** (market index) and **VIX** (fear index) added as market-context features.

### Target definition

The target uses the **Garman-Klass realised volatility estimator**:

```
trailing 20-day Garman-Klass variance   (baseline)
             │
             ▼
forward 5-day Garman-Klass variance     (realised)
             │
             ▼
forward > baseline  →  class 1 (expansion)
forward ≤ baseline  →  class 0 (contraction)
```

---

## Experimentation pipeline

The experimentation side is orchestrated with **Kedro** as a chain of modular
pipelines:

```
data_ingestion → data_validation → feature_engineering → data_split
      → outlier_handling → model_training → model_selection
```

1. **Data ingestion** — incremental download of OHLCV from Yahoo Finance into PostgreSQL.
2. **Data validation** — 8 checks: non-empty, required columns, no missing values,
   column dtypes, no duplicates, OHLCV financial consistency, date/gap/future checks,
   ticker whitelist.
3. **Feature engineering** — ~30 config-driven feature builders producing three
   model-specific datasets (LR, XGBoost, MLP):
   - price momentum / log returns / lagged returns
   - trend (SMA, EMA, distance from moving averages, EMA crossovers)
   - volatility (Garman-Klass, Parkinson, Rogers-Satchell, Yang-Zhang, Bollinger, vol-of-vol)
   - candle structure, volume, risk (drawdown, Sharpe), and SPY/VIX market context
4. **Data split** — chronological 80/20 train/test split.
5. **Outlier handling** — IsolationForest flagging + 1%/99% quantile clipping.
6. **Model training** — two classifiers:
   - **Logistic Regression** (StandardScaler + RobustScaler + OHE preprocessing)
   - **XGBoost** (gradient-boosted trees)
   - hyperparameters tuned via `GridSearchCV` with **TimeSeriesSplit(5)**,
     scored on ROC-AUC to avoid look-ahead leakage.
7. **Model selection** — picks the best run by ROC-AUC from MLflow and assigns the
   `champion` alias to the winning model version.

---

## Results

The current **champion** is the **Logistic Regression** model
(`volatility-expansion-predictor` v1), evaluated on the held-out chronological test set:

| Metric | Value |
|---|---|
| ROC-AUC | **0.735** |
| Accuracy | **0.673** |
| Precision | **0.684** |
| Recall | **0.455** |
| F1 | **0.546** |

All metrics are recorded per-run in MLflow, so historical experiments can be
compared and the champion can be re-promoted automatically.

---

## MLOps pipeline

```
                 ┌────────────────────────────────────────────┐
                 │          GitHub Actions (CI/CD)             │
                 │  run tests → build images → push to ECR     │
                 │  → force ECS deployment (OIDC role)         │
                 └───────────────┬────────────────────────────┘
                                 ▼
                 ┌────────────────────────────────────────────┐
   Route 53 ──►  │  ALB (HTTPS)  ── dashboard.* / api.*        │
                 └────────┬───────────────┬────────────────────┘
                          ▼               ▼
                 Streamlit dashboard  FastAPI serving
                 (ECS Fargate)        (ECS Fargate)
                          │               │
                          └──────┬────────┘
                                 ▼
                    MLflow (tracking + registry, internal)
                          │
                 RDS PostgreSQL (app + mlflow DB)   S3 (model artifacts)
                 Secrets Manager (credentials)
```

**Components**

| Layer | Technology |
|---|---|
| Orchestration | Kedro pipelines |
| Data versioning | DVC (raw data, features, pipeline outputs) |
| Experiment tracking | MLflow (params, metrics, artifacts, model registry) |
| Serving | FastAPI (`/prediction/{ticker}`, `/history`) |
| Dashboard | Streamlit (prediction card, gauge, candles + Bollinger, SHAP, market overview) |
| Compute | AWS ECS Fargate (ARM64 tasks) |
| Database | RDS PostgreSQL (private) |
| Object storage | S3 (MLflow artifacts + DVC remote) |
| Secrets | AWS Secrets Manager (no plaintext in task definitions) |
| Public entry | ALB + ACM TLS + Route 53 (`dashboard.*`, `api.*`) |
| CI/CD | GitHub Actions with OIDC (`github-actions-role`), per-container workflows |

**CI/CD flow**

```
push to main
   │
   ▼
"Run tests" workflow (pytest — must pass)
   │  success
   ▼
deploy-api.yml / deploy-dashboard.yml / deploy-mlflow.yml
   │  (each) assume github-actions-role via OIDC
   │        build Dockerfile → push to ECR (latest + commit SHA)
   │        force-new-deployment on the ECS service
   ▼
ECS pulls the new image → rolling deploy → ALB health check
```

**Security boundaries**

- Public (internet): ALB only — the dashboard and API.
- Internal-only: MLflow (task security group + your IP), RDS (not publicly
  accessible), S3 (private).
- Secrets injected at runtime from Secrets Manager; IAM roles scoped to
  least privilege (execution role, task role, CI role).

---

## Replicate the results

### 1. Clone and install

```bash
git clone https://github.com/Hirujan-R/financial-forecasting-platform.git
cd financial-forecasting-platform

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

### 2. Run the tests

```bash
pytest tests/ -q
```

### 3. Run the experimentation pipeline

```bash
# Run all Kedro pipelines (ingestion → validation → features → split → training → selection)
kedro run

# Or just the training stage
kedro run --pipeline=model_training

# DVC-managed variant
dvc repro
```

> Data ingestion requires a reachable PostgreSQL database. Connection settings
> are read from environment variables (`DATABASE_HOST`, `DATABASE_NAME`,
> `DATABASE_USER`, `DATABASE_PASSWORD`, `DATABASE_PORT`, `DATABASE_SSLMODE` —
> see `src/financial_forecasting_platform/database/connection.py`), or are
> provided automatically by the local `docker-compose` stack. The datasets in
> `data/` are versioned with **DVC** — pull them from the configured remote with
> `dvc pull` if you want to run the feature/model stages without re-downloading.

### 4. Inspect experiments

Start MLflow and open the UI:

```bash
mlflow ui
```

Or query runs/metrics programmatically:

```bash
python scripts/query_mlflow.py --runs --limit 10
```

### 5. Run the local stack (Docker Compose)

The whole serving stack can run locally:

```bash
docker compose up --build
```

- Dashboard: http://localhost:8501
- API: http://localhost:8000
- MLflow: http://localhost:5001

### 6. Deploy to AWS

The cloud deployment uses the same Docker images and CI/CD. Once the code is
pushed to `main`, the test workflow runs first, then each container is rebuilt,
pushed to ECR, and the ECS services are redeployed. Live URLs after deployment:

- Dashboard: `https://dashboard.hiru-volatility-expansion-prediction.com`
- API: `https://api.hiru-volatility-expansion-prediction.com`

---

## Project structure

```
financial-forecasting-platform/
├── conf/                       # Kedro config (catalog, parameters, mlflow)
├── data/                       # DVC-tracked datasets (01_raw … 08_reporting)
├── db/                         # Local bootstrap SQL for docker-compose
├── docs/
├── notebooks/                  # EDA notebooks
├── scripts/
│   └── query_mlflow.py         # Retrieve experiment results
├── src/financial_forecasting_platform/
│   ├── api/                    # FastAPI app + schemas
│   ├── dashboard/              # Streamlit app + components
│   ├── database/               # Postgres repositories + schema
│   ├── features/               # Feature engineering functions
│   ├── inference/              # Predictor, market data, model loading
│   └── pipelines/              # Kedro pipelines
├── tests/                      # pytest suite
├── .github/workflows/          # tests.yml + per-container deploy workflows
├── Dockerfile.api / .dashboard / .mlflow
├── docker-compose.yml
├── dvc.yaml
└── pyproject.toml
```

---

## Author

**Hirujan Rangaraj** — Computer Science MEng, University of Birmingham.
