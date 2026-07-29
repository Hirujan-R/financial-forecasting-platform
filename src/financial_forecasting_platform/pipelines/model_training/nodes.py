import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBClassifier

from sklearn.metrics import classification_report, accuracy_score   




def create_lr_pipeline(X: pd.DataFrame, onehot_chr_features: list | None = None, 
                       ordinal_chr_features: list | None = None, 
                       skewed_features: list | None = None 
                       ):
    if onehot_chr_features is None:
        onehot_chr_features = ["Ticker"]
    if ordinal_chr_features is None:
        ordinal_chr_features = ["day_of_week"]
    if skewed_features is None:
        skewed_features = ["log_return_volatility_5", "log_return_volatility_10", "log_return_volatility_20", "bollinger_upper_distance_20", "bollinger_lower_distance_20", "bollinger_bandwidth_20" ,
                  "range_percentage", "upper_shadow_pct", "lower_shadow_pct", "relative_volume_5", "relative_volume_20", "drawdown_20", "gk_variance_mean_5_lag_1", "gk_variance_mean_20_lag_1", "parkinson_volatility_5_lag_1", "parkinson_volatility_20_lag_1", 
    "rs_volatility_5_lag_1", "rs_volatility_20_lag_1", "yz_volatility_5_lag_1", "yz_volatility_20_lag_1", "yz_volatility_60_lag_1"]
    non_skewed_features = list(set(X.columns) - set(skewed_features) - {"day_of_week", "month", "is_outlier", "Ticker"})
    
    day_of_week_order = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot_encoder",
                OneHotEncoder(),
                onehot_chr_features
            ),
            (
                "ordinal_encoder",
                OrdinalEncoder(categories=day_of_week_order),
                ordinal_chr_features
            ),
            (
                "standard_scaler",
                StandardScaler(),
                non_skewed_features
            ),
            (
                "robust_scaler",
                RobustScaler(),
                skewed_features
            )
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("logreg", LogisticRegression(solver='saga', random_state=42, max_iter=1000))
    ])

    return pipeline


def create_xgb_pipeline(onehot_chr_features: list | None = None, 
                       ordinal_chr_features: list | None = None
                       ):
    if onehot_chr_features is None:
        onehot_chr_features = ["Ticker"]
    if ordinal_chr_features is None:
        ordinal_chr_features = ["day_of_week"]
    
    day_of_week_order = [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "one_hot_encoder",
                OneHotEncoder(),
                onehot_chr_features
            ),
            (
                "ordinal_encoder",
                OrdinalEncoder(categories=day_of_week_order),
                ordinal_chr_features
            )
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("xgb", XGBClassifier(objective='binary:logistic',eval_metric='logloss',
                                random_state=42))
    ])

    return pipeline

def train_model(X_train: pd.DataFrame, y_train: pd.Series,
                   pipeline, param_grid: dict):


    y_train_series = y_train.squeeze()
    
    tscv = TimeSeriesSplit(n_splits=5)

    # Configure GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring='roc_auc',
        n_jobs=-1
    )

    print("Starting Grid Search...")
    grid_search.fit(X_train, y_train_series)   

    print(f"Best Hyperparameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    return {
        "model": best_model,
        "params": best_params
    }
    

