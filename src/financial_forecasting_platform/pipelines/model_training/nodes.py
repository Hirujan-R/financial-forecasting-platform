import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from xgboost import XGBClassifier
import mlflow.sklearn
import mlflow

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


def train_model(experiment_tags: dict, X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    pipeline, param_grid: dict):
    with mlflow.start_run(nested=True, run_name=experiment_tags["model"]):
        mlflow.set_tags(experiment_tags)
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

        mlflow.log_params(best_params)
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path=experiment_tags['model'],  # Changed 'name=' to 'artifact_path='
            registered_model_name=experiment_tags['model'],
            pyfunc_predict_fn="predict",
            serialization_format="cloudpickle"  # Note: skops_trusted_types is for skops format
        )
        eval_data = pd.DataFrame(X_test)
        eval_data["target"] = y_test

        eval_dataset = mlflow.data.from_pandas(
            df=eval_data,
            targets="target",
            name=f"{experiment_tags['model']}_features"
        )
        mlflow.evaluate(
            model=model_info.model_uri,
            data=eval_dataset,
            model_type="classifier",
        )

            
    return model_info.model_uri



