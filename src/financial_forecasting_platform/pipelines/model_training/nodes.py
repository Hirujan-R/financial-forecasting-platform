import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, \
    recall_score, f1_score

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


def train_model(experiment_tags: dict, registered_model_name: str,
                     X_train: pd.DataFrame, y_train: pd.Series,
                    X_test: pd.DataFrame, y_test: pd.Series,
                    pipeline, param_grid: dict):
    with mlflow.start_run(nested=True, run_name=experiment_tags["model"]):
        mlflow.set_tags(experiment_tags)

        # Logging training dataset
        df_train = X_train.copy()
        df_train["target"] = y_train
        train_dataset = mlflow.data.from_pandas(df_train,
                                               name="Training dataset",
                                               targets="target")
        mlflow.log_input(dataset=train_dataset, context='train')

        # Time-series Cross Validation
        y_train_series = y_train.squeeze()
        
        tscv = TimeSeriesSplit(n_splits=5)

        # Configure GridSearchCV
        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=tscv,
            scoring='roc_auc',
            n_jobs=-1,
            refit=True
        )

        print("Starting Grid Search...")
        grid_search.fit(X_train, y_train_series)   

        print(f"Best Hyperparameters: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_


        # Logging evaluation dataset
        
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        eval_data = X_test.copy()
        eval_data["target"] = y_test
        eval_data["prediction"] = y_pred

        eval_dataset = mlflow.data.from_pandas(eval_data,
                                               name="Test dataset",
                                               predictions="prediction",
                                               targets="target")
        mlflow.log_input(dataset=eval_dataset, context='test')

        mlflow.log_params(best_params)
        model_info = mlflow.sklearn.log_model(
            sk_model=best_model, 
            artifact_path=experiment_tags['model'],  # Changed 'name=' to 'artifact_path='
            # registered_model_name=experiment_tags['model'],
            registered_model_name=registered_model_name,
            pyfunc_predict_fn="predict",
            serialization_format="cloudpickle"  # Note: skops_trusted_types is for skops format
        )
        model_version = model_info.registered_model_version
        for key, value in experiment_tags.items():
            mlflow.set_model_version_tag(
                name=registered_model_name,
                version=model_version,
                key=key,
                value=value
            )

        


        mlflow.log_metrics({
            "roc_auc": roc_auc_score(y_test, y_prob),
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred)
        })
            
    return model_info.model_uri



