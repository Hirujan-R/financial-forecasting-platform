import pandas as pd
from .load_parameters import load_parameters
from financial_forecasting_platform.pipelines.feature_engineering.nodes import (
    create_all_features,
    lr_feature_engineering,
    xgboost_feature_engineering,
    merge_dataframes
)
from financial_forecasting_platform.pipelines.outlier_handling.nodes import (
     outlier_detection, clip_outliers
)

def engineer_data(
        stock_data: pd.DataFrame,
        spy_data: pd.DataFrame,
        vix_data: pd.DataFrame,
        model_type: str):

        kedro_params = load_parameters()
        stock_features_config = kedro_params["stock_features_config"]
        stock_features_config = [
            feature for feature in stock_features_config
            if feature["function"] != "create_market_movement_target"
        ]
        feature_engineered_stock_data = create_all_features(
            stock_data, 
            stock_features_config,
            kedro_params["columns_to_drop"],
            kedro_params["date_column"]
        )
        feature_engineered_spy_data = create_all_features(
                spy_data, 
                kedro_params["spy_features_config"],
                kedro_params["spy_columns_to_drop"],
                kedro_params["date_column"]
        )
        feature_engineered_vix_data = create_all_features(
                vix_data, 
                kedro_params["vix_features_config"],
                kedro_params["vix_columns_to_drop"],
                kedro_params["date_column"]
        )
        merged_data = merge_dataframes(feature_engineered_stock_data, 
                                       feature_engineered_spy_data, 
                                       feature_engineered_vix_data)

        if model_type == "Logistic Regression":
            lr_features = kedro_params["lr_features"]
            lr_features.remove("market_movement")
            engineered_data = lr_feature_engineering(
                merged_data,
                lr_features
            )
            outlier_detected_data = outlier_detection(
                engineered_data,
                kedro_params["outlier_feature_selection"])
            outlier_clipped_data = clip_outliers(
                outlier_detected_data,
                kedro_params["clip_columns"]
            )
            return outlier_clipped_data
        elif model_type == "XGBoost":
            xgboost_features = kedro_params["xgboost_features"]
            xgboost_features.remove("market_movement")
            return xgboost_feature_engineering(
                merged_data,
                xgboost_features
            )
        else:
            raise ValueError("Don't provide an unsupported model.")
        