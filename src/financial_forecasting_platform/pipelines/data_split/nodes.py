import pandas as pd

def split_features_target(df: pd.DataFrame, 
                          target_variable: str = "market_movement") -> dict:
    return_df = df.copy()
    return_df.sort_index(inplace=True)
    return_df = return_df.dropna()
    X = return_df.drop([target_variable], axis=1)
    y = return_df[[target_variable]]
    return {"X": X, "y": y}


def split_train_test(X: pd.DataFrame, y: pd.Series, training_proportion: float = 0.8) -> dict:
    if training_proportion >= 1 or training_proportion <= 0:
        raise ValueError("training_proportion must be a float between 0 and 1.")
    X.sort_index(inplace=True)
    y.sort_index(inplace=True)
    split_index = int(len(X) * training_proportion)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return {
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test
    }
