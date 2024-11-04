import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import numpy as np
from main import load_data

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'train.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'test.csv')
STORE_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'store.csv')
PREDICTION_OUTPUT_PATH = os.path.join(BASE_DIR, 'datasets', 'test_predictions.csv')


def load_data(file_path):
    try:
        data = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None


def preprocess_data(data, store_data):
    data = data.merge(store_data, on='Store', how='left')

    # Convert Date to datetime format and add time-related features
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['DayOfYear'] = data['Date'].dt.dayofyear
    data['IsWeekend'] = data['DayOfWeek'].isin([6, 7]).astype(int)

    # Fill missing values in CompetitionDistance and normalize it
    data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].mean())
    scaler = StandardScaler()
    data['CompetitionDistance'] = scaler.fit_transform(data[['CompetitionDistance']])

    # Define features and target
    features = ['Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'CompetitionDistance', 'DayOfYear', 'IsWeekend']
    target = 'Sales' if 'Sales' in data.columns else None

    X = data[features]
    y = data[target] if target else None
    return X, y


def remove_outliers(df, feature):
    # IQR filtering to remove outliers
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[feature] >= (Q1 - 1.5 * IQR)) & (df[feature] <= (Q3 + 1.5 * IQR))]


def main():
    train_data = load_data(TRAIN_DATA_PATH)
    store_data = load_data(STORE_DATA_PATH)
    test_data = load_data(TEST_DATA_PATH)

    if train_data is None or store_data is None or test_data is None:
        print("Error loading data.")
        return

    # Remove outliers in Sales from train data
    train_data = remove_outliers(train_data, 'Sales')

    # Preprocess data
    X_train, y_train = preprocess_data(train_data, store_data)
    X_test, _ = preprocess_data(test_data, store_data)

    # Initialize XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    }

    # Convert training data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test)

    # Cross-validation on time-split data with early stopping
    tscv = TimeSeriesSplit(n_splits=5)
    mae_scores, mse_scores, rmse_scores, r2_scores, mape_scores = [], [], [], [], []

    for train_index, val_index in tscv.split(X_train):
        X_t, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        y_t, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

        dtrain_t = xgb.DMatrix(X_t, label=y_t)
        dval = xgb.DMatrix(X_val, label=y_val)

        # Train with early stopping on validation set
        model = xgb.train(
            params=params,
            dtrain=dtrain_t,
            num_boost_round=600,  # Set to desired number of boosting rounds, equivalent to n_estimators
            evals=[(dval, 'validation')],
            early_stopping_rounds=10,
            verbose_eval=False
        )

        # Predict and calculate metrics
        val_pred = model.predict(dval)
        mae_scores.append(mean_absolute_error(y_val, val_pred))
        mse_scores.append(mean_squared_error(y_val, val_pred))
        rmse_scores.append(np.sqrt(mse_scores[-1]))
        r2_scores.append(r2_score(y_val, val_pred))

        # MAPE Calculation (excluding zero values in y_val)
        non_zero_y_val = y_val[y_val != 0]
        non_zero_val_pred = val_pred[y_val != 0]
        mape = np.mean(np.abs((non_zero_y_val - non_zero_val_pred) / non_zero_y_val)) * 100
        mape_scores.append(mape)

    # Average metrics across folds
    print(f"Cross-validated MAE: {np.mean(mae_scores)}")
    print(f"Cross-validated MSE: {np.mean(mse_scores)}")
    print(f"Cross-validated RMSE: {np.mean(rmse_scores)}")
    print(f"Cross-validated R-squared (R²): {np.mean(r2_scores)}")
    print(f"Cross-validated MAPE: {np.mean(mape_scores)}%")

    # Final Model Training on Full Training Data
    final_model = xgb.train(params, dtrain, num_boost_round=model.best_iteration + 1)
    test_data['PredictedSales'] = final_model.predict(dtest)
    test_data.to_csv(PREDICTION_OUTPUT_PATH, index=False)
    print(f"Predictions saved to {PREDICTION_OUTPUT_PATH}.")


if __name__ == "__main__":
    main()
