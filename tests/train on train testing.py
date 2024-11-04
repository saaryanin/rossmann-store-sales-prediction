import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from main import load_data

# Define paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'train.csv')
STORE_DATA_PATH = os.path.join(BASE_DIR, 'datasets', 'store.csv')

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

    if train_data is None or store_data is None:
        print("Error loading data.")
        return

    # Remove outliers in Sales from train data
    train_data = remove_outliers(train_data, 'Sales')

    # Preprocess data
    X_train, y_train = preprocess_data(train_data, store_data)

    # Initialize XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    }

    # Convert training data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)

    # Train on the full training data
    final_model = xgb.train(params, dtrain, num_boost_round=600)

    # Predict on the same training data
    train_pred = final_model.predict(dtrain)

    # Calculate metrics
    mae = mean_absolute_error(y_train, train_pred)
    mse = mean_squared_error(y_train, train_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, train_pred)

    # MAPE calculation, avoiding division by zero
    non_zero_y_train = y_train[y_train != 0]
    non_zero_train_pred = train_pred[y_train != 0]
    mape = np.mean(np.abs((non_zero_y_train - non_zero_train_pred) / non_zero_y_train)) * 100

    # Print metrics
    print(f"Training Mean Absolute Error (MAE): {mae}")
    print(f"Training Mean Squared Error (MSE): {mse}")
    print(f"Training Root Mean Squared Error (RMSE): {rmse}")
    print(f"Training R-squared (R²): {r2}")
    print(f"Training Mean Absolute Percentage Error (MAPE): {mape}%")

if __name__ == "__main__":
    main()
