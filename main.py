import os
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib  # Import joblib to save the model

#paths for train, store, and model datasets
TRAIN_DATA_PATH = os.path.join('datasets', 'train.csv')
STORE_DATA_PATH = os.path.join('datasets', 'store.csv')
MODEL_PATH = os.path.join('models', 'xgboost_model.pkl')  # Path to save the model

def load_data(file_path):
    #Load dataset with low_memory=False
    try:
        data = pd.read_csv(file_path, low_memory=False)
        print(f"Data loaded successfully from {file_path}")
        return data
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def preprocess_data(train_data, store_data):
    # Merge, clean, and perform feature engineering on the data, including using IQR.
    data = train_data.merge(store_data, on='Store', how='left')

    # Convert Date to datetime format and extract features
    data['Date'] = pd.to_datetime(data['Date'])
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = data['Date'].dt.day
    data['WeekOfYear'] = data['Date'].dt.isocalendar().week

    # Weekend and holiday indicators
    data['IsWeekend'] = data['DayOfWeek'].isin([6, 7]).astype(int)
    data['IsHoliday'] = data['StateHoliday'].apply(lambda x: 1 if x != '0' else 0)

    # Fill missing values in CompetitionDistance
    data['CompetitionDistance'] = data['CompetitionDistance'].fillna(data['CompetitionDistance'].mean())

    # IQR on Sales
    Q1 = data['Sales'].quantile(0.25)
    Q3 = data['Sales'].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data['Sales'] >= (Q1 - 1.5 * IQR)) & (data['Sales'] <= (Q3 + 1.5 * IQR))]

    features = ['Store', 'DayOfWeek', 'Promo', 'Year', 'Month', 'Day', 'CompetitionDistance']
    target = 'Sales'

    X = data[features]
    y = data[target]
    return X, y

def main():
    train_data = load_data(TRAIN_DATA_PATH)
    store_data = load_data(STORE_DATA_PATH)

    if train_data is None or store_data is None:
        print("Error loading data. Exiting program.")
        return

    X, y = preprocess_data(train_data, store_data)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    # saving the model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    #predict and evaluate on the validation set
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mse ** 0.5  # Root Mean Squared Error
    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"Root Mean Squared Error (RMSE): {rmse}")

if __name__ == "__main__":
    main()
