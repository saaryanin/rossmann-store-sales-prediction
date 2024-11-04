Project Title: Rossmann Store Sales Prediction
Project Overview:
This project is an AI-powered model for predicting sales at Rossmann stores. The model is built using Python and utilizes XGBoost for prediction. The server is implemented with FastAPI, making it accessible through REST API endpoints for single and batch predictions.

Project Structure

.
├── datasets/             # Folder containing the dataset files (train.csv, test.csv, store.csv)
├── models/               # Folder where the trained model (xgboost_model.pkl) is saved
├── server/               # Contains the server script for handling API requests
├── static/               # Static files (optional for your project)
├── tests/                # Unit tests and validation scripts
├── utils/                # Utility functions
├── main.py               # Script to train the model and save it in the models/ folder
├── README.md             # Project documentation and setup instructions
└── requirements.txt      # List of required Python packages

Prerequisites:
-Python 3.9 (recommended)
-Internet connection to download necessary Python packages
Environment Setup:
Clone the repository (if applicable) or download the project files.

Navigate to the project directory:

cd path/to/your/project

Training the Model:
To train the model on the Rossmann Store Sales dataset, run main.py. This script will load the data, preprocess it, train the model, and save it as xgboost_model.pkl in the models/ folder.

Expected Output:
After running the training script, main.py will output evaluation metrics such as MAE, MSE, RMSE, and R² for validation data.
A trained model file (xgboost_model.pkl) will be saved in the models directory.

Running the API Server:
The server is implemented using FastAPI. You can start the server using Uvicorn, which is included in requirements.txt.

Navigate to the server folder and Run the server:


cd server
uvicorn server:app --reload --host 0.0.0.0 --port 8000

Once the server is running, you can access the API documentation at:

http://127.0.0.1:8000/docs

API Endpoints:
The server has three primary endpoints:

Health Check: Verifies that the server is running.

Single Prediction: Accepts a single data point in JSON format and returns the predicted sales.

Example request body:

{
  "Store": 1,
  "DayOfWeek": 5,
  "Promo": 1,
  "Year": 2021,
  "Month": 8,
  "Day": 15,
  "CompetitionDistance": 500.0
}
Batch Prediction: Accepts multiple data points (list of JSON objects) and returns predictions for each.

Example request body:

{
  "data": [
    {
      "Store": 1,
      "DayOfWeek": 5,
      "Promo": 1,
      "Year": 2021,
      "Month": 8,
      "Day": 15,
      "CompetitionDistance": 500.0
    },
    {
      "Store": 2,
      "DayOfWeek": 6,
      "Promo": 0,
      "Year": 2021,
      "Month": 8,
      "Day": 16,
      "CompetitionDistance": 750.0
    }
  ]
}

Testing the API:
You can test the API by using:

Swagger UI: Visit http://127.0.0.1:8000/docs to interact with the API.
curl commands: Send requests from the command line.
Postman or other API testing tools.

Additional Notes
Data Files: Ensure the datasets folder contains train.csv, test.csv, and store.csv for training and testing.
Model File: After training, the xgboost_model.pkl file should be located in the models folder. The API server will load this file when started.