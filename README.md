# Social Media Academic Impact Predictor
This project is an end-to-end machine learning application that predicts whether a student's social media usage is likely to negatively affect their academic performance. It was built as part of an assignment to demonstrate the complete ML lifecycle including training, experiment tracking with MLflow, backend/frontend integration, and deployment on Azure.

## Project Overview
Dataset: A survey dataset of students with features like age, gender, academic level, daily social media usage, preferred platform, sleep habits, and mental health score.

## Goal: 
Predict the binary outcome – whether a student perceives that social media use negatively affects their academic performance.

## Target Variable: Affects_Academic_Performance (Yes/No → 1/0)

## 🛠 Technologies Used
Component	Tool/Library
Model Training	Python, scikit-learn, pandas
Experiment Tracking	MLflow
API (optional)	FastAPI (planned)
Frontend	Streamlit
Deployment	Azure App Service
Model Serialization	pickle

## Features Used
Age
Gender
Academic Level
Country
Avg. Daily Social Media Usage (hours)
Most Used Platform
Sleep Hours Per Night
Mental Health Score (1 to 10)

## How to Run Locally
Clone the repository


`git clone <your_repo_url>`

`cd ml_app_assignment`

`Install dependencies`


`pip install -r requirements.txt`

Run Streamlit app

`streamlit run app.py`

## MLflow Experiment Tracking
MLflow was used to:

Log model parameters (n_estimators, model type)
Track metrics (accuracy, precision, recall)
Store model artifacts (model.pkl, scaler.pkl)


## To run the MLflow UI:

`mlflow ui`

Then visit: http://127.0.0.1:5000

## Sample Prediction Flow
User inputs age, gender, academic level, etc.
App converts inputs to numeric features using the same preprocessing used in training.
Model predicts Yes or No for academic impact.
Result is shown in a clear message on the UI.

