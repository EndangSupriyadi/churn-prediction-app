# =====================================================
# modelling.py
# =====================================================

import os
import pandas as pd
from joblib import dump
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# =====================================================
# Load processed data
# =====================================================
train_data = pd.read_csv("./dataset/preprocessing/train_preprocessed.csv")
test_data = pd.read_csv("./dataset/preprocessing/test_preprocessed.csv")

target_column = "Churn"

X_train = train_data.drop(columns=[target_column])
y_train = train_data[target_column]

X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

input_example = X_train.head(5)

# =====================================================
# MLflow setup
# =====================================================
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Eksperiment_SML_Endang_Supriyadi")

# =====================================================
# Train + Hyperparameter Tuning
# =====================================================
with mlflow.start_run():
    mlflow.autolog()

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Fit ulang model terbaik
    best_model.fit(X_train, y_train)

    # Save model lokal
    os.makedirs("./model", exist_ok=True)
    model_path = "./model/model.pkl"
    dump(best_model, model_path)
    print(f"✅ Model disimpan ke: {model_path}")

    # Log ke MLflow
    mlflow.log_params(best_params)
    mlflow.sklearn.log_model(best_model, name="random_forest_model", input_example=input_example)

    # Evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric("test_accuracy", accuracy)

print("===================================")
print("🎉 TRAINING SELESAI")
print("📊 Akurasi Test :", accuracy)
print("📦 Model Local :", model_path)
print("===================================")
