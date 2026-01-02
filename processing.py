# =====================================================
# processing.py
# =====================================================

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
from joblib import dump

def automate_preprocessing(data, target_column, save_pipeline_path, save_train_path, save_test_path):
    print("=== START AUTOMATED PREPROCESSING ===")

    # 1. Basic cleaning
    data = data.replace(r'^\s*$', np.nan, regex=True)

    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")
    data = data.dropna(subset=["TotalCharges"])

    # Encode target ke 0/1 dan drop missing
    data[target_column] = data[target_column].map({"No": 0, "Yes": 1})
    data = data.dropna(subset=[target_column])

    # Replace "No internet service" / "No phone service"
    cols_internet = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                     'TechSupport', 'StreamingTV', 'StreamingMovies']
    for col in cols_internet:
        if col in data.columns:
            data[col] = data[col].replace("No internet service", "No")

    if "MultipleLines" in data.columns:
        data["MultipleLines"] = data["MultipleLines"].replace("No phone service", "No")

    if "customerID" in data.columns:
        data = data.drop("customerID", axis=1)

    print("✔ Basic cleaning selesai")

    # 2. Identify numeric & categorical features
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # 3. Pipeline
    numeric_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', numeric_pipeline, numeric_features),
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='passthrough')

    # 4. Split
    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("✔ Train-Test split selesai")
    print("Train size:", X_train.shape)
    print("Test size :", X_test.shape)

    # 5. Fit pipeline & transform
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print("✔ Transformasi dengan pipeline selesai")

    # 6. SMOTETomek only on train
    smote_tomek = SMOTETomek(random_state=42)
    X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_processed, y_train)

    print("✔ SMOTE-Tomek selesai")
    print("Jumlah data sebelum :", y_train.value_counts().to_dict())
    print("Jumlah data sesudah :", y_train_balanced.value_counts().to_dict())

    # 7. Save pipeline & CSV
    os.makedirs(os.path.dirname(save_pipeline_path), exist_ok=True)
    dump(preprocessor, save_pipeline_path)
    print(f"✔ Pipeline disimpan ke: {save_pipeline_path}")

    # Save train CSV
    os.makedirs(os.path.dirname(save_train_path), exist_ok=True)
    X_train_df = pd.DataFrame(X_train_balanced)
    y_train_reset = pd.Series(y_train_balanced, name='Churn')  # pastikan reset index
    pd.concat([X_train_df, y_train_reset], axis=1).to_csv(save_train_path, index=False)
    print(f"✔ Train CSV disimpan ke: {save_train_path}")

    # Save test CSV
    os.makedirs(os.path.dirname(save_test_path), exist_ok=True)
    X_test_df = pd.DataFrame(X_test_processed)
    y_test_reset = y_test.reset_index(drop=True)  # RESET INDEX untuk match
    pd.concat([X_test_df, y_test_reset.rename('Churn')], axis=1).to_csv(save_test_path, index=False)
    print(f"✔ Test CSV disimpan ke: {save_test_path}")

    print("=== AUTOMATED PREPROCESSING DONE ===")
    return X_train_balanced, X_test_processed, y_train_balanced, y_test_reset

# =====================================================
# Run standalone jika ingin langsung testing
# =====================================================
if __name__ == "__main__":
    raw_data_path = "./dataset/Telco-Customer-Churn.csv"
    data = pd.read_csv(raw_data_path)
    target_column = "Churn"

    save_pipeline_path = "./preprocessing_pipeline.joblib"
    save_train_path = "./dataset/preprocessing/train_preprocessed.csv"
    save_test_path = "./dataset/preprocessing/test_preprocessed.csv"

    automate_preprocessing(
        data=data,
        target_column=target_column,
        save_pipeline_path=save_pipeline_path,
        save_train_path=save_train_path,
        save_test_path=save_test_path
    )
