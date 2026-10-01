import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    print("[-] Loading dataset...")
    # Update path to your downloaded Kaggle dataset
    data_path = "data/creditcard.csv"
    
    if not os.path.exists(data_path):
        print(f"[!] Dataset not found at {data_path}. Please place creditcard.csv in the data/ folder.")
        return

    df = pd.read_csv(data_path)
    
    print("[-] Exploring dataset...")
    print(f"Dataset Shape: {df.shape}")
    print("Class Distribution:\n", df["Class"].value_counts(normalize=True) * 100)

    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # 1. Stratified Train/Test Split (BEFORE SMOTE)
    print("[-] Splitting data (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Scale Amount feature
    print("[-] Scaling Amount feature...")
    scaler = StandardScaler()
    X_train["Amount"] = scaler.fit_transform(X_train[["Amount"]])
    X_test["Amount"] = scaler.transform(X_test[["Amount"]])

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    # 3. Apply SMOTE ONLY on Training Data
    print("[-] Applying SMOTE to training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print("Resampled training class counts:\n", y_train_resampled.value_counts())

    # 4. Train Random Forest Classifier
    print("[-] Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_resampled, y_train_resampled)

    # 5. Evaluate Model
    print("[-] Evaluating model on untouched test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n--- Evaluation Metrics ---")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    
    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # Save trained model
    joblib.dump(model, "models/fraud_model.pkl")
    print("[-] Model and scaler successfully saved to models/")

if __name__ == "__main__":
    main()