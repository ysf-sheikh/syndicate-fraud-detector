import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def get_scaler():
    """Returns a new StandardScaler instance."""
    return StandardScaler()

def preprocess_new_transaction(transaction_dict, scaler_path="models/scaler.pkl"):
    """
    Preprocesses a single incoming transaction dictionary for model inference.
    """
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training first.")
    
    scaler = joblib.load(scaler_path)
    
    # Convert incoming dict to DataFrame
    df = pd.DataFrame([transaction_dict])
    
    # Scale the Amount feature using the fitted scaler
    if "Amount" in df.columns:
        df["Amount"] = scaler.transform(df[["Amount"]])
        
    # Ensure columns match expected feature order (Time, V1-V28, Amount)
    # Assuming standard Kaggle columns structure
    expected_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
    
    # Reindex or fill missing features if necessary
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0
            
    return df[expected_cols]