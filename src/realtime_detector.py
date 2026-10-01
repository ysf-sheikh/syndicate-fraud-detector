import time
import random
import joblib
import pandas as pd
from preprocessing import preprocess_new_transaction

def simulate_realtime_stream():
    model_path = "models/fraud_model.pkl"
    scaler_path = "models/scaler.pkl"
    
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print("[!] Trained model not found. Run training script first.")
        return

    print("====================================")
    print("   CREDIT CARD FRAUD MONITOR        ")
    print("==================================----\n")

    # Generate a few simulated test transactions
    for i in range(1, 4):
        txn_id = f"TXN-{random.randint(10000, 99999)}"
        amount = round(random.uniform(10.0, 2500.0), 2)
        
        # Construct mock raw feature values (Time, V1-V28, Amount)
        mock_data = {
            "Time": random.randint(0, 172800),
            "Amount": amount
        }
        for v in range(1, 29):
            mock_data[f"V{v}"] = random.gauss(0, 1)

        # Preprocess transaction
        processed_df = preprocess_new_transaction(mock_data, scaler_path)
        
        # Predict probability
        fraud_probability = model.predict_proba(processed_df)[0][1]
        
        # Inject artificial high risk for demonstration on transaction 2 if desired, 
        # or rely on the model's prediction score.
        print(f"Transaction ID: {txn_id}")
        print(f"Amount: ${amount:,.2f}")
        print(f"Fraud Probability: {fraud_probability * 100:.2f}%\n")

        if fraud_probability >= 0.95:
            print("🚨 FRAUD ALERT")
            print("Risk threshold exceeded: 95%")
        else:
            print("✓ TRANSACTION APPROVED")
            
        print("====================================")
        time.sleep(1)

if __name__ == "__main__":
    simulate_realtime_stream()