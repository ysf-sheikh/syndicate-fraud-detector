# Credit Card Fraud Detection System

A machine learning-based credit card fraud detection system built with Python and scikit-learn. The project uses a Random Forest classifier, SMOTE for handling severe class imbalance, feature scaling, and a simulated real-time transaction monitoring pipeline.

The system is designed around a binary classification problem where each transaction is classified as either legitimate or fraudulent. During inference, transactions with a predicted fraud probability of at least 95% are flagged as high-risk fraud alerts.

---

## Project Overview

Credit card fraud datasets are highly imbalanced because legitimate transactions significantly outnumber fraudulent transactions. A model trained directly on such data can become biased toward the majority class.

This project addresses the imbalance by:

1. Splitting the dataset into training and testing sets using stratification.
2. Scaling the transaction `Amount` feature using `StandardScaler`.
3. Applying SMOTE only to the training data.
4. Training a Random Forest classifier on the resampled training data.
5. Evaluating the model using precision, recall, F1 score, and a confusion matrix.
6. Saving the trained model and scaler for later inference.
7. Simulating incoming transactions through a real-time detection script.
8. Flagging transactions when the predicted fraud probability reaches the 95% threshold.

---

## Features

- Random Forest fraud classification
- Stratified train/test splitting
- SMOTE-based class balancing
- Standardized transaction amount
- Precision, recall, and F1-score evaluation
- Confusion matrix visualization
- Persistent model and scaler artifacts using Joblib
- Simulated real-time transaction stream
- Probability-based fraud risk threshold
- Consistent preprocessing between training and inference

---

## Technology Stack

| Technology       | Purpose                            |
| ---------------- | ---------------------------------- |
| Python           | Core programming language          |
| Pandas           | Data loading and manipulation      |
| NumPy            | Numerical operations               |
| Scikit-learn     | Machine learning and evaluation    |
| imbalanced-learn | SMOTE oversampling                 |
| Joblib           | Saving and loading model artifacts |
| Matplotlib       | Visualization                      |
| Seaborn          | Confusion matrix visualization     |

---

## Dataset

This project is designed for the commonly used **Credit Card Fraud Detection** dataset containing European cardholder transactions.

The expected dataset file is:

```text
data/creditcard.csv
```

The dataset is expected to contain the following primary columns:

```text
Time
V1
V2
...
V28
Amount
Class
```

Where:

- `Time` represents the elapsed time between transactions.
- `V1` through `V28` are anonymized transaction features.
- `Amount` represents the transaction amount.
- `Class` is the target variable.
  - `0` = legitimate transaction
  - `1` = fraudulent transaction

The dataset itself is not included in this repository.

---

## Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   └── confusion_matrix.png
│
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── realtime_detector.py
│
├── requirements.txt
└── README.md
```

The `models/` and `outputs/` directories are generated when the training script is executed.

---

## Machine Learning Pipeline

The overall pipeline follows this sequence:

```text
Raw Dataset
     |
     v
Train/Test Split
     |
     +----------------------+
     |                      |
     v                      v
Training Data           Test Data
     |                      |
     v                      |
Scale Amount                |
     |                      |
     v                      |
Apply SMOTE                 |
     |                      |
     v                      |
Train Random Forest         |
     |                      |
     +----------+-----------+
                |
                v
        Evaluate on Test Set
                |
                v
        Save Model + Scaler
                |
                v
       Real-Time Inference
                |
                v
       Fraud Probability
                |
         +------+------+
         |             |
       >= 95%         < 95%
         |             |
         v             v
   FRAUD ALERT     APPROVED
```

---

## Handling Class Imbalance

Fraud detection datasets typically contain far fewer fraudulent transactions than legitimate transactions.

This project uses **SMOTE (Synthetic Minority Over-sampling Technique)** to address this imbalance.

Importantly, SMOTE is applied only after the train/test split:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

SMOTE is then applied exclusively to the training data:

```python
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train,
    y_train
)
```

The test set remains untouched. This prevents synthetic samples from entering the evaluation data and provides a more realistic assessment of model performance.

---

## Feature Preprocessing

The transaction `Amount` feature is standardized using `StandardScaler`.

During training:

```python
scaler = StandardScaler()

X_train["Amount"] = scaler.fit_transform(
    X_train[["Amount"]]
)

X_test["Amount"] = scaler.transform(
    X_test[["Amount"]]
)
```

The fitted scaler is saved as:

```text
models/scaler.pkl
```

The same scaler is loaded during inference so that incoming transactions receive the same transformation used during training.

---

## Model

The project uses a Random Forest classifier with 200 decision trees:

```python
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

Random Forest combines multiple decision trees to produce a classification result. In this project, the model also provides class probabilities through:

```python
model.predict_proba()
```

The probability for the fraud class is used by the real-time detection system.

---

## Model Evaluation

The model is evaluated against the original, untouched test set using:

- Precision
- Recall
- F1 Score
- Confusion Matrix

The evaluation metrics are calculated using:

```python
precision_score(y_test, y_pred)
recall_score(y_test, y_pred)
f1_score(y_test, y_pred)
```

A confusion matrix is also generated and saved to:

```text
outputs/confusion_matrix.png
```

### Why These Metrics Matter

**Precision** measures how many transactions identified as fraudulent were actually fraudulent.

**Recall** measures how many actual fraudulent transactions were successfully detected.

**F1 Score** provides a combined measure of precision and recall.

For fraud detection, examining multiple metrics is important because a model can achieve high overall accuracy while still performing poorly on the minority fraud class.

---

## Real-Time Detection

The `realtime_detector.py` script simulates incoming credit card transactions.

Each simulated transaction contains:

- Transaction ID
- Transaction amount
- Time
- V1 through V28 feature values

Example transaction structure:

```python
mock_data = {
    "Time": random.randint(0, 172800),
    "Amount": amount
}
```

The remaining anonymized features are generated for the simulation:

```python
for v in range(1, 29):
    mock_data[f"V{v}"] = random.gauss(0, 1)
```

The transaction is then passed through the same preprocessing pipeline used for inference.

---

## Fraud Risk Threshold

The real-time detector uses a 95% probability threshold.

```python
if fraud_probability >= 0.95:
    print("FRAUD ALERT")
else:
    print("TRANSACTION APPROVED")
```

This means:

```text
Fraud Probability >= 95%
        |
        v
   FRAUD ALERT
```

While:

```text
Fraud Probability < 95%
        |
        v
TRANSACTION APPROVED
```

The threshold is a configurable business rule rather than a universal definition of fraud.

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ysf-sheikh/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Dataset Setup

Create the data directory:

```bash
mkdir data
```

Place the downloaded dataset inside the directory:

```text
data/creditcard.csv
```

The expected final path is:

```text
data/creditcard.csv
```

---

## Training the Model

From the project root directory, run:

```bash
python src/train_model.py
```

The script will:

1. Load the dataset.
2. Display the dataset dimensions.
3. Display the class distribution.
4. Split the data using stratification.
5. Scale the `Amount` feature.
6. Save the fitted scaler.
7. Apply SMOTE to the training data.
8. Train the Random Forest classifier.
9. Evaluate the model.
10. Generate the confusion matrix.
11. Save the trained model.

After successful training, the following files should be generated:

```text
models/
├── fraud_model.pkl
└── scaler.pkl
```

and:

```text
outputs/
└── confusion_matrix.png
```

---

## Running the Real-Time Detector

After training has completed, run:

```bash
python src/realtime_detector.py
```

The script loads:

```text
models/fraud_model.pkl
models/scaler.pkl
```

and generates three simulated transactions.

A typical output structure is:

```text
====================================
   CREDIT CARD FRAUD MONITOR
====================================

Transaction ID: TXN-12345
Amount: $1,250.50
Fraud Probability: 2.34%

TRANSACTION APPROVED
====================================
```

If the predicted probability reaches the threshold:

```text
Transaction ID: TXN-54321
Amount: $2,100.75
Fraud Probability: 97.21%

FRAUD ALERT
Risk threshold exceeded: 95%
====================================
```

The exact probabilities and transaction IDs will vary because the simulation generates random input values.

---

## Preprocessing for Inference

The `preprocessing.py` module provides:

```python
preprocess_new_transaction()
```

This function:

1. Loads the saved scaler.
2. Converts the incoming transaction dictionary into a Pandas DataFrame.
3. Scales the `Amount` feature.
4. Ensures all expected features exist.
5. Places the features in the expected order.

The expected feature order is:

```text
Time
V1
V2
...
V28
Amount
```

Missing features are currently filled with `0.0` by the preprocessing function.

This allows the inference pipeline to maintain a consistent feature structure before passing data to the trained model.

---

## Dependencies

The project uses the following Python packages:

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
imbalanced-learn>=0.10.0
joblib>=1.2.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

Install them using:

```bash
pip install -r requirements.txt
```

---

## Generated Artifacts

### `models/fraud_model.pkl`

Serialized Random Forest classifier used for fraud prediction.

### `models/scaler.pkl`

Serialized `StandardScaler` fitted on the training `Amount` feature.

### `outputs/confusion_matrix.png`

Visualization of actual versus predicted transaction classes on the test set.

---

## Important Considerations

### Synthetic Real-Time Transactions

The real-time detector currently generates synthetic feature values rather than receiving actual transactions from a payment system or API.

Therefore, it is a simulation of an inference pipeline rather than a production payment-monitoring service.

### Risk Threshold

The 95% threshold is a demonstration rule:

```python
fraud_probability >= 0.95
```

In a production fraud detection environment, the threshold would normally be selected based on business requirements, fraud losses, false-positive costs, investigation capacity, and validation data.

### Model Probability

The displayed fraud probability is the Random Forest's `predict_proba()` output. It should not automatically be interpreted as a perfectly calibrated real-world probability of fraud.

### Data Consistency

The inference data must follow the same feature structure expected during training:

```text
Time, V1-V28, Amount
```

Changes to the dataset structure or preprocessing pipeline should be reflected consistently in both training and inference.

---

## Limitations

This project is intended as a machine learning and fraud detection demonstration rather than a production financial security system.

Current limitations include:

- Simulated real-time transactions
- No live payment gateway integration
- No transaction database
- No API endpoint
- No persistent transaction history
- No model monitoring
- No automated retraining
- No probability calibration
- No production authentication or authorization
- No alert notification service
- No concept-drift detection
- Randomly generated inference features do not represent genuine transaction distributions

---

## Potential Future Improvements

Possible extensions include:

- REST API for transaction scoring
- Streamlit or web-based monitoring dashboard
- Transaction database integration
- Live transaction ingestion
- Configurable fraud thresholds
- Precision-recall curve analysis
- ROC-AUC evaluation
- Probability calibration
- Hyperparameter optimization
- Cross-validation
- Feature importance analysis
- Model monitoring
- Data drift detection
- Automated model retraining
- Alert logging and investigation workflow
- Integration with financial risk dashboards

---

## Security and Data Privacy

The project should not be used with real customer payment information without appropriate security, privacy, regulatory, and organizational controls.

Any real-world deployment should consider applicable requirements for financial data, personally identifiable information, access control, encryption, audit logging, and data retention.

---

## License

Add the project's chosen license here if this repository is intended for public distribution.

Example:

```text
MIT License
```

---

## Author

**Yousuf Sheikh**

GitHub: [ysf-sheikh](https://github.com/ysf-sheikh)

---

## Disclaimer

This project is intended for educational, research, and demonstration purposes. It is not a production-ready fraud prevention system and should not be used as the sole mechanism for approving, declining, or blocking real financial transactions.
