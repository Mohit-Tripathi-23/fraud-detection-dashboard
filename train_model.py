import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "fraud_detection_model.pkl"

def generate_synthetic_data(n_samples: int = 3000) -> pd.DataFrame:
    np.random.seed(42)

    transaction_amount = np.random.randint(500, 200000, n_samples)
    account_age_days = np.random.randint(10, 4000, n_samples)
    transactions_last_24h = np.random.randint(0, 25, n_samples)
    avg_transaction_amount = np.random.randint(500, 50000, n_samples)
    location_change = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    device_change = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])
    failed_logins_last_24h = np.random.randint(0, 10, n_samples)
    international_transaction = np.random.choice([0, 1], n_samples, p=[0.85, 0.15])

    df = pd.DataFrame({
        "transaction_amount": transaction_amount,
        "account_age_days": account_age_days,
        "transactions_last_24h": transactions_last_24h,
        "avg_transaction_amount": avg_transaction_amount,
        "location_change": location_change,
        "device_change": device_change,
        "failed_logins_last_24h": failed_logins_last_24h,
        "international_transaction": international_transaction
    })

    labels = []
    for _, row in df.iterrows():
        risk_score = 0

        if row["transaction_amount"] > row["avg_transaction_amount"] * 4:
            risk_score += 2
        elif row["transaction_amount"] > row["avg_transaction_amount"] * 2:
            risk_score += 1

        if row["transactions_last_24h"] > 10:
            risk_score += 1

        if row["failed_logins_last_24h"] > 3:
            risk_score += 2
        elif row["failed_logins_last_24h"] > 0:
            risk_score += 1

        if row["location_change"] == 1:
            risk_score += 1

        if row["device_change"] == 1:
            risk_score += 1

        if row["international_transaction"] == 1:
            risk_score += 1

        if row["account_age_days"] < 90:
            risk_score += 1

        if risk_score >= 6:
            labels.append(2)  # High Risk
        elif risk_score >= 3:
            labels.append(1)  # Medium Risk
        else:
            labels.append(0)  # Low Risk

    df["target"] = labels
    return df

def train_and_save():
    df = generate_synthetic_data()

    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    print(f"Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    train_and_save()