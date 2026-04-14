import pickle
from pathlib import Path
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_detection_model.pkl"

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def risk_label(pred: int) -> str:
    mapping = {
        0: "Low Risk",
        1: "Medium Risk",
        2: "High Risk"
    }
    return mapping.get(pred, "Unknown")

def main():
    st.title("AI-Based Fraud Detection and Risk Monitoring Dashboard")
    st.write(
        "This Streamlit application simulates real-time fraud risk assessment "
        "for banking transactions using machine learning."
    )

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run train_model.py first.")
        return

    model = load_model()

    tab1, tab2, tab3 = st.tabs(["Single Transaction", "Batch Upload", "Fraud Analytics"])

    with tab1:
        st.subheader("Single Transaction Risk Assessment")

        col1, col2 = st.columns(2)

        with col1:
            transaction_amount = st.number_input(
                "Transaction Amount",
                min_value=100.0,
                max_value=1000000.0,
                value=25000.0,
                step=500.0
            )
            account_age_days = st.number_input(
                "Account Age (days)",
                min_value=1,
                max_value=5000,
                value=365,
                step=10
            )
            transactions_last_24h = st.slider("Transactions in Last 24 Hours", 0, 50, 3)
            avg_transaction_amount = st.number_input(
                "Average Historical Transaction Amount",
                min_value=100.0,
                max_value=500000.0,
                value=12000.0,
                step=500.0
            )

        with col2:
            location_change = st.selectbox("Location Changed?", ["No", "Yes"])
            device_change = st.selectbox("Device Changed?", ["No", "Yes"])
            failed_logins_last_24h = st.slider("Failed Logins in Last 24 Hours", 0, 20, 0)
            international_transaction = st.selectbox("International Transaction?", ["No", "Yes"])

        input_df = pd.DataFrame([{
            "transaction_amount": transaction_amount,
            "account_age_days": account_age_days,
            "transactions_last_24h": transactions_last_24h,
            "avg_transaction_amount": avg_transaction_amount,
            "location_change": 1 if location_change == "Yes" else 0,
            "device_change": 1 if device_change == "Yes" else 0,
            "failed_logins_last_24h": failed_logins_last_24h,
            "international_transaction": 1 if international_transaction == "Yes" else 0
        }])

        if st.button("Assess Fraud Risk"):
            pred = model.predict(input_df)[0]
            probs = model.predict_proba(input_df)[0]
            label = risk_label(pred)

            st.success(f"Fraud Risk Level: **{label}**")

            prob_df = pd.DataFrame({
                "Risk Level": ["Low Risk", "Medium Risk", "High Risk"],
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Risk Level"))

            reasons = []
            if transaction_amount > avg_transaction_amount * 3:
                reasons.append("- Transaction amount is significantly above historical average.")
            if transactions_last_24h > 10:
                reasons.append("- Unusually high transaction frequency detected.")
            if failed_logins_last_24h > 3:
                reasons.append("- Multiple failed logins increase account risk.")
            if location_change == "Yes":
                reasons.append("- Location mismatch detected.")
            if device_change == "Yes":
                reasons.append("- Device change detected.")
            if international_transaction == "Yes":
                reasons.append("- International transaction flagged for added review.")
            if account_age_days < 90:
                reasons.append("- New account profiles may carry higher fraud risk.")

            st.markdown("### Risk Explanation")
            if reasons:
                st.warning("\n".join(reasons))
            else:
                st.info("No major fraud indicators detected.")

    with tab2:
        st.subheader("Batch Fraud Assessment")
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                required_columns = [
                    "transaction_amount",
                    "account_age_days",
                    "transactions_last_24h",
                    "avg_transaction_amount",
                    "location_change",
                    "device_change",
                    "failed_logins_last_24h",
                    "international_transaction"
                ]

                missing = [col for col in required_columns if col not in df.columns]
                if missing:
                    st.error(f"Missing required columns: {missing}")
                else:
                    preds = model.predict(df[required_columns])
                    probs = model.predict_proba(df[required_columns])

                    df["risk_level"] = [risk_label(p) for p in preds]
                    df["high_risk_probability"] = probs[:, 2]

                    st.success("Batch fraud assessment completed.")
                    st.dataframe(df)

                    csv = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results CSV",
                        csv,
                        "fraud_assessment_results.csv",
                        "text/csv"
                    )
            except Exception as exc:
                st.error(f"Error processing file: {exc}")

    with tab3:
        st.subheader("Fraud Monitoring Analytics")

        demo_metrics = pd.DataFrame({
            "Risk Level": ["Low Risk", "Medium Risk", "High Risk"],
            "Count": [68, 22, 10]
        })
        st.bar_chart(demo_metrics.set_index("Risk Level"))

        st.markdown("### Monitoring Insights")
        st.write("- Sudden amount spikes are a major fraud signal.")
        st.write("- Device and location changes should trigger additional verification.")
        st.write("- Combining behavior-based features improves fraud detection quality.")

if __name__ == "__main__":
    main()