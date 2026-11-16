import pandas as pd
import pytest

# A textbook flight risk: new, month-to-month, fibre, no support, paying by electronic check.
RISKY = {
    "SeniorCitizen": 1,
    "tenure": 1,
    "MonthlyCharges": 95.0,
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
}
# A loyal customer: six years in, two-year contract, automatic payment.
LOYAL = RISKY | {
    "SeniorCitizen": 0,
    "tenure": 72,
    "Partner": "Yes",
    "Dependents": "Yes",
    "OnlineSecurity": "Yes",
    "TechSupport": "Yes",
    "Contract": "Two year",
    "PaymentMethod": "Credit card (automatic)",
}


@pytest.fixture(scope="session")
def csv_rows():
    """Rows exactly as the raw billing export has them, extra columns included."""
    return pd.read_csv("data/telco.csv", dtype={"TotalCharges": str}).to_dict("records")
