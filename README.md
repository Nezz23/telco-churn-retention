# Telco Churn: Who Is Worth Saving?

Most churn projects stop at "will this customer leave?" This one answers the question a retention team actually has: **which customers are worth an offer, and why are they at risk?**

An XGBoost model scores churn risk. Expected-value targeting turns that score into a send/don't-send decision. Per-customer SHAP reasons explain each score. A FastAPI service serves all three, and CI retrains and gates the model every week.

```
customer ─▶ FastAPI /predict ─▶ XGBoost pipeline ─▶ P(churn)
                                       │                │
                                 SHAP reasons    expected value = P × save rate × 12 × monthly bill − offer cost
                                       ▼                ▼
                     {"churn_probability": 0.8826, "expected_value": 201.85, "send_offer": true,
                      "top_reasons": ["tenure=1", "Contract=Month-to-month", "TotalCharges=95.0"]}
```

## Results

The data is the [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset: 7,043 customers, 26.5% of whom churned. There is a stratified 80/20 split, and every number below is on the 1,409 held-out customers.

**Model**

| Metric | Value |
|---|---|
| ROC-AUC (test) | 0.845 |
| ROC-AUC (5-fold CV) | 0.847 |
| Brier score | 0.135 |
| Calibration error (mean abs, 10 bins) | 0.026 |

The probabilities are well calibrated, which matters because the profit policy multiplies by them. Class re-weighting was left off on purpose, because it would inflate them.

**Targeting policy.** Realised profit on the test customers, under the assumptions below:

| Policy | Customers contacted | Profit |
|---|---|---|
| Contact everyone | 1,409 | −$42,926 |
| Contact if P(churn) > 0.5 | 294 | $25,504 |
| **Expected-value targeting (this model)** | **411** | **$28,551** |
| Perfect foresight (ceiling) | 341 | $61,368 |

The model captures 47% of the profit that perfect foresight would get. Among the customers it contacts, 56% would really have churned (precision), and it reaches 62% of all churners (recall). It skips high-risk customers on cheap plans, because saving them is worth less than the offer costs.

**Robustness to the save-rate assumption**

| Offers save… | Expected-value policy | Contact everyone |
|---|---|---|
| 10% of churners | $128 | −$108,242 |
| 20% | $8,660 | −$75,584 |
| 30% (default) | $28,551 | −$42,926 |
| 50% | $78,721 | $22,389 |

The policy never loses money across this range. Blanket outreach loses money unless offers are very effective.

**Assumptions** (in `train.py`, not in the data): each offer costs $100, it keeps 30% of would-be churners, and a saved customer is worth 12 months of their bill.

**Serving** (local uvicorn, single worker, Apple M4 Pro, scores plus reasons over real HTTP, `bench.py`)

| Batch size | Predictions/s | Latency per request |
|---|---|---|
| 1 | 76 | 13 ms |
| 100 | 3,751 | 27 ms |
| 1,000 | 16,282 | 61 ms |

## Run it

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python train.py                      # trains, logs to MLflow, writes model/ and metrics
.venv/bin/python -m pytest -q                  # behaviour tests + quality gate
.venv/bin/uvicorn app:app --port 8000          # serve
.venv/bin/python bench.py                      # throughput, with the server running
.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db

docker build -t telco-churn . && docker run -p 8000:8000 telco-churn
```

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' -d '[{"SeniorCitizen":1,"tenure":1,
"MonthlyCharges":95,"TotalCharges":95,"gender":"Female","Partner":"No","Dependents":"No","PhoneService":"Yes",
"MultipleLines":"No","InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
"DeviceProtection":"No","TechSupport":"No","StreamingTV":"No","StreamingMovies":"No","Contract":"Month-to-month",
"PaperlessBilling":"Yes","PaymentMethod":"Electronic check"}]'
```

## CI

`.github/workflows/ci.yml` runs on every push and every Monday. It retrains, runs the tests, and builds and smoke-tests the Docker image. The tests act as a quality gate: they fail if ROC-AUC drops below 0.80 or if the policy stops beating "contact everyone".

## Limitations

- The dataset is a static snapshot, so the weekly retrain shows the pipeline is reproducible. It does not adapt to new data.
- The campaign economics are assumptions. The sensitivity table shows how much the result depends on them.
- The model predicts churn, not whether an offer *changes* behaviour. The right next step is an uplift model trained on A/B test data from a real campaign.
