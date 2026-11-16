FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# No training here: the image serves the committed, CI-tested artifact, and the API verifies its SHA-256 at startup.
COPY churn ./churn
COPY web ./web
COPY models ./models
COPY reports ./reports
RUN useradd --system --no-create-home app
USER app
EXPOSE 8000
CMD ["sh", "-c", "uvicorn churn.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
