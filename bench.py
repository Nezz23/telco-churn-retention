"""Throughput of the running API over real HTTP. Start it first: uvicorn app:app --port 8000"""
import sys
import time

import httpx

from train import load

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/predict"
rows = load()[0].to_dict("records")

with httpx.Client(timeout=60) as client:
    for batch in (1, 100, 1000):
        n_requests = max(1, 2000 // batch)
        start = time.perf_counter()
        for i in range(n_requests):
            client.post(URL, json=rows[(i * batch) % len(rows):][:batch]).raise_for_status()
        elapsed = time.perf_counter() - start
        print(f"batch={batch:>4}  {n_requests * batch / elapsed:>8.0f} predictions/s  "
              f"{elapsed / n_requests * 1000:.1f} ms/request")
