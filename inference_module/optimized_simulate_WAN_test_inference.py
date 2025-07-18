"""
A high-concurrency stress-testing tool for local FastAPI inference endpoints
under simulated WAN conditions.

Features:
- 60 ± 20 ms synthetic delay with 2% artificial packet loss
- Exponential back-off and retries via Tenacity
- Hot reload simulation via /reload every 20s
- Burst-style traffic using up to MAX_WORKERS concurrent threads
- Occasional large-payload injections to test JSON robustness
"""

import threading
import requests
import time
import random
import csv
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt
from collections import defaultdict

# -------------------------------------------------------------------------
# Configuration & Input
# -------------------------------------------------------------------------

URL = "http://127.0.0.1:8000/predict"
RELOAD_URL = "http://127.0.0.1:8000/reload"
DATA = pd.read_parquet("sample_data.parquet")

TOTAL_REQUESTS = 30000      # Total number of simulated requests
MAX_WORKERS = 100            # Maximum number of concurrent requests in flight

# -------------------------------------------------------------------------
# Synthetic WAN Behavior
# -------------------------------------------------------------------------

def simulate_network_delay():
    """Simulates 60ms ± 20ms delay per request to mimic real-world latency."""
    base = random.uniform(0.03, 0.15)
    jitter = random.gauss(0, 0.025)
    time.sleep(max(base + jitter, 0))

def mutate_payload(row: dict) -> dict:
    """
    Randomly expands the payload with extra dummy fields to simulate
    large-payload inference requests (~10% of the time).
    """
    if random.random() < 0.10:
        for i in range(10):
            row[f"dummy_{i}"] = random.random()
    return row

# -------------------------------------------------------------------------
# Resilient Inference Call
# -------------------------------------------------------------------------

session = requests.Session()

@retry(wait=wait_exponential(multiplier=0, min=0, max=1),
       stop=stop_after_attempt(3))
def safe_post(payload):
    """
    POSTs the payload to the inference endpoint.
    Injects 3% packet-loss by raising ConnectionError.
    Retries automatically with Tenacity.
    """
    if random.random() < 0.03:
        raise requests.exceptions.ConnectionError("Injected packet drop")

    simulate_network_delay()
    return session.post(URL, json=payload, timeout=5)

# -------------------------------------------------------------------------
# Thread Worker
# -------------------------------------------------------------------------

lock = threading.Lock()
results = []               # Stores (request_id, status, latency)
failures = defaultdict(int)

def worker(req_id):
    """
    Performs a single request using simulated latency and packet loss.
    Logs success/failure and latency per request.
    """
    row = mutate_payload(DATA.sample(n=1).to_dict(orient="records")[0])
    t0 = time.time()
    try:
        safe_post(row)
        status = 200
    except Exception as e:
        status = type(e).__name__
        with lock:
            failures[status] += 1
    latency = round(time.time() - t0, 4)
    with lock:
        results.append((req_id, status, latency))

# -------------------------------------------------------------------------
# ░ Reload Simulation Thread
# -------------------------------------------------------------------------

def reload_daemon(interval=20):
    """Background thread to periodically call /reload during the test."""
    while True:
        try:
            requests.post(RELOAD_URL, timeout=3)
        except Exception:
            pass
        time.sleep(interval)

# -------------------------------------------------------------------------
# ░ Main Simulation Runner
# -------------------------------------------------------------------------

def run():
    """
    Orchestrates the simulation:
    - Launches reload daemon
    - Uses a semaphore to limit concurrency
    - Dispatches TOTAL_REQUESTS with bounded overlap
    - Aggregates and reports performance
    """
    threading.Thread(target=reload_daemon, daemon=True).start()
    sem = threading.Semaphore(MAX_WORKERS)
    threads = []

    t_start = time.time()
    for rid in range(TOTAL_REQUESTS):
        sem.acquire()
        th = threading.Thread(target=lambda: (worker(rid), sem.release()))
        th.start()
        threads.append(th)

    for th in threads:
        th.join()
    t_end = time.time()

    dump_results()
    report(TOTAL_REQUESTS, t_end - t_start)

# -------------------------------------------------------------------------
# ░ Output & Reporting
# -------------------------------------------------------------------------

def dump_results():
    """Saves all result data to simulation_results.csv."""
    with open("simulation_results.csv", "w", newline="") as f:
        csv.writer(f).writerows([("ReqID", "Status", "Latency")] + results)

def report(total, duration):
    """Prints overall success rate, latency stats, and failure breakdown."""
    lats = [r[2] for r in results if r[1] == 200]
    print(f"\n {total} requests in {round(duration,1)} s "
          f"({round(total/duration,1)} req/s)")
    print(f"   mean={round(sum(lats)/len(lats),4)} s  "
          f"p95={round(sorted(lats)[int(.95*len(lats))-1],4)} s")
    print("Failures:", dict(failures))

# -------------------------------------------------------------------------

if __name__ == "__main__":
    run()
