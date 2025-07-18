"""
Simulates realistic WAN-style network conditions against a local FastAPI inference server.

Implements:
- Synthetic network latency, jitter, and packet-loss.
- Exponential back-off retries using Tenacity.
- Load ramping (3 → 6 → 10 → 25 concurrent threads per batch).
- Periodic /reload calls to simulate hot-reloading under pressure.
- Occasional large payloads to test JSON overhead resilience.
"""

import threading
import requests
import time
import random
import csv
import psutil
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt

URL = "http://127.0.0.1:8000/predict"
RELOAD_URL = "http://127.0.0.1:8000/reload"
DATA = pd.read_parquet("sample_data.parquet")

# -------------------------------------------------------------------------
# WAN Simulation Layer
# Inject realistic delay, jitter, and optional payload bloat
# -------------------------------------------------------------------------

def simulate_network_delay():
    """Applies synthetic WAN delay with ± jitter to mimic public Internet conditions."""
    base = random.uniform(0.03, 0.15)  # 30–150 ms RTT baseline
    jitter = random.gauss(0, 0.025)    # ±25 ms jitter
    time.sleep(max(base + jitter, 0))

def mutate_payload(row: dict) -> dict:
    """Randomly augments the payload with extra fields to simulate heavier-than-average requests."""
    if random.random() < 0.10:  # 10% of requests are oversized
        for i in range(10):
            row[f"dummy_{i}"] = random.random()
    return row

# -------------------------------------------------------------------------
# Resilient Request Function
# Uses Tenacity for connection retry with back-off; simulates 3% drop rate
# -------------------------------------------------------------------------

session = requests.Session()  # Connection pooling for lower overhead

@retry(wait=wait_exponential(multiplier=0.05, min=0.05, max=1),
       stop=stop_after_attempt(3))
def safe_post(json_payload):
    """
    Sends an HTTP POST request with retry/back-off logic.
    Introduces artificial 3% packet drop failure rate before sending.
    """
    if random.random() < 0.03:
        raise requests.exceptions.ConnectionError("Injected packet loss")

    simulate_network_delay()
    return session.post(URL, json=json_payload, timeout=5)

# -------------------------------------------------------------------------
# Threaded Load Execution
# Worker threads simulate inference requests and record their results
# -------------------------------------------------------------------------

lock = threading.Lock()
results = []         # Stores tuples: (batch_id, thread_id, status, latency)
failures = {}        # Tracks failure type counts

def send_request(batch_idx, local_tid):
    """
    Thread worker: sends a request and logs the result with latency.
    Records HTTP 200 or error type for later analysis.
    """
    row = DATA.sample(n=1).to_dict(orient="records")[0]
    payload = mutate_payload(row)
    start = time.time()

    try:
        safe_post(payload)
        status = 200
    except Exception as e:
        status = type(e).__name__
        with lock:
            failures[status] = failures.get(status, 0) + 1

    latency = round(time.time() - start, 4)
    with lock:
        results.append((batch_idx, local_tid, status, latency))

# -------------------------------------------------------------------------
# ░ Reload Trigger Daemon ░
# Fires the /reload endpoint every N seconds in the background
# -------------------------------------------------------------------------

def reload_daemon(interval=20):
    """Continuously calls the /reload endpoint to simulate live model swaps."""
    while True:
        try:
            requests.post(RELOAD_URL, timeout=3)
        except Exception:
            pass
        time.sleep(interval)

# -------------------------------------------------------------------------
# ░ Simulation Orchestration ░
# Ramps up load, coordinates threads, dumps CSV and prints metrics
# -------------------------------------------------------------------------

# (threads_per_batch, iterations)
RAMP_STEPS = [
    (3, 200),
    (6, 200),
    (10, 200),
    (25, 200)  # Spike phase
]

def run_load():
    """
    Main simulation runner:
    - Starts the reload daemon
    - Ramps up traffic in burst steps
    - Joins all threads
    - Outputs summary and CSV
    """
    print("\n  If you want **kernel-level** delay/loss add:")
    print("    sudo tc qdisc add dev lo root netem delay 60ms 20ms loss 2%")
    print("    (run again without sudo to remove)\n")

    threading.Thread(target=reload_daemon, daemon=True).start()
    total_reqs = 0
    t0 = time.time()

    for threads_per_batch, iterations in RAMP_STEPS:
        for i in range(iterations):
            thread_batch = []
            for tid in range(threads_per_batch):
                th = threading.Thread(target=send_request, args=(f"{i}", tid))
                thread_batch.append(th)
                th.start()
            for th in thread_batch:
                th.join()
            total_reqs += threads_per_batch

    t1 = time.time()
    dump_results()
    report(total_reqs, t1 - t0)

# -------------------------------------------------------------------------
# Output & Reporting
# Writes results to CSV and prints latency + failure metrics
# -------------------------------------------------------------------------

def dump_results():
    """Saves raw simulation outcomes to CSV."""
    with open("simulation_results.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["Batch", "Thread", "Status", "Latency"])
        wr.writerows(results)

def report(total, duration):
    """Prints final statistics on performance and failure breakdown."""
    latencies = [r[3] for r in results if r[2] == 200]
    print(f"\n {total} requests in {round(duration,2)} s "
          f"({round(total/duration,1)} req/s)")
    print(f"   mean={round(sum(latencies)/len(latencies),4)} s  "
          f"p95={round(sorted(latencies)[int(0.95*len(latencies))-1],4)} s")
    print("Failures:", failures)

# -------------------------------------------------------------------------

if __name__ == "__main__":
    run_load()
