import time
import httpx

URL = "http://127.0.0.1:8000/predict"
PAYLOAD = {"text": "Hello world"}

N_REQUESTS = 20

latencies = []

with httpx.Client(timeout=30.0) as client:
    for i in range(N_REQUESTS):
        start = time.time()
        r = client.post(URL, json=PAYLOAD)
        elapsed = time.time() - start
        latencies.append(elapsed)

        assert r.status_code == 200

avg_latency = sum(latencies) / len(latencies)

print(f"Total requests: {N_REQUESTS}")
print(f"Average latency: {avg_latency:.3f} seconds")
print(f"Min latency: {min(latencies):.3f} seconds")
print(f"Max latency: {max(latencies):.3f} seconds")
