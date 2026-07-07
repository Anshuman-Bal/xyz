import requests
import json
import time

url = "https://llama3-8b-endpoint.eastus.inference.ml.azure.com/score"
headers = {
    "Authorization": "Bearer ",
    "Content-Type": "application/json"
}
payload = {
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 100,
    "temperature": 0.7
}

print("Testing Azure ML endpoint...")
try:
    response = requests.post(url, headers=headers, json=payload)
    print("Status:", response.status_code)
    print("Body:", response.text)
except Exception as e:
    print("Error:", e)
