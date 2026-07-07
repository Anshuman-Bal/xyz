import os
import requests

def test():
    endpoint_url = "https://llama3-8b-endpoint.eastus.inference.ml.azure.com/score"
    api_key = ""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [
            {"role": "user", "content": "Please generate a prompt."}
        ],
        "max_tokens": 256,
        "temperature": 0.7,
    }

    response = requests.post(endpoint_url, headers=headers, json=payload)
    print(f"First request status: {response.status_code}")
    print(f"First request response: {response.text}")

if __name__ == '__main__':
    test()
