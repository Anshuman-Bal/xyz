import json
import urllib.request
import ssl

def test_azure_llm():
    url = 'https://llama3-8b-endpoint.eastus.inference.ml.azure.com/score'
    api_key = ''

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}'
    }

    # Azure ML managed endpoints for Llama 3 generally support the OpenAI chat/completions format
    data = {
        "messages": [
            {"role": "user", "content": "Hello, are you working? Please respond with 'Yes'."}
        ],
        "max_tokens": 50,
        "temperature": 0.1
    }

    body = str.encode(json.dumps(data))
    
    req = urllib.request.Request(url, body, headers)
    
    try:
        response = urllib.request.urlopen(req)
        result = response.read()
        print("Success! Endpoint is working.")
        print("Response:")
        print(json.dumps(json.loads(result.decode("utf-8")), indent=2))
    except urllib.error.HTTPError as error:
        print("The request failed with status code: " + str(error.code))
        print(error.info())
        print(error.read().decode("utf8", 'ignore'))

if __name__ == "__main__":
    test_azure_llm()
