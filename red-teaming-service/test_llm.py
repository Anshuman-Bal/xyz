import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from rta_brain import CustomAzureMLChatModel

# Load the environment variables from .env
load_dotenv()

def main():
    endpoint_url = os.getenv("LLM_Endpoint_URL")
    api_key = os.getenv("LLM_API_Key")

    if not endpoint_url or not api_key:
        print("Error: Missing LLM_Endpoint_URL or LLM_API_Key in .env file.")
        return

    print(f"Testing Azure ML endpoint: {endpoint_url}")
    print(f"API Key present: {bool(api_key)}")

    # Initialize the same CustomAzureMLChatModel we use in our app
    llm = CustomAzureMLChatModel(
        endpoint_url=endpoint_url,
        api_key=api_key,
        model_name="llama3-8b-endpoint",
        temperature=0.7,
    )

    messages = [
        HumanMessage(content="Hello! Are you working? Reply with a short confirmation message if you can read this.")
    ]

    print("\nSending request to LLM...")
    try:
        response = llm.invoke(messages)
        print("\n--- LLM Response ---")
        print(response.content.strip())
        print("--------------------\n")
        print("Success! The LLM endpoint is reachable and responded.")
    except Exception as e:
        print(f"\nFailed to get a response. Error: {e}")

if __name__ == "__main__":
    main()
