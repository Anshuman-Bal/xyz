import json
from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, Field

from secret_manager import (
    SecretPayloadMixin,
    SecretName,
    resolve_credentials,
    resolve_storage_credentials,
    CLOUD_PROVIDER,
)
from storage_utils import get_storage_backend

app = FastAPI(
    title="System Prompt Extraction API",
    description="Extracts the agent's system prompt from cloud storage artifacts."
)

class SystemPromptRequest(SecretPayloadMixin):
    model_config = ConfigDict(populate_by_name=True)
    secretJson: SecretName = Field(
        default_factory=lambda: SecretName(useCredentialStore=False, params=[])
    )
    artifacts_prefix: str = Field(
        default="telemetry-logs/artifacts",
        description="Prefix path to the agent's artifacts folder"
    )

@app.post("/system-prompt")
def extract_system_prompt(req: SystemPromptRequest):
    provider = CLOUD_PROVIDER
    creds = resolve_credentials(req.secretJson, require_keys=[])
    creds.update(resolve_storage_credentials("performanceMonitoring"))
    
    try:
        if provider == "aws":
            bucket = creds.get("s3_bucket")
            access_key = creds.get("s3_access_key_id")
            secret_key = creds.get("s3_secret_access_key")
            if not bucket:
                raise HTTPException(500, "Missing AWS s3_bucket in credentials")
                
            storage = get_storage_backend(
                cloud_provider="aws", 
                container_name=bucket, 
                aws_access_key_id=access_key, 
                aws_secret_access_key=secret_key
            )
        elif provider == "azure":
            container = creds.get("blob-container")
            account = creds.get("storage-account")
            key = creds.get("storage-key")
            if not container:
                raise HTTPException(500, "Missing Azure blob-container in credentials")
                
            storage = get_storage_backend(
                cloud_provider="azure",
                container_name=container,
                account_name=account,
                account_key=key
            )
        else:
            raise HTTPException(500, f"Unsupported cloud provider {provider}")
    except Exception as e:
        raise HTTPException(500, f"Failed to initialize storage backend: {str(e)}")
        
    try:
        keys = storage.list_keys("", suffix="_llm_prompt.json")
        if req.artifacts_prefix:
            keys = [k for k in keys if req.artifacts_prefix in k]
    except Exception as e:
        raise HTTPException(500, f"Failed to list objects in storage: {str(e)}")

    if not keys:
        raise HTTPException(404, "No system prompt artifact found")
        
    try:
        data_bytes = storage.download_bytes(keys[0])
        data = json.loads(data_bytes.decode("utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to download or parse artifact: {str(e)}")
    
    return {"system_prompt": data.get("prompt", "")}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8007, reload=True)
