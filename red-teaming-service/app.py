import uuid
import logging
import os
import yaml
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rta_brain import RedTeamBrain
from rta_executor import RedTeamExecutor
from secret_manager import resolve_storage_credentials, CLOUD_PROVIDER
from storage_utils import get_storage_backend

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("red_teaming_api")

app = FastAPI(
    title="Red Teaming Service API",
    description="Stateless API for running red-teaming campaigns against agent endpoints.",
    version="1.0.0",
)

# ── Models ───────────────────────────────────────────────────────────────────

class RedTeamRequest(BaseModel):
    target_endpoint: str = Field(..., description="The HTTP endpoint of the victim agent to attack.")
    system_prompt: str = Field(..., description="The system prompt of the victim agent.")
    attack_selection: List[str] = Field(..., description="List of attack categories to perform.")
    telemetry_id: str = Field(..., description="Telemetry ID used to store final results.")
    
    # Optional overrides for advanced usage
    campaign_settings: Optional[Dict[str, Any]] = Field(
        default={
            "mutation": {"enabled": False, "modes": {}},
            "chaining": {"enabled": False}
        },
        description="Optional mutation and chaining settings."
    )

class GenericHttpResponse(BaseModel):
    status: str = Field(..., description="Success or Failed", example="SUCCESS")
    message: str = Field(..., description="Message")
    data: Optional[Any] = Field(default=None, description="JSON Output")
    additionalData: Optional[Any] = Field(default=None, description="Additional Data")

# ── Endpoints ────────────────────────────────────────────────────────────────

def _run_campaign_background(req: RedTeamRequest, run_id: str):
    try:
        # Construct the config dictionary that rta_brain expects
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rta_config.yaml")
        yaml_categories = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                yaml_config = yaml.safe_load(f) or {}
                yaml_categories = yaml_config.get("attack_selection", {}).get("categories", {})

        selected_categories = {}
        for category in req.attack_selection:
            if category in yaml_categories:
                cat_config = yaml_categories[category].copy()
                cat_config["enabled"] = True
                selected_categories[category] = cat_config
            else:
                selected_categories[category] = {"enabled": True}

        config = {
            "target_agent": {
                "endpoint": req.target_endpoint,
                "agent_id": "api-target",
                "tools": []
            },
            "attack_library": {
                "directory": "attack_library"
            },
            "campaign_settings": req.campaign_settings,
            "attack_selection": {
                "categories": selected_categories
            }
        }

        # Initialize Brain & Executor
        brain = RedTeamBrain(config=config, system_prompt=req.system_prompt)
        executor = RedTeamExecutor()

        results = executor.run_full_test_plan(brain)
        
        # Save to storage
        try:
            storage_creds = resolve_storage_credentials("redTeaming")
            location = storage_creds.get("location")
            if not location:
                raise ValueError("Storage location not found in application.yaml")

            backend = None
            if CLOUD_PROVIDER == "aws":
                backend = get_storage_backend(
                    cloud_provider="aws",
                    container_name=location,
                    region=storage_creds.get("region"),
                    aws_access_key_id=storage_creds.get("s3_access_key_id"),
                    aws_secret_access_key=storage_creds.get("s3_secret_access_key")
                )
            elif CLOUD_PROVIDER == "azure":
                backend = get_storage_backend(
                    cloud_provider="azure",
                    container_name=location,
                    account_name=storage_creds.get("blob-storage-account"),
                    account_key=storage_creds.get("blob-storage-key")
                )
            
            if backend:
                path_prefix = storage_creds.get("path_prefix", "").strip("/")
                if path_prefix:
                    key = f"{path_prefix}/{req.telemetry_id}.json"
                else:
                    key = f"{req.telemetry_id}.json"
                    
                backend.upload_json(results, key)
                logger.info(f"Results saved to {CLOUD_PROVIDER} at {key}")
            else:
                logger.warning("No backend configured to save results.")
                
        except Exception as storage_ex:
            logger.error(f"Failed to save results to storage: {storage_ex}")

        logger.info(f"Campaign {run_id} finished successfully.")

    except Exception as e:
        logger.error(f"Error during campaign execution: {e}", exc_info=True)


@app.post("/run-campaign", response_model=GenericHttpResponse)
def run_campaign(req: RedTeamRequest, background_tasks: BackgroundTasks):
    """
    Run an end-to-end red teaming campaign statelessly.
    The attacks are generated, executed, and evaluated in the background.
    """
    try:
        run_id = str(uuid.uuid4())
        logger.info(f"Starting Red Team Campaign {run_id} targeting {req.target_endpoint}")

        background_tasks.add_task(_run_campaign_background, req, run_id)

        return GenericHttpResponse(
            status="SUCCESS",
            message="Red teaming started",
            data={
                "run_id": run_id,
                "telemetry_id": req.telemetry_id
            }
        )

    except Exception as e:
        logger.error(f"Error initiating campaign: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "FAILED",
                "message": str(e),
                "data": None,
                "additionalData": None
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
