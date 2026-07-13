import uuid
import logging
import os
import yaml
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rta_brain import RedTeamBrain
from rta_executor import RedTeamExecutor

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
    
    api_schema: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional API JSON schema expected by the target endpoint. If omitted, the attacker LLM will attempt to infer the schema from the target's system prompt."
    )

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
                "tools": [],
                "api_schema": req.api_schema
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

        # Run campaign
        # (This runs synchronously, but since it's a test/PoC service, that's fine. 
        # For full async, we would use await asyncio.to_thread(executor.run_full_test_plan, brain))
        results = executor.run_full_test_plan(brain)

        # Build summary
        status_counts = {}
        for run in results.get("attack_runs", []):
            category = run.get("category", "UNKNOWN_CATEGORY")
            status = run.get("result_status", "UNKNOWN_STATUS")
            
            if category not in status_counts:
                status_counts[category] = {"VULNERABLE": 0, "SAFE": 0, "ERROR": 0}
            if status in status_counts[category]:
                status_counts[category][status] += 1
            else:
                status_counts[category][status] = 1

        logger.info(f"Campaign {run_id} finished successfully.")

        return GenericHttpResponse(
            status="SUCCESS",
            message="Campaign finished successfully.",
            data={
                "run_id": run_id,
                "total_attack_definitions": len(results.get("attack_definitions", [])),
                "total_attack_runs": len(results.get("attack_runs", [])),
                "status_summary": status_counts,
                "attack_definitions": results.get("attack_definitions", []),
                "attack_runs": results.get("attack_runs", [])
            }
        )

    except Exception as e:
        logger.error(f"Error during campaign execution: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "status": "FAILED",
                "message": str(e),
                "data": None,
                "additionalData": None
            }
        )

@app.post("/run_campaign")
async def run_campaign(req: RedTeamRequest):
    run_id = str(uuid.uuid4())
    return _run_campaign_background(req, run_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
