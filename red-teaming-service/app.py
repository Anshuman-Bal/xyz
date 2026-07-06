import uuid
import logging
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
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
    attack_selection: Dict[str, Any] = Field(..., description="Config matching the attack_selection block from YAML.")
    
    # Optional overrides for advanced usage
    campaign_settings: Optional[Dict[str, Any]] = Field(
        default={
            "mutation": {"enabled": False, "modes": {}},
            "chaining": {"enabled": False}
        },
        description="Optional mutation and chaining settings."
    )

class RedTeamResponse(BaseModel):
    message: str
    run_id: str
    total_attack_definitions: int
    total_attack_runs: int
    status_summary: dict
    attack_definitions: list
    attack_runs: list

# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/run-campaign", response_model=RedTeamResponse)
def run_campaign(req: RedTeamRequest):
    """
    Run an end-to-end red teaming campaign statelessly.
    The attacks are generated, executed, and evaluated in-memory.
    """
    try:
        run_id = str(uuid.uuid4())
        logger.info(f"Starting Red Team Campaign {run_id} targeting {req.target_endpoint}")

        # Construct the config dictionary that rta_brain expects
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
                "categories": req.attack_selection
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

        return RedTeamResponse(
            message="Campaign finished.",
            run_id=run_id,
            total_attack_definitions=len(results.get("attack_definitions", [])),
            total_attack_runs=len(results.get("attack_runs", [])),
            status_summary=status_counts,
            attack_definitions=results.get("attack_definitions", []),
            attack_runs=results.get("attack_runs", [])
        )

    except Exception as e:
        logger.error(f"Error during campaign execution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
