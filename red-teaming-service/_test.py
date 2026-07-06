import json
import os
import time
from marshmallow import pprint
import yaml
from typing import List, Dict, Any

def _load_config(config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
config =_load_config("red_teaming/rta_config.yaml")
pprint(config['target_agent']['tools'])