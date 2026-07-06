from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import boto3
import yaml
from fastapi import HTTPException
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource


# ---------------------------------------------------------------------------
# application.yaml → Pydantic Settings
# ---------------------------------------------------------------------------

_YAML_PATH: Path = Path(__file__).resolve().parent / "application.yaml"


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Custom Pydantic settings source that reads from application.yaml at the repo root."""

    def __call__(self) -> dict[str, Any]:
        if not _YAML_PATH.exists():
            return {}
        try:
            with _YAML_PATH.open("r", encoding="utf-8") as fh:
                return yaml.safe_load(fh) or {}
        except Exception:
            return {}

    def get_field_value(self, field: Any, field_name: str) -> Any:  # type: ignore[override]
        return None, field_name, False


# ---------------------------------------------------------------------------
# Typed models for application.yaml structure
# ---------------------------------------------------------------------------

class ToggleSettings(BaseModel):
    useCloud: str = ""


class StorageBlock(BaseModel):
    secretName: str = ""
    s3: list[str] = Field(default_factory=list)
    azureBlob: list[str] = Field(default_factory=list)


class StoragePathParams(BaseModel):
    bedRockCustomDiscovery: StorageBlock = Field(default_factory=StorageBlock)
    sddDetection: StorageBlock = Field(default_factory=StorageBlock)
    performanceMonitoring: StorageBlock = Field(default_factory=StorageBlock)
    vectordbPoisoning: StorageBlock = Field(default_factory=StorageBlock)
    opensearchAnomalyDetection: StorageBlock = Field(default_factory=StorageBlock)
    redTeaming: StorageBlock = Field(default_factory=StorageBlock)


class AppSettings(BaseSettings):
    toggle: ToggleSettings = Field(default_factory=ToggleSettings)
    storagePathParams: StoragePathParams = Field(default_factory=StoragePathParams)

    model_config = {"arbitrary_types_allowed": True}

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        **kwargs: Any,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (YamlSettingsSource(settings_cls),)


# Resolved once at startup / import time
app_settings: AppSettings = AppSettings()
CLOUD_PROVIDER: str = app_settings.toggle.useCloud.strip().lower()


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class SecretParam(BaseModel):
    """A single credential parameter entry."""
    name: str | None = None          # logical name used in the creds dict — auto-assigned from connectionType if not provided
    key: str                         # secret store key / Azure Key Vault secret name
    value: str | None = None         # plain-text value when not using the credential store


# Single schema per connectionType:
#   "all"      → ordered list of internal param names (required first, then optional)
#   "required" → how many of the first names are mandatory
_CONNECTION_TYPE_SCHEMA: dict[str, dict] = {
    "accesskey": {
        "all":      ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"],
        "required": 2,   # aws_session_token is optional
    },
    "iam": {
        "all":      ["role_arn", "external_id", "role_session_name"],  #yet to finalised 
        "required": 1,   # external_id and role_session_name are optional
    },
    "databricks": {
        "all":      ["databricks_token", "databricks_host"],
        "required": 2,
    },
    "credentials": {
        "all":      ["databricks_token", "databricks_host"],
        "required": 2,
    },
    "service_principal": {
        "all":      ["tenant_id", "client_id", "client_secret", "subscription_id"],
        "required": 4,
    },
    "azurenative": {
        "all":      ["tenant_id", "client_id", "client_secret", "subscription_id"],
        "required": 4,
    },
    "pat": {
        "all":      ["snowflake_account_url", "snowflake_pat"],
        "required": 2,
    },
    "service_account_gcp": {
        "all":      ["GCP_SA_CLIENT_EMAIL", "GCP_SA_PRIVATE_KEY", "GCP_SA_PROJECT_ID", "ORGANIZATION_ID"],
        "required": 4,
    },
    "api_key": {
        "all": ["apiKey"],
        "required": 1,
    },
    "basicauth": {
        "all": ["host", "username", "password"],
        "required": 0,
    }
}


def get_required_keys(secret: "SecretName") -> list[str]:
    """
    Return the list of credential keys that MUST be present after resolution,
    derived from the connectionType in the payload.
    Returns an empty list for unknown connection types (no forced validation).
    """
    connection = secret.connectionType.strip().lower()
    schema = _CONNECTION_TYPE_SCHEMA.get(connection)
    if schema is None:
        return []
    return schema["all"][:schema["required"]]


class SecretName(BaseModel):
    useCredentialStore: bool = True
    secretId: str | None = None  # AWS Secrets Manager secret ID or Azure Key Vault name
    connectionType: str = "AccessKey"
    params: list[SecretParam] = Field(default_factory=list)


class SecretPayloadMixin(BaseModel):
    secretJson: SecretName

    @model_validator(mode="before")
    @classmethod
    def parse_secret_json_string(cls, data: Any) -> Any:
        # Automatically parse 'secretJson' if the client sent it as a JSON string instead of an object
        if isinstance(data, dict):
            sj = data.get("secretJson")
            if isinstance(sj, str):
                try:
                    data["secretJson"] = json.loads(sj)
                except json.JSONDecodeError:
                    pass # Let Pydantic throw the standard validation error
        return data


# ---------------------------------------------------------------------------
# Credential resolution helpers
# ---------------------------------------------------------------------------

def _resolve_aws_from_secrets_manager(secret_id: str) -> dict[str, str]:
    """
    Fetch a JSON secret from AWS Secrets Manager.
    Uses the IAM role / ambient credentials of the process to authenticate.
    Returns a flat dict of key→value pairs.
    """
    sm = boto3.client("secretsmanager")
    try:
        resp = sm.get_secret_value(SecretId=secret_id)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"AWS Secrets Manager error for '{secret_id}': {exc}",
        )
    raw = resp.get("SecretString") or ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {secret_id: raw}


def _resolve_azure_secret(secret_id: str, secret_key: str) -> str:
    """
    Fetch a single secret from Azure Key Vault using the official SDK.
    Builds the vault URL from secret_id (e.g. 'grafyn-bedrockagent' → https://grafyn-bedrockagent.vault.azure.net)
    and authenticates using DefaultAzureCredential (az login, env vars, managed identity, etc.).
    Raises HTTPException(502) on any failure, including empty/disabled secrets.
    """
    try:
        from azure.identity import AzureCliCredential
        from azure.keyvault.secrets import SecretClient

        keyvault_url = f"https://{secret_id}.vault.azure.net"
        credential = AzureCliCredential()
        client = SecretClient(vault_url=keyvault_url, credential=credential)
        retrieved = client.get_secret(secret_key)
        value = retrieved.value
        if not value:
            raise HTTPException(
                status_code=502,
                detail=f"Azure Key Vault secret '{secret_key}' in vault '{secret_id}' is empty or disabled.",
            )
        return value
    except HTTPException:
        raise  # re-raise our own HTTPExceptions unchanged
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail="azure-identity and azure-keyvault-secrets packages are required. Run: pip install azure-identity azure-keyvault-secrets",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Azure Key Vault error for secret '{secret_key}' in vault '{secret_id}': {exc}",
        )


def _parse_literal_item(item: str) -> tuple[str, str] | None:
    """
    If an item from the YAML list contains '=', treat it as a literal key=value pair.
    Returns (key, value) or None if no '=' found.
    e.g. 's3_bucket = "qa-grafyn.ai"' → ('s3_bucket', 'qa-grafyn.ai')
    """
    if "=" in item:
        k, v = item.split("=", 1)
        return k.strip(), v.strip().strip('"').strip("'")
    return None


def _assign_param_names_from_connection_type(secret: SecretName) -> list[SecretParam]:
    """
    Look at connectionType and assign the correct internal `name` to each param
    based on its position in the list.

    - AccessKey → [aws_access_key_id, aws_secret_access_key, (aws_session_token)]
    - IAM       → [role_arn, (external_id), (role_session_name)]

    If a param already has a `name` set by the user, it is left unchanged.
    """
    connection = secret.connectionType.strip().lower()
    schema = _CONNECTION_TYPE_SCHEMA.get(connection)

    if schema is None:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown connectionType '{secret.connectionType}'. Supported values: AccessKey, IAM, Databricks, Credentials, service_principal, BasicAuth",
        )

    all_names = schema["all"]
    updated: list[SecretParam] = []
    for i, param in enumerate(secret.params):
        if param.name:                        # user already provided a name — respect it
            updated.append(param)
        elif i < len(all_names):              # assign name from the schema
            updated.append(param.model_copy(update={"name": all_names[i]}))
        else:
            updated.append(param)             # extra params beyond the schema — keep as-is
    return updated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_credentials(secret: SecretName, require_keys: list[str] | None = None) -> dict[str, str]:
    """
    Resolve credentials from the API request payload.
    - useCredentialStore=true  → fetches from AWS Secrets Manager or Azure Key Vault
    - useCredentialStore=false → reads inline `value` fields from params
    """
    if require_keys is None:
        require_keys = get_required_keys(secret)
    # Step 1: Assign param names from connectionType (for any params missing a name)
    params = _assign_param_names_from_connection_type(secret)

    # Filter out job id and volume path params so they are ignored and don't cause lookup/validation errors
    ignored_keys = {"discovery-job-id", "volume-path"}
    ignored_names = {"discovery_job_id", "volume_path"}
    params = [
        p for p in params
        if (p.key or "").strip().lower() not in ignored_keys
        and (p.name or "").strip().lower().replace("-", "_") not in ignored_names
    ]

    provider = CLOUD_PROVIDER

    # --- DEBUG ---
    print(f"[DEBUG] useCredentialStore={secret.useCredentialStore}")
    print(f"[DEBUG] connectionType={secret.connectionType!r}  provider={provider!r}")
    print(f"[DEBUG] secretId={secret.secretId!r}")
    print(f"[DEBUG] params after filter: {[(p.name, p.key) for p in params]}")
    print(f"[DEBUG] require_keys={require_keys}")
    # -------------

    creds: dict[str, str] = {}

    if secret.useCredentialStore:
        if provider == "aws":
            if not secret.secretId:
                raise HTTPException(
                    status_code=422,
                    detail="secretJson.secretId is required when useCredentialStore=true (cloud: aws)",
                )
            kv = _resolve_aws_from_secrets_manager(secret.secretId)
            print(f"[DEBUG] AWS secret keys returned: {list(kv.keys())}")
            # Build a normalised lookup: lowercase + underscores, so we match regardless of
            # how the key is stored in AWS (DATABRICKS_TOKEN / databricks-token / databricks_token)
            kv_norm = {k.lower().replace("-", "_"): v for k, v in kv.items()}
            for param in params:
                if not param.name:
                    continue
                # Try normalised variants of both key and name fields
                candidates = [
                    (param.key or "").lower().replace("-", "_"),
                    (param.name or "").lower().replace("-", "_"),
                ]
                val = next((kv_norm[c] for c in candidates if c in kv_norm), None)
                print(f"[DEBUG] AWS lookup for param.key={param.key!r} -> {'found' if val else 'NOT FOUND'}")
                if val is not None:
                    creds[param.name.lower().replace("-", "_")] = str(val)


        elif provider == "azure":
            if not secret.secretId:
                raise HTTPException(
                    status_code=422,
                    detail="secretJson.secretId is required to identify the Azure Key Vault when useCredentialStore=true (cloud: azure)",
                )
            for param in params:
                if not param.name:                    # skip params with no resolved name
                    continue
                val = _resolve_azure_secret(secret.secretId, param.key)
                print(f"[DEBUG] Azure resolved '{param.key}' → '{val[:4]}...' (len={len(val)})" if val else f"[DEBUG] Azure returned empty for '{param.key}'")
                creds[param.name.lower().replace("-", "_")] = val  # normalise key: lowercase + hyphens→underscores

        else:
            raise HTTPException(
                status_code=422,
                detail=f"Unsupported cloud provider in application.yaml toggle.useCloud: '{provider}'",
            )

    else:
        # Inline values: use name-assigned params so name is always populated
        for param in params:
            if param.name and param.value is not None:
                creds[param.name.lower().replace("-", "_")] = param.value  # normalise key: lowercase + hyphens→underscores

    print(f"[DEBUG] creds keys resolved: {list(creds.keys())}")

    if require_keys:
        # All creds keys are already lowercase — simple lookup
        missing = [k for k in require_keys if not creds.get(k.lower())]
        if missing:
            raise HTTPException(
                status_code=422,
                detail=f"Missing required credential(s): {missing}",
            )

    return creds



def resolve_storage_credentials(service: str) -> dict[str, str]:
    """
    Read storage credentials from application.yaml's storagePathParams block using
    the typed AppSettings object (no manual YAML parsing needed).

    Supports a mix of literal values (key="value") and cloud-fetched secrets.
    """
    # Use the typed settings object — no raw yaml.safe_load needed
    storage_block: StorageBlock = getattr(app_settings.storagePathParams, service, StorageBlock())
    secret_name = storage_block.secretName
    if not secret_name:
        return {}

    provider = CLOUD_PROVIDER
    creds: dict[str, str] = {}

    if provider == "aws":
        keys_to_fetch = storage_block.s3
        if not keys_to_fetch:
            return {}
        # Fetch entire secret once, then pluck only the requested keys
        try:
            kv = _resolve_aws_from_secrets_manager(secret_name)
        except Exception:
            kv = {}

        for item in keys_to_fetch:
            parsed = _parse_literal_item(item)
            if parsed:
                creds[parsed[0]] = parsed[1]
            else:
                val = kv.get(item.strip())
                if val is not None:
                    creds[item.strip()] = str(val)

    elif provider == "azure":
        keys_to_fetch = storage_block.azureBlob
        if not keys_to_fetch:
            return {}

        for item in keys_to_fetch:
            parsed = _parse_literal_item(item)
            if parsed:
                creds[parsed[0]] = parsed[1]
            else:
                try:
                    val = _resolve_azure_secret(secret_name, item.strip())
                    creds[item.strip()] = val
                except Exception as e:
                    print(f"Failed to fetch {item.strip()} from Azure KV: {e}")
                    pass

    return creds
