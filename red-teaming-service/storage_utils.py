"""
storage_utils.py
----------------
Cloud-agnostic storage backend for VectorDB Poisoning Detection.
"""

import io
import json
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List

import boto3
from boto3.s3.transfer import TransferConfig

class StorageBackend(ABC):
    """Base interface for cloud storage."""

    @property
    @abstractmethod
    def bucket_name(self) -> str:
        """Return the container or bucket name for compatibility with existing logging."""
        pass

    @abstractmethod
    def upload_json(self, data: Dict[str, Any], key: str) -> None:
        pass

    @abstractmethod
    def upload_parquet_bytes(self, buffer: io.BytesIO, key: str) -> None:
        pass

    @abstractmethod
    def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: str | None = None,
        multipart_threshold_bytes: int = 64 * 1024 * 1024,
        multipart_chunksize_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        pass

    @abstractmethod
    def download_bytes(self, key: str) -> bytes:
        pass

    @abstractmethod
    def list_keys(self, prefix: str, suffix: str | None = None) -> List[str]:
        pass

    @abstractmethod
    def delete_prefix(self, prefix: str, suffix: str | None = None) -> int:
        pass

class AWSStorageBackend(StorageBackend):
    def __init__(self, bucket_name: str, region_name: str | None = None, aws_access_key_id: str | None = None, aws_secret_access_key: str | None = None):
        self._bucket_name = bucket_name
        if aws_access_key_id and aws_secret_access_key:
            self.client = boto3.client(
                "s3",
                region_name=region_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            self.client = boto3.client("s3", region_name=region_name)

    @property
    def bucket_name(self) -> str:
        return self._bucket_name

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        return prefix.strip("/")

    def upload_json(self, data: Dict[str, Any], key: str) -> None:
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

    def upload_parquet_bytes(self, buffer: io.BytesIO, key: str) -> None:
        buffer.seek(0)
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=key,
            Body=buffer.read(),
            ContentType="application/octet-stream",
        )

    def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: str | None = None,
        multipart_threshold_bytes: int = 64 * 1024 * 1024,
        multipart_chunksize_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        file_size = os.path.getsize(file_path)

        if file_size < multipart_threshold_bytes:
            with open(file_path, "rb") as file_obj:
                self.client.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=file_obj.read(),
                    ContentType=content_type or "application/octet-stream",
                )
            return

        transfer_config = TransferConfig(
            multipart_threshold=multipart_threshold_bytes,
            multipart_chunksize=multipart_chunksize_bytes,
        )
        extra_args = {"ContentType": content_type or "application/octet-stream"}

        with open(file_path, "rb") as file_obj:
            self.client.upload_fileobj(
                Fileobj=file_obj,
                Bucket=self.bucket_name,
                Key=key,
                ExtraArgs=extra_args,
                Config=transfer_config,
            )

    def download_bytes(self, key: str) -> bytes:
        response = self.client.get_object(Bucket=self.bucket_name, Key=key)
        return response["Body"].read()

    def list_keys(self, prefix: str, suffix: str | None = None) -> List[str]:
        normalized = self._normalize_prefix(prefix)
        paginator = self.client.get_paginator("list_objects_v2")
        page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=f"{normalized}/")

        keys: List[str] = []
        for page in page_iterator:
            for item in page.get("Contents", []):
                key = item.get("Key")
                if not key:
                    continue
                if suffix and not key.endswith(suffix):
                    continue
                keys.append(key)

        return sorted(keys)

    def delete_prefix(self, prefix: str, suffix: str | None = None) -> int:
        keys = self.list_keys(prefix, suffix=suffix)
        if not keys:
            return 0

        deleted_count = 0
        for start in range(0, len(keys), 1000):
            chunk = keys[start:start + 1000]
            self.client.delete_objects(
                Bucket=self.bucket_name,
                Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
            )
            deleted_count += len(chunk)

        return deleted_count

class AzureBlobStorageBackend(StorageBackend):
    def __init__(self, container_name: str, account_name: str, account_key: str):
        try:
            from azure.storage.blob import BlobServiceClient, ContentSettings
        except ImportError as exc:
            raise ImportError("azure-storage-blob is required for Azure storage backend") from exc

        self._container_name = container_name

        if not account_name or not account_key:
            raise ValueError("account_name and account_key are required for Azure Blob Storage")

        conn_str = (
            f"DefaultEndpointsProtocol=https;"
            f"AccountName={account_name};"
            f"AccountKey={account_key};"
            f"EndpointSuffix=core.windows.net"
        )
        self.service_client = BlobServiceClient.from_connection_string(conn_str)
        self.container_client = self.service_client.get_container_client(self._container_name)

    @property
    def bucket_name(self) -> str:
        return self._container_name

    @staticmethod
    def _normalize_prefix(prefix: str) -> str:
        return prefix.strip("/")

    def upload_json(self, data: Dict[str, Any], key: str) -> None:
        from azure.storage.blob import ContentSettings
        body = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        blob_client = self.container_client.get_blob_client(key)
        blob_client.upload_blob(
            body,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json")
        )

    def upload_parquet_bytes(self, buffer: io.BytesIO, key: str) -> None:
        from azure.storage.blob import ContentSettings
        buffer.seek(0)
        blob_client = self.container_client.get_blob_client(key)
        blob_client.upload_blob(
            buffer.read(),
            overwrite=True,
            content_settings=ContentSettings(content_type="application/octet-stream")
        )

    def upload_file(
        self,
        file_path: str,
        key: str,
        content_type: str | None = None,
        multipart_threshold_bytes: int = 64 * 1024 * 1024,
        multipart_chunksize_bytes: int = 8 * 1024 * 1024,
    ) -> None:
        from azure.storage.blob import ContentSettings
        blob_client = self.container_client.get_blob_client(key)
        with open(file_path, "rb") as file_obj:
            blob_client.upload_blob(
                file_obj,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type or "application/octet-stream"),

                max_concurrency=4
            )

    def download_bytes(self, key: str) -> bytes:
        blob_client = self.container_client.get_blob_client(key)
        stream = blob_client.download_blob()
        return stream.readall()

    def list_keys(self, prefix: str, suffix: str | None = None) -> List[str]:
        normalized = self._normalize_prefix(prefix)

        search_prefix = f"{normalized}/" if normalized else ""

        keys: List[str] = []
        blobs = self.container_client.list_blobs(name_starts_with=search_prefix)
        for blob in blobs:
            if suffix and not blob.name.endswith(suffix):
                continue
            keys.append(blob.name)

        return sorted(keys)

    def delete_prefix(self, prefix: str, suffix: str | None = None) -> int:
        keys = self.list_keys(prefix, suffix=suffix)
        if not keys:
            return 0

        deleted_count = 0
        for key in keys:
            blob_client = self.container_client.get_blob_client(key)
            blob_client.delete_blob()
            deleted_count += 1

        return deleted_count

def get_storage_backend(
    cloud_provider: str,
    container_name: str,
    account_name: str | None = None,
    account_key: str | None = None,
    region: str | None = None,
    aws_access_key_id: str | None = None,
    aws_secret_access_key: str | None = None
) -> StorageBackend:
    """Factory to return the appropriate storage backend based on cloud provider."""
    provider = cloud_provider.strip().lower()
    if provider == "aws":
        return AWSStorageBackend(
            bucket_name=container_name,
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
    elif provider == "azure":
        return AzureBlobStorageBackend(
            container_name=container_name,
            account_name=account_name or "",
            account_key=account_key or ""
        )
    else:
        raise ValueError(f"Unsupported cloud provider: {cloud_provider}")
