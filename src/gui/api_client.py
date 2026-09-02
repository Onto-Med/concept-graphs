"""Small HTTP client used by the Streamlit GUI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class APIError(Exception):
    """Raised when the API returns a non-success response."""

    status_code: int
    message: str
    payload: Any | None = None

    def __str__(self) -> str:
        return f"HTTP {self.status_code}: {self.message}"


class ConceptGraphsClient:
    """Thin client around the Concept Graphs Flask API."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _handle(self, response: requests.Response) -> Any:
        content_type = response.headers.get("Content-Type", "")
        payload: Any
        if "application/json" in content_type:
            payload = response.json()
        else:
            payload = response.text
        if response.status_code >= 400:
            if isinstance(payload, dict):
                message = payload.get("error") or payload.get("message") or json.dumps(payload)
            else:
                message = str(payload)
            raise APIError(response.status_code, message, payload)
        return payload

    def health(self) -> Any:
        return self._handle(requests.get(self._url("/openapi"), timeout=self.timeout))

    def list_processes(self) -> Any:
        return self._handle(requests.get(self._url("/processes"), timeout=self.timeout))

    def status(self, process: str) -> Any:
        return self._handle(
            requests.get(self._url("/status"), params={"process": process}, timeout=self.timeout)
        )

    def stop_process(self, process: str, hard_stop: bool = False) -> Any:
        return self._handle(
            requests.get(
                self._url(f"/processes/{process}/stop"),
                params={"hard_stop": str(hard_stop).lower()},
                timeout=self.timeout,
            )
        )

    def delete_process(self, process: str, hard_stop: bool = False) -> Any:
        return self._handle(
            requests.delete(
                self._url(f"/processes/{process}/delete"),
                params={"hard_stop": str(hard_stop).lower()},
                timeout=self.timeout,
            )
        )

    def pipeline_configuration(self, process: str, language: str, default: bool = True) -> Any:
        return self._handle(
            requests.get(
                self._url("/pipeline/configuration"),
                params={
                    "process": process,
                    "language": language,
                    "default": str(default).lower(),
                },
                timeout=self.timeout,
            )
        )

    def start_pipeline_json(
        self,
        process: str,
        language: str,
        config: dict[str, Any],
        skip_present: bool = True,
        skip_steps: list[str] | None = None,
        return_statistics: bool = False,
    ) -> Any:
        return self._handle(
            requests.post(
                self._url("/pipeline"),
                params={
                    "process": process,
                    "language": language,
                    "skip_present": str(skip_present).lower(),
                    "skip_steps": ",".join(skip_steps or []),
                    "return_statistics": str(return_statistics).lower(),
                },
                json=config,
                timeout=self.timeout,
            )
        )

    def start_pipeline_upload(
        self,
        process: str,
        language: str,
        data_file: Any,
        config_files: dict[str, Any],
        skip_present: bool = True,
        skip_steps: list[str] | None = None,
        return_statistics: bool = False,
    ) -> Any:
        files = {"data": data_file}
        files.update({k: v for k, v in config_files.items() if v is not None})
        return self._handle(
            requests.post(
                self._url("/pipeline"),
                params={
                    "process": process,
                    "language": language,
                    "skip_present": str(skip_present).lower(),
                    "skip_steps": ",".join(skip_steps or []),
                    "return_statistics": str(return_statistics).lower(),
                },
                files=files,
                timeout=None,
            )
        )

    def artifact(self, path: str, process: str, **params: Any) -> Any:
        query = {"process": process, **params}
        return self._handle(requests.get(self._url(path), params=query, timeout=self.timeout))

    def graph_html(self, process: str, graph_id: int) -> str:
        response = requests.get(
            self._url(f"/graph/{graph_id}"),
            params={"process": process, "draw": "true"},
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise APIError(response.status_code, response.text, response.text)
        return response.text

    def init_rag(self, process: str, config: dict[str, Any], force: bool = False) -> Any:
        return self._handle(
            requests.post(
                self._url("/rag/init"),
                params={"process": process, "force": str(force).lower()},
                json=config,
                timeout=self.timeout,
            )
        )

    def ask_rag(
        self, process: str, question: str, doc_ids: list[str] | None = None, limit: int = 15
    ) -> Any:
        body = {"doc_ids": doc_ids or [], "limit": limit}
        return self._handle(
            requests.post(
                self._url("/rag/question"),
                params={"process": process, "q": question},
                json=body,
                timeout=None,
            )
        )

    def expand_query(
        self, payload: dict[str, Any], api_key: str | None = None, auth_header: str = "X-LLM-API-Key"
    ) -> Any:
        headers = {auth_header: api_key} if api_key else None
        return self._handle(
            requests.post(
                self._url("/query-expansion"),
                json=payload,
                headers=headers,
                timeout=None,
            )
        )
