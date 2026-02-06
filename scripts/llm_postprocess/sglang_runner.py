from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class SGLangServerConfig:
    model_path: str
    backend: str = "sglang"
    tp_size: int = 1
    pp_size: int = 1
    dtype: str = "auto"
    quantization: str | None = None
    trust_remote_code: bool = False
    mem_fraction_static: float | None = None
    max_running_requests: int | None = None
    port: int = 30000
    host: str = "127.0.0.1"
    extra_args_json: str = "{}"
    request_timeout_s: float = 120.0
    max_parallel_http: int = 16


class VLLMEndpointBackend:
    backend_type = "vllm"

    def __init__(
        self,
        *,
        base_url: str,
        model_path: str,
        api_key: str | None,
        request_timeout_s: float,
        max_parallel_http: int,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_path = model_path
        self.request_timeout_s = float(request_timeout_s)
        self.max_parallel_http = max(1, int(max_parallel_http))
        self._session = requests.Session()
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})
        self._session.headers.update({"Content-Type": "application/json"})

        if self.base_url.endswith("/v1/chat/completions"):
            self.chat_url = self.base_url
        elif self.base_url.endswith("/v1"):
            self.chat_url = self.base_url + "/chat/completions"
        else:
            self.chat_url = self.base_url + "/v1/chat/completions"

    def run_batch(
        self,
        batch: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        json_schema: str | None,
    ) -> list[dict[str, str]]:
        schema_obj: dict[str, Any] | None = None
        if json_schema:
            try:
                schema_obj = json.loads(json_schema)
            except Exception:
                schema_obj = None

        def _run_one(item: dict[str, str]) -> dict[str, str]:
            payload: dict[str, Any] = {
                "model": self.model_path,
                "messages": [
                    {"role": "system", "content": item["system_prompt"]},
                    {"role": "user", "content": item["user_prompt"]},
                ],
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_new_tokens),
            }
            if schema_obj is not None:
                payload["guided_json"] = schema_obj

            resp = self._session.post(
                self.chat_url,
                json=payload,
                timeout=self.request_timeout_s,
            )
            resp.raise_for_status()
            body = resp.json()
            content = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            return {"raw": str(content)}

        outputs: list[dict[str, str]] = [{"raw": ""} for _ in batch]
        workers = min(self.max_parallel_http, len(batch))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_run_one, batch[idx]): idx for idx in range(len(batch))
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    outputs[idx] = future.result()
                except Exception as e:
                    outputs[idx] = {"raw": "", "error": str(e)}
        return outputs

    def shutdown(self) -> None:
        self._session.close()


def _maybe_parse_json(raw: str | None) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    if not raw:
        return extra
    try:
        extra = json.loads(raw)
    except Exception:
        extra = {}
    return extra


def make_backend(
    *,
    sglang_url: str | None,
    api_key: str | None,
    server: SGLangServerConfig,
):
    backend_kind = (server.backend or "sglang").lower()
    endpoint_url = sglang_url or f"http://{server.host}:{int(server.port)}"

    if backend_kind == "vllm":
        return VLLMEndpointBackend(
            base_url=endpoint_url,
            model_path=server.model_path,
            api_key=api_key,
            request_timeout_s=server.request_timeout_s,
            max_parallel_http=server.max_parallel_http,
        )

    import sglang as sgl
    from sglang.lang.backend.runtime_endpoint import RuntimeEndpoint

    if sglang_url:
        return RuntimeEndpoint(base_url=endpoint_url, api_key=api_key)

    extra = _maybe_parse_json(server.extra_args_json)
    return sgl.Runtime(
        model_path=server.model_path,
        tp_size=int(server.tp_size),
        pp_size=int(server.pp_size),
        dtype=str(server.dtype),
        quantization=server.quantization,
        trust_remote_code=bool(server.trust_remote_code),
        mem_fraction_static=server.mem_fraction_static,
        max_running_requests=server.max_running_requests,
        host=server.host,
        port=int(server.port),
        **extra,
    )


def shutdown_backend(backend) -> None:
    if backend is None:
        return
    if hasattr(backend, "shutdown"):
        try:
            backend.shutdown()
        except Exception:
            pass


def backend_model_name(backend) -> str | None:
    if hasattr(backend, "model_path"):
        return str(getattr(backend, "model_path"))
    if hasattr(backend, "server_args") and hasattr(backend.server_args, "model_path"):
        return str(backend.server_args.model_path)
    return None


def run_batch_requests(
    *,
    backend,
    program,
    batch: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    json_schema: str | None,
) -> list[dict[str, str]]:
    if isinstance(backend, VLLMEndpointBackend):
        return backend.run_batch(
            batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            json_schema=json_schema,
        )
    if program is None:
        raise ValueError("program must be set for non-vLLM backends")
    outputs = program.run_batch(
        batch,
        backend=backend,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        progress_bar=False,
    )
    normalized: list[dict[str, str]] = []
    for item in outputs:
        if isinstance(item, dict):
            normalized.append({"raw": str(item.get("raw", ""))})
            continue

        raw_text = None
        if hasattr(item, "get_var"):
            try:
                raw_text = item.get_var("raw")
            except Exception:
                raw_text = None
        if raw_text is None and hasattr(item, "__getitem__"):
            try:
                raw_text = item["raw"]
            except Exception:
                raw_text = None
        if raw_text is None and hasattr(item, "text"):
            try:
                raw_text = item.text()
            except Exception:
                raw_text = None

        normalized.append({"raw": "" if raw_text is None else str(raw_text)})

    return normalized


def build_program(*, json_schema: str | None):
    import sglang as sgl

    @sgl.function
    def _edit_script(s, system_prompt: str, user_prompt: str):
        s += sgl.system(system_prompt)
        s += sgl.user(user_prompt)
        s += sgl.assistant(
            sgl.gen(
                "raw",
                temperature=0.0,
                top_p=1.0,
                json_schema=json_schema,
            )
        )

    return _edit_script
