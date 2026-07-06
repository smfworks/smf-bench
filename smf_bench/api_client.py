"""
API Client — OpenAI-compatible client with text and binary output handling.

Supports:
- /v1/chat/completions (text, vision, video, audio, tool calling)
- /v1/completions (raw text completion for throughput tests)
- /v1/embeddings (embedding models)
- Streaming (for TTFT measurement)
- Binary output handling (for media generation models — image/audio/video)

The client is endpoint-agnostic: point it at any OpenAI-compatible server
(vLLM, Ollama, TGI, etc.) and it just works.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


@dataclass
class APIResponse:
    """Normalized response from an API call."""
    text: str = ""
    reasoning: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    elapsed: float = 0.0
    ttft: float | None = None      # time to first token (seconds), if streaming
    finish_reason: str = ""
    binary_path: str | None = None  # path to saved binary output (images, audio)
    raw: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


class APIClient:
    """Async OpenAI-compatible API client."""

    def __init__(
        self,
        base_url: str = "http://localhost:8888/v1",
        model: str = "",
        api_key: str = "dummy",
        timeout: float = 300.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = httpx.Timeout(timeout, connect=10.0)
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "APIClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if not self._client:
            raise RuntimeError("APIClient must be used as async context manager")
        return self._client

    def _payload(self, messages: list[dict], **kwargs: Any) -> dict:
        payload = {
            "model": kwargs.get("model", self.model),
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.6),
        }
        if "tools" in kwargs:
            payload["tools"] = kwargs["tools"]
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]
        # Pass through extra kwargs (e.g. chat_template_kwargs)
        for k, v in kwargs.items():
            if k not in ("model", "max_tokens", "temperature", "tools", "stop"):
                payload[k] = v
        return payload

    async def chat(
        self,
        messages: list[dict],
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> APIResponse:
        """Send a chat completion request. Returns normalized APIResponse."""
        client = self._ensure_client()
        url = f"{self.base_url}/chat/completions"
        payload = self._payload(messages, **kwargs)
        start = time.perf_counter()

        try:
            if stream:
                return await self._stream_chat(client, url, payload, start)
            else:
                resp = await client.post(url, json=payload)
                elapsed = time.perf_counter() - start
                if resp.status_code != 200:
                    return APIResponse(
                        error=f"HTTP {resp.status_code}: {resp.text[:500]}",
                        elapsed=elapsed,
                        raw={"status_code": resp.status_code},
                    )
                data = resp.json()
                return self._parse_response(data, elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            return APIResponse(error=str(e)[:500], elapsed=elapsed)

    async def _stream_chat(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict,
        start: float,
    ) -> APIResponse:
        """Handle streaming response for TTFT measurement."""
        payload = {**payload, "stream": True}
        first_token_time: float | None = None
        content = ""
        reasoning = ""
        tool_calls: list[dict] = []
        finish_reason = ""

        async with client.stream("POST", url, json=payload) as resp:
            if resp.status_code != 200:
                body = await resp.aread()
                return APIResponse(
                    error=f"HTTP {resp.status_code}: {body.decode()[:500]}",
                    elapsed=time.perf_counter() - start,
                )
            async for line in resp.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                choices = chunk.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                if "content" in delta and delta["content"]:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    content += delta["content"]
                if "reasoning" in delta and delta["reasoning"]:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    reasoning += delta["reasoning"]
                if "tool_calls" in delta and delta["tool_calls"]:
                    tool_calls.extend(delta["tool_calls"])
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]

        elapsed = time.perf_counter() - start
        ttft = (first_token_time - start) if first_token_time else None
        return APIResponse(
            text=content,
            reasoning=reasoning,
            tool_calls=tool_calls,
            elapsed=elapsed,
            ttft=ttft,
            finish_reason=finish_reason,
        )

    def _parse_response(self, data: dict, elapsed: float) -> APIResponse:
        """Parse a non-streaming chat completion response."""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return APIResponse(
            text=message.get("content", "") or "",
            reasoning=message.get("reasoning", "") or "",
            tool_calls=message.get("tool_calls", []),
            usage=data.get("usage", {}),
            elapsed=elapsed,
            finish_reason=choice.get("finish_reason", ""),
            raw=data,
        )

    async def completion(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.6,
    ) -> APIResponse:
        """Send a raw /v1/completions request for throughput testing."""
        client = self._ensure_client()
        url = f"{self.base_url}/completions"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        start = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            elapsed = time.perf_counter() - start
            if resp.status_code != 200:
                return APIResponse(
                    error=f"HTTP {resp.status_code}: {resp.text[:500]}",
                    elapsed=elapsed,
                )
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            return APIResponse(
                text=choice.get("text", ""),
                usage=data.get("usage", {}),
                elapsed=elapsed,
                finish_reason=choice.get("finish_reason", ""),
                raw=data,
            )
        except Exception as e:
            return APIResponse(error=str(e)[:500], elapsed=time.perf_counter() - start)

    async def embed(self, input_text: str) -> APIResponse:
        """Send a /v1/embeddings request."""
        client = self._ensure_client()
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": input_text}
        start = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            elapsed = time.perf_counter() - start
            if resp.status_code != 200:
                return APIResponse(error=f"HTTP {resp.status_code}", elapsed=elapsed)
            data = resp.json()
            return APIResponse(
                raw=data,
                elapsed=elapsed,
            )
        except Exception as e:
            return APIResponse(error=str(e)[:500], elapsed=time.perf_counter() - start)

    async def generate_image(
        self,
        prompt: str,
        save_path: str | Path,
        **kwargs: Any,
    ) -> APIResponse:
        """Call an image generation endpoint and save the result.

        Works with OpenAI DALL-E style /v1/images/generations or
        vLLM's image generation endpoints.
        """
        client = self._ensure_client()
        url = f"{self.base_url}/images/generations"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "response_format": "b64_json",
            **kwargs,
        }
        start = time.perf_counter()
        try:
            resp = await client.post(url, json=payload)
            elapsed = time.perf_counter() - start
            if resp.status_code != 200:
                return APIResponse(error=f"HTTP {resp.status_code}: {resp.text[:500]}", elapsed=elapsed)
            data = resp.json()
            images = data.get("data", [])
            if not images:
                return APIResponse(error="No images in response", elapsed=elapsed)
            b64 = images[0].get("b64_json", "")
            if b64:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(base64.b64decode(b64))
                return APIResponse(
                    binary_path=str(save_path),
                    elapsed=elapsed,
                    raw=data,
                )
            return APIResponse(error="No b64_json in response", elapsed=elapsed, raw=data)
        except Exception as e:
            return APIResponse(error=str(e)[:500], elapsed=time.perf_counter() - start)

    async def health_check(self) -> bool:
        """Check if the endpoint is reachable."""
        client = self._ensure_client()
        try:
            resp = await client.get(f"{self.base_url}/models", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def get_model_info(self) -> dict | None:
        """Get model info from the /v1/models endpoint."""
        client = self._ensure_client()
        try:
            resp = await client.get(f"{self.base_url}/models")
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        return None


# ─── Helpers for building multimodal messages ─────────────────────────────────

def make_image_message(prompt: str, image_b64: str, mime_type: str = "image/png") -> dict:
    """Build a chat message with an inline base64 image."""
    data_url = f"data:{mime_type};base64,{image_b64}"
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ],
    }


def make_video_message(prompt: str, video_b64: str, mime_type: str = "video/mp4") -> dict:
    """Build a chat message with an inline base64 video."""
    data_url = f"data:{mime_type};base64,{video_b64}"
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "video_url", "video_url": {"url": data_url}},
        ],
    }


def make_audio_message(prompt: str, audio_b64: str, mime_type: str = "audio/wav") -> dict:
    """Build a chat message with inline base64 audio."""
    data_url = f"data:{mime_type};base64,{audio_b64}"
    return {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {"data": audio_b64, "format": mime_type.split("/")[-1]}},
        ],
    }


def encode_file_b64(path: str | Path) -> str:
    """Read a file and return its base64-encoded content."""
    return base64.b64encode(Path(path).read_bytes()).decode()