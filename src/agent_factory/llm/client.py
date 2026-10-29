"""Client for LLMP's OpenAI-compatible /v1 API.

Every call asks for JSON matching a pydantic model, validates the reply (with one repair attempt),
passes a seed, and tags the call with the run and the role so LLMP's usage stats can be grouped by
them. The privacy guard runs before any request. Replies are cached on disk, keyed by the full
request, so a rerun with the same seed replays without calling a model.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar
from urllib.parse import urlsplit

import httpx
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, ValidationError

from agent_factory.llm.privacy import LLMP_LOG_MODE, PrivacyMode, check_base_url, check_model

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """A model call failed or its reply could not be used."""


class LLM(Protocol):
    def structured(
        self,
        schema: type[T],
        *,
        model: str,
        system: str,
        user: str,
        agent: str,
        temperature: float,
        seed: int,
    ) -> T: ...


@dataclass(frozen=True)
class LLMPSettings:
    base_url: str
    api_key: str

    @classmethod
    def from_env(cls) -> LLMPSettings:
        """`LLMP_BASE_URL`/`LLMP_API_KEY`, falling back to v1's `LLMP_URL`/`LLMP_PASSWORD`."""
        load_dotenv()
        base_url = os.getenv("LLMP_BASE_URL", "").strip()
        if not base_url and (legacy := os.getenv("LLMP_URL", "").strip()):
            parts = urlsplit(legacy)
            base_url = f"{parts.scheme}://{parts.netloc}/v1"
        if not base_url:
            raise LLMError("set LLMP_BASE_URL (for example http://localhost:8000/v1) in .env")
        api_key = (os.getenv("LLMP_API_KEY") or os.getenv("LLMP_PASSWORD") or "").strip()
        return cls(base_url.rstrip("/"), api_key)


class LLMPClient:
    def __init__(
        self,
        settings: LLMPSettings,
        privacy: PrivacyMode,
        run_id: str,
        cache_dir: Path | None = None,
        http_client: httpx.Client | None = None,
        src: str = "agent_factory",
    ):
        check_base_url(settings.base_url, privacy)
        self.settings = settings
        self.privacy = privacy
        self.run_id = run_id
        self.cache_dir = cache_dir
        self.src = src
        self.requests = 0
        self.cache_hits = 0
        self._client = openai.OpenAI(
            base_url=settings.base_url,
            api_key=settings.api_key or "none",
            timeout=600,
            max_retries=2,
            http_client=http_client,
        )

    def structured(
        self,
        schema: type[T],
        *,
        model: str,
        system: str,
        user: str,
        agent: str,
        temperature: float,
        seed: int,
    ) -> T:
        check_model(model, self.privacy)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        for attempt in range(2):
            content = self._complete(model, messages, schema, temperature, seed, agent)
            try:
                return schema.model_validate_json(_extract_json(content))
            except ValidationError as e:
                if attempt:
                    raise LLMError(f"{agent}: {model} did not return valid {schema.__name__}: {e}") from e
                messages = [
                    *messages,
                    {"role": "assistant", "content": content},
                    {"role": "user", "content": f"That reply does not match the required JSON schema:\n{e}\n"
                                                "Reply again with only the corrected JSON object."},
                ]
        raise AssertionError("unreachable")

    def _complete(self, model: str, messages: list[dict], schema: type[BaseModel],
                  temperature: float, seed: int, agent: str) -> str:
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "seed": seed,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema.__name__, "schema": inline_schema(schema)},
            },
        }
        key = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        llmp: dict[str, Any] = {"src": self.src, "run_id": self.run_id, "agent": agent,
                                "session_id": f"{self.run_id}/{agent}"}
        if (log := LLMP_LOG_MODE[self.privacy]) is not None:
            llmp["log"] = log
        self.requests += 1
        try:
            response = self._client.chat.completions.create(**request, extra_body={"llmp": llmp})
        except openai.APIConnectionError as e:
            raise LLMError(f"cannot reach LLMP at {self.settings.base_url}: is it running? ({e})") from e
        except openai.APIStatusError as e:
            raise LLMError(f"{agent}: LLMP returned {e.status_code} for {model}: {e.message}") from e
        content = response.choices[0].message.content or ""
        self._cache_put(key, content)
        return content

    def _cache_get(self, key: str) -> str | None:
        if self.cache_dir is None:
            return None
        path = self.cache_dir / f"{key}.json"
        return json.loads(path.read_text(encoding="utf-8"))["content"] if path.exists() else None

    def _cache_put(self, key: str, content: str) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.cache_dir / f"{key}.json").write_text(json.dumps({"content": content}), encoding="utf-8")


def inline_schema(schema: type[BaseModel]) -> dict:
    """The model's JSON schema with `$ref`s expanded, since not every backend resolves them."""
    raw = schema.model_json_schema()
    definitions = raw.pop("$defs", {})

    def expand(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return expand(definitions[node["$ref"].rsplit("/", 1)[-1]])
            return {k: expand(v) for k, v in node.items()}
        if isinstance(node, list):
            return [expand(item) for item in node]
        return node

    return expand(raw)


def _extract_json(content: str) -> str:
    """Tolerate code fences or text around the JSON object."""
    text = content.strip()
    start, end = text.find("{"), text.rfind("}")
    return text[start : end + 1] if start != -1 and end > start else text
