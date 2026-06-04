import os
import json
import logging
from typing import Any

import ollama
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Thin wrapper around the Ollama Python SDK.
    Loads configuration from environment variables and exposes
    a single `chat` method used by the rest of the system.
    """

    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        timeout: int | None = None,
    ):
        self.host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
        self.timeout = timeout or int(os.getenv("OLLAMA_TIMEOUT", "120"))
        self._client = ollama.Client(host=self.host, timeout=self.timeout)

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        json_mode: bool = False,
        temperature: float = 0.2,
    ) -> str:
        """
        Sends a prompt to the local Ollama model and returns the raw text.
        Args:
            prompt (str): User prompt.
            system (str | None): Optional system message.
            json_mode (bool): If True, forces the model to reply with JSON.
            temperature (float): Sampling temperature.
        Returns:
            str: Model response text.
        Raises:
            RuntimeError: If the call to the model fails.
        """
        messages: list[dict[str, Any]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "options": {"temperature": temperature},
        }
        if json_mode:
            kwargs["format"] = "json"

        logger.debug("Calling Ollama model=%s json=%s", self.model, json_mode)

        try:
            response = self._client.chat(**kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"Ollama call failed (host={self.host}, model={self.model}): {exc}"
            ) from exc

        content = response.get("message", {}).get("content", "")
        return content.strip()

    def chat_json(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """
        Calls the model expecting a JSON response and returns it parsed.
        Retries up to 2 times if the model returns invalid JSON.
        Args:
            prompt (str): User prompt.
            system (str | None): Optional system message.
            temperature (float): Sampling temperature.
        Returns:
            dict[str, Any]: Parsed JSON.
        Raises:
            RuntimeError: If JSON parsing keeps failing.
        """
        last_error: Exception | None = None
        for attempt in range(3):
            raw = self.chat(
                prompt=prompt,
                system=system,
                json_mode=True,
                temperature=temperature,
            )
            try:
                return json.loads(raw)
            except json.JSONDecodeError as exc:
                last_error = exc
                logger.warning(
                    "Invalid JSON from Ollama (attempt %d/3): %s",
                    attempt + 1,
                    exc,
                )
        raise RuntimeError(
            f"Ollama returned invalid JSON after 3 attempts: {last_error}"
        )

    def is_available(self) -> bool:
        """
        Returns True if the configured model is reachable on the host.
        """
        try:
            models = self._client.list()
            names = [m.get("model") or m.get("name") for m in models.get("models", [])]
            return any(n and n.startswith(self.model.split(":")[0]) for n in names)
        except Exception as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False
