"""Thin wrapper around the Gemini API (google-genai SDK) for grounding a
human command against a scene graph's room JSON.

Falls back to a deterministic stub response if no API key is configured or
the client fails to initialize, so the rest of task_planning stays testable
without a real key -- same convention as NOVICDecoder.is_stub
(semantic_decoding/novic/novic_decoder.py) and VLLMTagger.is_stub
(semantic_decoding/vllm_tagger.py).
"""
from __future__ import annotations

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)


def resolve_api_key() -> str:
    """The key is never stored in configs (they are committed): $GEMINI_API_KEY, else GEMINI_API_KEY=... in
    the repo's git-ignored .env file."""
    import os
    from pathlib import Path

    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    env = Path(__file__).resolve().parents[1] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("GEMINI_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.model = model
        self._client = None
        api_key = api_key or resolve_api_key()

        if api_key:
            try:
                from google import genai

                self._client = genai.Client(api_key=api_key)
                logger.info(f"Gemini client ready (model={model}).")
            except Exception as e:
                logger.error(f"Failed to init Gemini client: {e}. Using stub mode.")
        else:
            logger.warning("No Gemini API key configured. Using stub mode.")

    @property
    def is_stub(self) -> bool:
        return self._client is None

    def generate(self, system_prompt: str, user_prompt: str, tools: list[Callable] | None = None) -> str:
        """Returns the raw THOUGHT:/CALL: response text. `tools` are plain
        Python callables (e.g. task_planning/clip_tool.py's
        get_clip_similarity) -- google-genai introspects their signature and
        docstring to build the tool schema, and automatic function calling
        (on by default when a callable, not a Tool object, is passed) handles
        the call/result round-trip internally, so this stays a single
        generate_content() call either way. Ignored in stub mode."""
        if self._client is None:
            return self._stub_response(user_prompt)

        config = {"system_instruction": system_prompt}
        if tools:
            config["tools"] = tools

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return response.text

    @staticmethod
    def _stub_response(user_prompt: str) -> str:
        """Deterministic placeholder: picks the first receptacle_id it finds
        in the room JSON embedded in the prompt, so downstream parsing stays
        exercisable without a real API call."""
        rec_id_match = re.search(r'"([^"]+)":\s*\{\s*"class"', user_prompt)
        rec_id = rec_id_match.group(1) if rec_id_match else "unknown_0"
        return (
            "THOUGHT: stub mode, no Gemini API key configured -- "
            f"picking the first receptacle seen ({rec_id!r}) as a placeholder.\n"
            f'CALL: tool_navigate_to_receptacle("{rec_id}")'
        )
