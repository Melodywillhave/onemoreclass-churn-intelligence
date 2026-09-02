from __future__ import annotations

import json
from typing import Any

import requests
from pydantic import ValidationError

from src.rag.schemas import RetentionRecommendation

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"


def generate_recommendation(
    prompt: str,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    """Generate and validate a structured retention recommendation."""

    if not prompt.strip():
        raise ValueError("prompt must not be empty.")

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
        },
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120,
    )
    response.raise_for_status()

    result = response.json()
    generated_text = result["response"]

    try:
        generated_data = json.loads(generated_text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Ollama returned invalid JSON."
        ) from exc

    try:
        validated = RetentionRecommendation.model_validate(
            generated_data
        )
    except ValidationError as exc:
        raise ValueError(
            "Ollama returned JSON that does not match "
            "the retention recommendation schema."
        ) from exc

    return validated.model_dump()