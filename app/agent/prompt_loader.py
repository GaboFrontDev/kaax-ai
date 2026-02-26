from __future__ import annotations

import logging
import re
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class PromptSchema(BaseModel):
    version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    prompt: str = Field(min_length=1)


SYSTEM_PROMPTS: dict[str, str] = {
    "default": (
        "Rol: asistente técnico de operación comercial. "
        "Reglas: no inventar resultados de tools; pedir aclaración cuando falten datos críticos. "
        "Idioma: responde siempre en español neutro, salvo que el usuario pida explícitamente otro idioma."
    ),
    "intent_router": "Clasifica intención en JSON estricto y no ejecutes tools.",
}

_PROMPT_CACHE: dict[str, str] = {}


def _get_prompts_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "prompts"


def _parse_prompt_yaml(raw: str) -> PromptSchema:
    version: str | None = None
    prompt_block_start: int | None = None
    lines = raw.splitlines()

    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if version is None and stripped.startswith("version:"):
            value = stripped.split(":", 1)[1].strip().strip('"').strip("'")
            if not re.fullmatch(r"\d+\.\d+\.\d+", value):
                raise ValueError("prompt version must follow semver x.y.z")
            version = value
            continue

        if stripped == "prompt: |":
            prompt_block_start = index + 1
            break

    if version is None:
        raise ValueError("prompt file missing version")
    if prompt_block_start is None:
        raise ValueError("prompt file missing 'prompt: |' block")

    prompt_lines = lines[prompt_block_start:]
    non_empty = [line for line in prompt_lines if line.strip()]
    common_indent = (
        min(len(line) - len(line.lstrip(" ")) for line in non_empty)
        if non_empty
        else 0
    )
    normalized_lines = [
        line[common_indent:] if len(line) >= common_indent else ""
        for line in prompt_lines
    ]
    prompt = "\n".join(normalized_lines).rstrip()
    return PromptSchema(version=version, prompt=prompt)


def _load_file_prompt(name: str) -> str | None:
    safe_name = Path(name).name
    if safe_name != name:
        raise ValueError(f"invalid prompt name '{name}'")

    prompt_path = _get_prompts_dir() / f"{safe_name}.yaml"
    if not prompt_path.exists():
        return None

    raw = prompt_path.read_text(encoding="utf-8")
    model = _parse_prompt_yaml(raw)
    return model.prompt


def load_prompt(name: str | None) -> str:
    prompt_name = name or "default"

    cached = _PROMPT_CACHE.get(prompt_name)
    if cached is not None:
        return cached

    try:
        file_prompt = _load_file_prompt(prompt_name)
        if file_prompt is not None:
            _PROMPT_CACHE[prompt_name] = file_prompt
            return file_prompt
    except Exception:
        logger.exception("prompt_load_failed_name_%s", prompt_name)

    fallback = SYSTEM_PROMPTS.get(prompt_name, SYSTEM_PROMPTS["default"])
    _PROMPT_CACHE[prompt_name] = fallback
    return fallback
