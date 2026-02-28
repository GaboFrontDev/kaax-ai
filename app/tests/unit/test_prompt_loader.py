from pathlib import Path

import app.agent.prompt_loader as prompt_loader


def test_load_prompt_reads_yaml_file(monkeypatch) -> None:
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    monkeypatch.setattr(prompt_loader, "_get_prompts_dir", lambda: prompts_dir)
    prompt_loader._PROMPT_CACHE.clear()

    prompt = prompt_loader.load_prompt("agent")

    assert "customer support and lead-contact assistant" in prompt
    assert "Always respond in neutral Spanish" in prompt


def test_load_prompt_falls_back_to_default_when_file_missing(monkeypatch) -> None:
    prompts_dir = Path(__file__).resolve().parents[2] / "prompts"
    monkeypatch.setattr(prompt_loader, "_get_prompts_dir", lambda: prompts_dir)
    prompt_loader._PROMPT_CACHE.clear()

    prompt = prompt_loader.load_prompt("does-not-exist")

    assert prompt == prompt_loader.SYSTEM_PROMPTS["default"]
