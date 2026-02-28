from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal, Protocol

from app.agent.prompt_loader import load_prompt
from app.agent.result_parser import extract_response_text

logger = logging.getLogger(__name__)

SubagentName = Literal["greeting", "core_capture", "knowledge", "inventory"]


class SubagentRunner(Protocol):
    async def run(
        self,
        *,
        agent_name: SubagentName,
        user_message: str,
        context: dict[str, Any],
    ) -> str | None: ...


class LangChainSubagentRunner:
    def __init__(
        self,
        *,
        model_name: str,
        aws_region: str,
        temperature: float,
    ) -> None:
        self._model_name = model_name
        self._aws_region = aws_region
        self._temperature = temperature

        self._lock = asyncio.Lock()
        self._agents: dict[SubagentName, Any] = {}
        self._bootstrap_error: str | None = None

    async def run(
        self,
        *,
        agent_name: SubagentName,
        user_message: str,
        context: dict[str, Any],
    ) -> str | None:
        agent = await self._get_or_build_agent(agent_name)
        if agent is None:
            return None

        payload = self._build_user_payload(
            user_message=user_message,
            context=context,
        )

        try:
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": payload}]}
            )
            text = extract_response_text(result).strip()
            return text or None
        except Exception as exc:  # pragma: no cover - provider/network dependent
            logger.warning("subagent_invoke_failed agent=%s error=%s", agent_name, exc)
            return None

    async def _get_or_build_agent(self, agent_name: SubagentName) -> Any | None:
        existing = self._agents.get(agent_name)
        if existing is not None:
            return existing
        if self._bootstrap_error is not None:
            return None

        async with self._lock:
            existing = self._agents.get(agent_name)
            if existing is not None:
                return existing
            if self._bootstrap_error is not None:
                return None

            try:
                from langchain.agents import create_agent
                from langchain_aws import ChatBedrockConverse
            except Exception as exc:  # pragma: no cover - optional dependency
                self._bootstrap_error = f"{type(exc).__name__}: {exc}"
                logger.info("subagent_bootstrap_unavailable error=%s", self._bootstrap_error)
                return None

            prompt_name = self._prompt_name_for(agent_name)
            system_prompt = load_prompt(prompt_name)
            model = ChatBedrockConverse(
                model_id=self._model_name,
                region_name=self._aws_region,
                temperature=self._temperature,
                disable_streaming=True,
            )
            try:
                agent = create_agent(
                    model=model,
                    tools=[],
                    system_prompt=system_prompt,
                )
            except Exception as exc:  # pragma: no cover - provider/network dependent
                self._bootstrap_error = f"{type(exc).__name__}: {exc}"
                logger.warning("subagent_create_failed agent=%s error=%s", agent_name, exc)
                return None

            self._agents[agent_name] = agent
            return agent

    @staticmethod
    def _prompt_name_for(agent_name: SubagentName) -> str:
        if agent_name == "greeting":
            return "agent_greeting"
        if agent_name == "core_capture":
            return "agent_core_capture"
        if agent_name == "inventory":
            return "agent_inventory"
        return "agent_knowledge"

    @staticmethod
    def _build_user_payload(*, user_message: str, context: dict[str, Any]) -> str:
        serialized = json.dumps(context, ensure_ascii=True)
        return (
            "Contexto estructurado de la sesion (JSON):\n"
            f"{serialized}\n\n"
            "Mensaje del usuario:\n"
            f"{user_message}\n\n"
            "Responde al usuario siguiendo el system prompt."
        )


async def invoke_core_capture(
    *,
    runner: SubagentRunner | None,
    user_message: str,
    context: dict[str, Any],
) -> str | None:
    if runner is None:
        return None
    return await runner.run(agent_name="core_capture", user_message=user_message, context=context)


async def invoke_greeting(
    *,
    runner: SubagentRunner | None,
    user_message: str,
    context: dict[str, Any],
) -> str | None:
    if runner is None:
        return None
    return await runner.run(agent_name="greeting", user_message=user_message, context=context)


async def invoke_knowledge(
    *,
    runner: SubagentRunner | None,
    user_message: str,
    context: dict[str, Any],
) -> str | None:
    if runner is None:
        return None
    return await runner.run(agent_name="knowledge", user_message=user_message, context=context)


async def invoke_inventory(
    *,
    runner: SubagentRunner | None,
    user_message: str,
    context: dict[str, Any],
) -> str | None:
    if runner is None:
        return None
    return await runner.run(agent_name="inventory", user_message=user_message, context=context)
