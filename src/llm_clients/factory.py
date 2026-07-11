"""Fábricas de clientes para los endpoints vLLM (API compatible con OpenAI).

Requiere el extra `embeddings-openai` instalado (paquete `openai`).
"""

from __future__ import annotations

from openai import OpenAI

from .config import VLLMConfig, load_config


def get_chat_client(config: VLLMConfig | None = None) -> OpenAI:
    config = config or load_config()
    return OpenAI(
        base_url=config.chat_base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )


def get_embedding_client(config: VLLMConfig | None = None) -> OpenAI:
    config = config or load_config()
    return OpenAI(
        base_url=config.embedding_base_url,
        api_key=config.api_key,
        timeout=config.timeout,
    )
