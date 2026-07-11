"""Configuración de los endpoints vLLM (compatibles con OpenAI) del servidor H200
de la universidad. Requiere VPN GlobalProtect activa hacia 172.28.230.10.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class VLLMConfig:
    chat_base_url: str
    chat_model: str
    embedding_base_url: str
    embedding_model: str
    api_key: str
    timeout: float


def load_config() -> VLLMConfig:
    return VLLMConfig(
        chat_base_url=os.getenv("VLLM_CHAT_BASE_URL", "http://172.28.230.10:12555/v1"),
        chat_model=os.getenv("VLLM_CHAT_MODEL", "deepseek-ai/DeepSeek-V4-Flash"),
        embedding_base_url=os.getenv(
            "VLLM_EMBEDDING_BASE_URL", "http://172.28.230.10:12556/v1"
        ),
        embedding_model=os.getenv("VLLM_EMBEDDING_MODEL", "BAAI/bge-m3"),
        api_key=os.getenv("VLLM_API_KEY", "local"),
        timeout=float(os.getenv("VLLM_TIMEOUT", "60")),
    )
