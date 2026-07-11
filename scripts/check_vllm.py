#!/usr/bin/env python3
"""Prueba de humo de los endpoints vLLM (chat y embeddings) del servidor H200.

Requiere VPN GlobalProtect activa hacia 172.28.230.10; sin ella las
peticiones fallan por timeout, no por error de autenticación.

Uso:
    micromamba activate base
    python scripts/check_vllm.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import APIConnectionError, APITimeoutError

from src.llm_clients import get_chat_client, get_embedding_client, load_config

MENSAJE_VPN = (
    "   -> Verifica que la VPN GlobalProtect esté activa: sin ella, el tráfico "
    "hacia 172.28.230.10 se descarta silenciosamente."
)


def check_chat(config) -> bool:
    print(f"🔌 Probando chat en {config.chat_base_url} (modelo: {config.chat_model}) ...")
    client = get_chat_client(config)
    try:
        response = client.chat.completions.create(
            model=config.chat_model,
            messages=[{"role": "user", "content": "hola"}],
            temperature=0.1,
            max_tokens=200,  # deepseek-v4-flash razona antes de responder; 16 no alcanza
        )
        print(f"✅ Chat OK: {response.choices[0].message.content!r}")
        return True
    except (APIConnectionError, APITimeoutError) as e:
        print(f"❌ Chat falló: {e}")
        print(MENSAJE_VPN)
        return False


def check_embeddings(config) -> bool:
    print(
        f"🔌 Probando embeddings en {config.embedding_base_url} "
        f"(modelo: {config.embedding_model}) ..."
    )
    client = get_embedding_client(config)
    try:
        response = client.embeddings.create(
            model=config.embedding_model,
            input=["texto de prueba"],
        )
        dim = len(response.data[0].embedding)
        print(f"✅ Embeddings OK: vector de dimensión {dim}")
        return True
    except (APIConnectionError, APITimeoutError) as e:
        print(f"❌ Embeddings falló: {e}")
        print(MENSAJE_VPN)
        return False


def main() -> None:
    config = load_config()
    chat_ok = check_chat(config)
    embeddings_ok = check_embeddings(config)

    if chat_ok and embeddings_ok:
        print("\n🎉 Ambos endpoints vLLM responden correctamente.")
    else:
        print("\n⚠️  Al menos un endpoint vLLM no respondió.")
        sys.exit(1)


if __name__ == "__main__":
    main()
