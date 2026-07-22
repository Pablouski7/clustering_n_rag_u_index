"""Embeddings de artículos con llama-nemotron-embed-vl-1b-v2 (NVIDIA), modo solo texto.

`nvidia/llama-nemotron-embed-vl-1b-v2` es un modelo **multimodal** (Eagle VLM: Llama 3.2 1B
como modelo de lenguaje + SigLip2 400M como codificador de imagen, ~1.7 B parámetros en
total) entrenado con aprendizaje contrastivo para *retrieval* de páginas de documento.
Produce vectores de **2,048 dimensiones** por mean pooling sobre los tokens de salida.

Aquí se usa **solo la rama de texto** (`encode_document`): el corpus son artículos de
prensa en texto plano, sin imágenes de página. La rama de visión queda cargada en memoria
pero nunca se ejecuta. Es un uso legítimo del modelo — la tarjeta documenta el modo
texto-solo — pero conviene tenerlo presente al interpretar resultados: el modelo fue
optimizado para emparejar *consultas* con *páginas*, no para medir similitud entre pares de
documentos, que es lo que pide el clustering.

**Sin chunking, a diferencia de BETO.** La ventana evaluada del modelo es de 10,240 tokens y
el artículo más largo del corpus ronda los 6,900, así que todo entra en un solo forward pass.
Esto lo convierte en la única técnica de la comparación que ve el artículo completo sin
truncarlo (TF-IDF, MiniLM y BGE-M3 cortan a 2,000 caracteres) ni promediar chunks (BETO), y
por tanto sin el artefacto de longitud que introduce ese promediado.

**Costo en CPU.** Con ~1.7 B parámetros y textos de varios miles de tokens, este es de lejos
el método más caro de la comparación: en CPU tarda horas, frente a los ~10 minutos de BETO.
Se usa `bfloat16` para que los pesos ocupen ~3.4 GB en vez de ~6.8 GB. Los textos se ordenan
por longitud antes de embeder para que cada lote agrupe textos de tamaño similar y el padding
no desperdicie cómputo; el orden original se restaura al final.
"""

from __future__ import annotations

import numpy as np
import torch

MODELO_NEMOTRON = "nvidia/llama-nemotron-embed-vl-1b-v2"


def cargar_modelo(
    modelo_nombre: str = MODELO_NEMOTRON,
    device: str | None = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """Carga el modelo como `SentenceTransformer` (requiere `trust_remote_code`)."""
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # `sdpa` en vez de flash-attention: esta máquina no tiene GPU y flash-attn no
    # compila para CPU.
    return SentenceTransformer(
        modelo_nombre,
        trust_remote_code=True,
        device=device,
        model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
    )


@torch.inference_mode()
def embed_articulos(
    textos: list[str],
    modelo=None,
    batch_size: int = 2,
    normalizar: bool = True,
    **kwargs_modelo,
) -> np.ndarray:
    """Devuelve (len(textos), 2048): un embedding Nemotron por artículo.

    `modelo` permite reutilizar una instancia ya cargada; si es None se carga una nueva con
    `cargar_modelo(**kwargs_modelo)`. Con `normalizar=True` los vectores salen
    L2-normalizados, listos para que la distancia euclídea sea monótona respecto a la
    coseno (ver `hc_clustering`).

    `batch_size` es deliberadamente bajo: los artículos largos ocupan miles de tokens y la
    atención crece cuadráticamente, así que lotes grandes disparan la memoria sin ganar
    velocidad en CPU.
    """
    modelo = modelo if modelo is not None else cargar_modelo(**kwargs_modelo)

    # Ordenar por longitud agrupa textos similares en cada lote y reduce el padding.
    orden = np.argsort([len(t) for t in textos])
    ordenados = [textos[i] for i in orden]

    emb_ordenados = modelo.encode_document(
        ordenados,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    emb = np.empty_like(emb_ordenados)
    emb[orden] = emb_ordenados  # deshacer el orden por longitud
    emb = emb.astype(np.float32)

    if normalizar:
        emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    return emb
