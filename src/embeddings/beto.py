"""Embeddings de artículos con BETO (BERT español) mediante chunking + attention pooling.

BETO (`dccuchile/bert-base-spanish-wwm-cased`) tiene una ventana de contexto de 512
tokens, insuficiente para un artículo de prensa completo. La estrategia implementada:

1. **Chunking**: el texto se tokeniza completo y se parte en ventanas solapadas de
   `max_tokens` (incluyendo [CLS]/[SEP]), con solape `stride_ratio` para no cortar ideas
   en la frontera.
2. **Pooling intra-chunk**: media de los estados ocultos de la última capa ponderada por
   la máscara de atención (mean pooling enmascarado), que empíricamente supera al vector
   [CLS] en modelos BERT sin fine-tuning para similitud semántica.
3. **Pooling inter-chunk**: media de los vectores de chunk.

Por qué media y no attention pooling: se probó una atención sin parámetros (query = centroide
de los chunks, scores = producto punto escalado) y resultó indistinguible de la media sobre
este corpus — coseno de 0.9999 entre ambos embeddings y r = 0.9988 entre las matrices de
distancia. La razón es que los artículos de prensa apenas exceden la ventana de BETO: la
mediana es de **1 chunk** y el 65% de los artículos cabe entero en 512 tokens, así que el
pooling inter-chunk casi nunca combina más de dos vectores. Además, con dos chunks los pesos
de esa atención dependen solo de la norma de cada chunk (correlacionada con la longitud, no
con la relevancia temática), no de su contenido.

Una atención *aprendible* sí podría aportar, pero requiere señal supervisada: sin entrenar,
sus pesos serían aleatorios y peores que la media. Si este módulo se aplica a documentos
largos (informes, transcripciones), donde la mediana de chunks sea alta, vale la pena
reconsiderar el pooling.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

MODELO_BETO = "dccuchile/bert-base-spanish-wwm-cased"


def partir_en_chunks(
    tokenizador, texto: str, max_tokens: int = 512, stride_ratio: float = 0.2
) -> list[list[int]]:
    """Parte `texto` en ventanas solapadas de `max_tokens` ids (con [CLS] y [SEP])."""
    ids = tokenizador(texto, add_special_tokens=False, truncation=False)["input_ids"]
    utiles = max_tokens - 2  # espacio para [CLS] y [SEP]
    paso = max(1, int(utiles * (1 - stride_ratio)))
    cls_id, sep_id = tokenizador.cls_token_id, tokenizador.sep_token_id

    chunks = [
        [cls_id, *ids[i : i + utiles], sep_id] for i in range(0, max(len(ids), 1), paso)
    ]
    # La última ventana puede quedar casi vacía por el solape: se descarta si es residual.
    if len(chunks) > 1 and len(chunks[-1]) < utiles * stride_ratio:
        chunks.pop()
    return chunks


def _codificar_chunks(
    modelo, chunks: list[list[int]], pad_id: int, device: torch.device
) -> torch.Tensor:
    """Pasa un lote de chunks por BETO y devuelve (n_chunks, dim) con mean pooling."""
    largo = max(len(c) for c in chunks)
    input_ids = torch.full((len(chunks), largo), pad_id, dtype=torch.long)
    mascara = torch.zeros((len(chunks), largo), dtype=torch.long)
    for i, c in enumerate(chunks):
        input_ids[i, : len(c)] = torch.tensor(c)
        mascara[i, : len(c)] = 1

    input_ids, mascara = input_ids.to(device), mascara.to(device)
    salida = modelo(input_ids=input_ids, attention_mask=mascara).last_hidden_state
    m = mascara.unsqueeze(-1).to(salida.dtype)
    return (salida * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)


@torch.inference_mode()
def embed_articulos(
    textos: list[str],
    modelo_nombre: str = MODELO_BETO,
    max_tokens: int = 512,
    stride_ratio: float = 0.2,
    batch_chunks: int = 16,
    device: str | None = None,
    normalizar: bool = True,
) -> np.ndarray:
    """Devuelve (len(textos), 768): un embedding BETO por artículo.

    `batch_chunks` es el número de chunks procesados por forward pass; bajarlo si la
    memoria es limitada. Con `normalizar=True` los vectores salen L2-normalizados, listos
    para que la distancia euclídea sea monótona respecto a la coseno (ver `hc_clustering`).
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizador = AutoTokenizer.from_pretrained(modelo_nombre)
    modelo = AutoModel.from_pretrained(modelo_nombre).to(device).eval()

    # Los chunks de todos los artículos se aplanan en una sola lista para que cada
    # forward pass vaya lleno: la mayoría de artículos aporta 1-3 chunks, así que
    # batchear por artículo desaprovecha el paralelismo.
    chunks_por_texto = [partir_en_chunks(tokenizador, t, max_tokens, stride_ratio) for t in textos]
    plano = [c for chunks in chunks_por_texto for c in chunks]

    vecs = torch.cat([
        _codificar_chunks(modelo, plano[i : i + batch_chunks], tokenizador.pad_token_id, device).cpu()
        for i in tqdm(range(0, len(plano), batch_chunks), desc="BETO (chunks)")
    ])

    vectores, inicio = [], 0
    for chunks in chunks_por_texto:
        vectores.append(vecs[inicio : inicio + len(chunks)].mean(dim=0))
        inicio += len(chunks)

    emb = torch.stack(vectores)
    if normalizar:
        emb = F.normalize(emb, p=2, dim=1)
    return emb.numpy()
