# Arquitectura del Proyecto

## Overview
Este documento describe la arquitectura del pipeline de embeddings, clustering y Retrieval-Augmented Generation (RAG) en este repositorio. **Embeddings y clustering están implementados**, íntegramente dentro de `notebooks/pipeline.ipynb`; **RAG está pendiente** (Milvus ya está listo pero el notebook no lo usa todavía). Ver `CLAUDE.md` para el detalle línea a línea; aquí solo el mapa general.

## Componentes del Sistema

1. **Ingesta y Data Wrangling** (secciones 1-2 del notebook):
   - Carga de `data/raw/stratified_grid_2019_2026.csv` (~28,777 artículos).
   - Normalización de texto, deduplicación, mapeo a `seccion_canonica` (17 macro-secciones) y filtro de artículos ≤40 palabras → corpus de 26,210 artículos.

2. **Generación de Embeddings** (sección 3, dentro del notebook, no en `src/`):
   - `EmbeddingSpec` (config inmutable por modelo, forma parte de la huella de caché) + `EmbeddingRunner` (carga, inferencia, checkpoints, liberación de GPU) — arquitectura común a los cinco modelos comparados: BGE-M3 (remoto, vía `src/llm_clients` contra vLLM), Jina-v3, E5-large, Qwen3-Embedding-0.6B y Jina-v2-ES (los cuatro últimos locales, vía `sentence-transformers`).
   - Chunking + agregación ponderada por cobertura nueva (`dividir_ids_con_cobertura`, `agregar_chunks`) para artículos que exceden la ventana del modelo.
   - Persistencia en `data/embeddings/` (`{spec.cache_name}_full.npy` + `_ids.npy` + `_meta.json`), validada contra `id_articulo` y una huella SHA-256 del corpus + parámetros.

3. **Clustering** (sección 4, también dentro del notebook, no en `src/clustering/`):
   - HDBSCAN (paquete standalone `hdbscan`) sobre cada embedding, con y sin reducción previa vía UMAP. Toda la lógica vive en `analizar_espacio(emb, nombre, data, ...)`, que corre clustering + barrido jerárquico + soft clustering + métricas + plots sobre cualquier matriz `[N, D]`; `comparar_espacios(resultados)` arma las tablas comparativas.
   - Métricas jerarquizadas en 4 niveles (AMI/DBCV/silhouette/coherencia deciden; ARI y otras se reportan sin decidir).

4. **AE / VAE** (sección 5, exploratorio):
   - Autoencoder/VAE genérico sobre cualquier matriz `[N, D]`, evaluado con el mismo `analizar_espacio` sobre los latentes de los cinco embeddings.

5. **Base Vectorial (Milvus):**
   - `docker-compose.yml` levanta Milvus + etcd + minio (+ Attu opcional). `scripts/check_milvus.py` es la única prueba de humo actual — el notebook todavía no escribe embeddings a Milvus.

6. **Retrieval-Augmented Generation (RAG):**
   - Pendiente. `src/rag/` está vacío. El entorno (Milvus + cliente vLLM para `deepseek-v4-flash`) ya está listo para cuando se implemente.
