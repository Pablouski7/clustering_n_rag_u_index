# Guía para Agregar Nuevas Funcionalidades

Este documento es una guía sobre cómo agregar nuevas características.

## Pipeline Steps

### 1. Embeddings
- Los métodos de embeddings viven **dentro de `notebooks/pipeline.ipynb`** (sección 3), no en módulos: cada uno es una función `embed_*` con el mismo contrato de caché (`cargar_cache_full`/`guardar_cache_full`). Para agregar uno, seguí ese patrón.
- Si consume un servicio externo, usá las fábricas de `src/llm_clients/` en vez de instanciar clientes a mano.

### 2. Clustering
- Implementa nuevos algoritmos de clustering en `src/clustering/`.

### 3. RAG (Generación Aumentada por Recuperación)
- Implementa utilidades de consulta y recuperación semántica en `src/rag/`.
