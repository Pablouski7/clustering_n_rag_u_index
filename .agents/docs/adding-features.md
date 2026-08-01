# Guía para Agregar Nuevas Funcionalidades

Este documento es una guía sobre cómo agregar nuevas características.

## Pipeline Steps

### 1. Embeddings
- Los modelos de embedding viven **dentro de `notebooks/pipeline.ipynb`** (sección 3), no en módulos: cada uno es una `EmbeddingSpec` (config inmutable) agregada a `MODELOS_PRINCIPALES`, consumida por `EmbeddingRunner` con el mismo contrato de caché (`cargar_cache_full`/`guardar_cache_full`). Para agregar un modelo, seguí ese patrón.
- Si consume un servicio externo (como BGE-M3 vía vLLM), usá las fábricas de `src/llm_clients/` en vez de instanciar clientes a mano.

### 2. Clustering
- Igual que embeddings: la lógica vive **dentro del notebook** (sección 4, función `analizar_espacio`), no en `src/clustering/`. Ese paquete está vacío a propósito — seguí el patrón del notebook en vez de reintroducirlo, salvo que el usuario decida extraer el pipeline de módulos.

### 3. RAG (Generación Aumentada por Recuperación)
- Pendiente de implementar. `src/rag/` está vacío; es el único de los tres pasos sin patrón establecido todavía en el notebook — confirmá con el usuario si el RAG también debe vivir ahí o merece módulos propios en `src/rag/`.
