# Guía para Agregar Nuevas Funcionalidades

Este documento es una guía sobre cómo agregar nuevas características.

## Pipeline Steps

### 1. Embeddings
- Para agregar un nuevo proveedor o modelo de embeddings, crea o modifica un módulo dentro de `src/embeddings/`.
- Sigue el patrón establecido en `src/llm_clients/` si consume servicios externos.

### 2. Clustering
- Implementa nuevos algoritmos de clustering en `src/clustering/`.

### 3. RAG (Generación Aumentada por Recuperación)
- Implementa utilidades de consulta y recuperación semántica en `src/rag/`.
