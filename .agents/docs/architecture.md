# Arquitectura del Proyecto

## Overview
Este documento describe la arquitectura detallada del pipeline de procesamiento de texto, embeddings, clustering y Retrieval-Augmented Generation (RAG) en este repositorio.

## Componentes del Sistema
1. **Ingesta y Procesamiento de Datos:**
   - Carga de artículos desde archivos CSV (`data/raw/`).
   - Limpieza y normalización de textos.

2. **Generación de Embeddings:**
   - Integración con modelos de embeddings locales y remotos (vLLM, OpenAI, Voyage, etc.).
   - Persistencia de embeddings en archivos o bases vectoriales.

3. **Clustering:**
   - Algoritmos para agrupar artículos similares basados en sus embeddings (K-means, HDBSCAN, etc.).

4. **Base Vectorial (Milvus):**
   - Almacenamiento indexado de embeddings para búsquedas semánticas eficientes.

5. **Retrieval-Augmented Generation (RAG):**
   - Recuperación de fragmentos de prensa relevantes para responder consultas con LLMs.
