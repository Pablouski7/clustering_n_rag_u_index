# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Estado del repositorio

Repositorio de tesis de maestría USFQ. Objetivo del proyecto: **embeddings + clustering + RAG** sobre la muestra de artículos de prensa, usando **Milvus** como base vectorial. Está en fase de **setup**: el entorno y la base vectorial quedan listos, pero el pipeline de embeddings/clustering/RAG **aún no está implementado** (el enfoque de embeddings está por decidir).

Contenido relevante:

- `data/raw/stratified_sample_2019_2026.csv` — muestra estratificada de ~9,551 artículos periodísticos (2019–2026), de tres periódicos ecuatorianos: **Diario Expreso**, **El Universo**, **Primicias**. Se versiona en git (es el dataset fuente).
- `data/processed/`, `data/embeddings/` — datos derivados, ignorados por git.
- `scripts/sample_stratified_articles.py` — script de referencia que generó el CSV (ver sección abajo; no ejecutable tal cual aquí).
- `src/embeddings/`, `src/clustering/`, `src/rag/` — módulos del pipeline, por implementar.
- `src/llm_clients/` — config (`config.py`) y fábricas de clientes (`factory.py`) para los endpoints vLLM (chat/embeddings) del servidor H200 de la universidad, vía SDK `openai` (API compatible con OpenAI). Ver sección "LLMs vía vLLM" abajo.
- `scripts/check_milvus.py` — prueba de humo de la base vectorial.
- `scripts/check_vllm.py` — prueba de humo de los endpoints vLLM (chat y embeddings).

## Comandos

```bash
# Entorno (esta máquina usa micromamba base; ver README para uv/conda)
micromamba activate base && pip install -r requirements.txt

# Base vectorial
docker compose up -d              # levanta etcd + minio + milvus
docker compose --profile ui up -d # además Attu (UI en http://localhost:8000)
docker compose ps                 # verificar que 'milvus' esté (healthy)
python scripts/check_milvus.py    # prueba de conectividad
docker compose down               # parar (agregar -v para borrar datos)

# LLMs vía vLLM (requiere VPN GlobalProtect activa, ver README)
pip install "openai>=1.40"        # extra embeddings-openai en pyproject.toml
python scripts/check_vllm.py      # prueba de conectividad (chat + embeddings)
```

## LLMs vía vLLM (servidor H200)

La universidad expone `deepseek-v4-flash` (chat/razonamiento), `BGE-M3` (embeddings), `gemma-4-31B` y `glm-ocr` vía vLLM con API compatible con OpenAI (HTTP, sin TLS) en `172.28.230.10`. Requiere VPN GlobalProtect activa; sin ella las peticiones fallan por timeout/conexión, no por autenticación (no hay auth real todavía).

Los IDs de modelo reales reportados por `GET {base_url}/models` difieren de los nombres cortos de la documentación: `deepseek-ai/DeepSeek-V4-Flash` (no `deepseek-v4-flash`) y `BAAI/bge-m3` (no `BGE-M3`) — estos son los valores por defecto en `src/llm_clients/config.py`. Si vLLM cambia de versión, revalidar contra ese endpoint antes de fijar nombres.

`deepseek-ai/DeepSeek-V4-Flash` es un modelo de razonamiento: puede consumir el presupuesto de `max_tokens` en el campo `reasoning` antes de emitir `content`, dejando la respuesta vacía si el límite es muy bajo.

Configuración vía variables de entorno (ver `.env.example`): `VLLM_CHAT_BASE_URL`, `VLLM_CHAT_MODEL`, `VLLM_EMBEDDING_BASE_URL`, `VLLM_EMBEDDING_MODEL`, `VLLM_API_KEY`, `VLLM_TIMEOUT`.

Para depurar prompts o validar payloads, la documentación institucional recomienda probar primero con `curl` directo (mejor estabilidad reportada que vía SDK) antes de integrar en agentes automáticos.

`gemma-4-31B` y `glm-ocr` (puertos `12559`/`12560`) también están expuestos por la universidad pero **no tienen cliente implementado aquí** (fuera del alcance actual: solo chat + embeddings). Nota: la doc institucional llama a ese modelo `gemma-4-31B`, pero `GET /v1/models` en el puerto `12559` reporta `google/gemma-4-12B-it` (12B, no 31B) — discrepancia de la doc/despliegue de origen, no de este repo; revalidar si se llega a integrar.

La universidad también ofrece un servidor **Ollama** independiente (`172.21.230.33:11434`, API nativa `/api/generate` y `/api/tags`, no OpenAI-compatible) para prototipado rápido con modelos livianos (ej. `llama3.1:8b`). Recomendación institucional: vLLM para integraciones institucionales/agentes/cargas concurrentes, Ollama para pruebas rápidas y modelos pequeños. No implementado en este repo.

Milvus: gRPC en `localhost:19530`, health en `localhost:9091`. etcd/minio quedan solo en la red interna del compose. Datos en `docker/volumes/` (ignorado por git). No hay linter ni suite de tests configurados todavía.

## Entornos multi-gestor

Las dependencias tienen como fuente de verdad `pyproject.toml`. Los tres archivos deben mantenerse en sync al agregar dependencias: `pyproject.toml` (uv), `requirements.txt` (pip, y del que depende `environment.yml`), `environment.yml` (conda/micromamba). Las libs de embeddings/RAG/viz van como *extras* opcionales en `pyproject.toml` (`embeddings-local`, `embeddings-openai`, `embeddings-voyage`, `rag`, `viz`, `notebook`), no en el núcleo, porque el enfoque está sin decidir.

## Contexto de infraestructura

En esta máquina corre un contenedor `uindex-db` (MySQL 8, puerto 3310) que es muy probablemente la base de datos de origen de los artículos — la fuente de `get_sqlalchemy_url()` en el script de muestreo. El nombre del repo (`clustering_n_rag_u_index`) refuerza esa relación con el proyecto "u-index".

## Punto importante: `scripts/sample_stratified_articles.py` no es ejecutable tal cual

El script importa módulos que **no existen en este repositorio**:

```python
from config import get_normalized_terms
from config.config import get_sqlalchemy_url
from src.utilities.text_utils import (
    limpiar_y_normalizar_texto,
    generate_mask_for_texto,
    generate_mask_with_exclusion,
)
```

También referencia `src/data_processing/data_processor.py` (en el docstring) para replicar la lógica de cálculo de flags ICOR. Ninguno de estos paths (`config/`, `src/utilities/`, `src/data_processing/`) existe aquí. El script asume que se ejecuta desde la raíz de un proyecto más grande (probablemente el proyecto de origen de la base de datos de artículos) que no ha sido incorporado a este directorio. Antes de intentar ejecutar o modificar este script, verifica si esos módulos deben copiarse/vincularse desde otro repositorio, o pregunta al usuario por el proyecto de origen — no los recrees por suposición.

El script también asume conexión a una base de datos relacional (vía SQLAlchemy) con tablas `articulos`, `fuentes`, `fechas`, `periodicos`, `secciones`, cuya configuración vendría de `get_sqlalchemy_url()`.

## Esquema del CSV (`data/raw/stratified_sample_2019_2026.csv`)

Columnas, en este orden:

```
id_articulo, id_pressreader, titulo, texto, seccion, fecha,
id_periodico, nombre_periodico, anio,
incertidumbre, economic_uncertainty, political_uncertainty,
icor_index, icor_v2_index, icor_v3_1_index
```

- `incertidumbre`, `economic_uncertainty`, `political_uncertainty`: flags booleanas (0/1) ya almacenadas en la BD de origen.
- `icor_index`, `icor_v2_index`, `icor_v3_1_index`: flags de detección de "corrupción" (ICOR) calculadas al vuelo por el script mediante matching de términos normalizados sobre `texto`, en tres versiones/variantes (base, v2 con exclusión, v3.1 con exclusión).

## Lógica de muestreo estratificado (para referencia al leer o modificar el script)

- Estratos = (año × periódico). Rango de años: 2019–2026 (`START_YEAR`/`END_YEAR` en el script).
- Cuota por estrato = `ceil(TARGET_TOTAL / n_estratos)`, con `TARGET_TOTAL = 10_000`.
- Si algún estrato tiene menos de `MIN_PER_STRATUM = 140` artículos disponibles, el script aborta (`sys.exit(1)`).
- Muestreo aleatorio con `RANDOM_STATE = 42` para reproducibilidad.
- Salida ordenada por `anio`, `nombre_periodico`, `id_articulo`.

## Idioma

Los nombres de variables, docstrings, prints y comentarios en el código existente están en español — mantener esa convención al editar `scripts/sample_stratified_articles.py` u otro código de este repo.
