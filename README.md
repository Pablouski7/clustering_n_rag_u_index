# clustering_n_rag_u_index

Proyecto de titulación de maestría (USFQ). Objetivo: aplicar **embeddings**,
**clustering** y **RAG** sobre una muestra de artículos de prensa ecuatoriana
(Diario Expreso, El Universo, Primicias; 2019–2026), usando **Milvus** como
base de datos vectorial.

> Estado: **setup inicial**. El pipeline de embeddings/clustering/RAG aún no
> está implementado (el enfoque de embeddings está por decidir). Este repo deja
> listo el entorno y la base vectorial.

## Estructura

```
data/raw/          Muestra fuente (stratified_sample_2019_2026.csv, ~9.5k artículos)
data/processed/    Datos derivados (ignorado por git salvo .gitkeep)
src/
  embeddings/  clustering/  rag/   Módulos del pipeline (por implementar)
  llm_clients/                     Config + fábricas de clientes vLLM (chat/embeddings)
scripts/
  check_milvus.py                  Prueba de humo de la base vectorial
  check_vllm.py                    Prueba de humo de los endpoints vLLM (chat/embeddings)
  sample_stratified_articles.py    Script de referencia que generó la muestra (ver CLAUDE.md)
docker-compose.yml Milvus standalone (+ etcd, minio, y Attu opcional)
```

## Entorno Python

El repo está preparado para tres gestores. Elegí uno.

**micromamba / conda** (flujo por defecto en esta máquina):
```bash
micromamba activate base
pip install -r requirements.txt
# o crear un entorno dedicado:
micromamba env create -f environment.yml && micromamba activate clustering-rag-uindex
```

**uv:**
```bash
uv sync                 # instala el núcleo en .venv
uv sync --extra rag     # añadir extras: embeddings-local | embeddings-openai | rag | viz | notebook
```

**pip puro:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Las dependencias de embeddings/RAG se instalan como *extras* cuando se defina el
enfoque (ver `[project.optional-dependencies]` en `pyproject.toml`).

## Base vectorial (Milvus)

```bash
cp .env.example .env            # ajustar si hace falta
docker compose up -d            # levanta etcd + minio + milvus
docker compose ps               # esperar a que 'milvus' esté (healthy)
python scripts/check_milvus.py  # prueba de conectividad
```

- Milvus gRPC: `localhost:19530` · health: `localhost:9091`
- UI opcional (Attu): `docker compose --profile ui up -d` → http://localhost:8000
- Datos persistidos en `docker/volumes/` (ignorado por git).
- Parar: `docker compose down` (agregar `-v` para borrar datos).

## LLMs vía vLLM (servidor H200 de la universidad)

La universidad expone modelos vía vLLM con API compatible con OpenAI. Requiere
**VPN GlobalProtect activa** (sin ella, el tráfico hacia `172.28.230.10` se
descarta) y usa **HTTP** (no hay TLS configurado).

```bash
uv sync --extra embeddings-openai   # o: pip install "openai>=1.40" (ver requirements/extras)
cp .env.example .env                # ajustar VLLM_* si hace falta
python scripts/check_vllm.py        # prueba de conectividad (chat + embeddings)
```

- Config y fábricas de clientes: `src/llm_clients/` (`config.py`, `factory.py`).
- Variables de entorno relevantes (ver `.env.example`): `VLLM_CHAT_BASE_URL`,
  `VLLM_CHAT_MODEL`, `VLLM_EMBEDDING_BASE_URL`, `VLLM_EMBEDDING_MODEL`,
  `VLLM_API_KEY`, `VLLM_TIMEOUT`.
- Los IDs de modelo por defecto (`deepseek-ai/DeepSeek-V4-Flash`, `BAAI/bge-m3`)
  son los reportados por `GET {base_url}/models`, no los nombres cortos de la
  documentación original — verificar contra ese endpoint si vLLM cambia de versión.
- `deepseek-ai/DeepSeek-V4-Flash` es un modelo de razonamiento: consume tokens
  en el campo `reasoning` antes de emitir `content`, así que `max_tokens` debe
  ser generoso (ver `scripts/check_vllm.py`).

## Regenerar la muestra

`scripts/sample_stratified_articles.py` depende de módulos de un proyecto de origen
que no viven en este repo (ver **CLAUDE.md**). No es ejecutable tal cual aquí.
