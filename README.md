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
scripts/
  check_milvus.py                  Prueba de humo de la base vectorial
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

## Regenerar la muestra

`scripts/sample_stratified_articles.py` depende de módulos de un proyecto de origen
que no viven en este repo (ver **CLAUDE.md**). No es ejecutable tal cual aquí.
