# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Estado del repositorio

Repositorio de tesis de maestría USFQ. Objetivo del proyecto: **embeddings + clustering + RAG** sobre la muestra de artículos de prensa, usando **Milvus** como base vectorial. Está en fase de **setup**: el entorno y la base vectorial quedan listos, pero el pipeline de embeddings/clustering/RAG **aún no está implementado** (el enfoque de embeddings está por decidir).

Contenido relevante:

- `data/raw/stratified_sample_2019_2026.csv` — muestra estratificada de ~9,551 artículos periodísticos (2019–2026), de tres periódicos ecuatorianos: **Diario Expreso**, **El Universo**, **Primicias**. Se versiona en git (es el dataset fuente).
- `data/raw/stratified_grid_2019_2026.csv` — muestra grid (año × periódico con tope por celda + secciones aplanadas), ~28,777 artículos. **Es la que usa `notebooks/eda.ipynb`**; incluye la columna `seccion_canonica`. Generada por `scripts/sampling_for_clustering/sample_grid_balanced.py`.
- `notebooks/eda.ipynb` — pipeline principal, en cinco secciones: **1. EDA** (sobre texto crudo) → **2. Data Wrangling** (normalización, dedupe, sección canónica, descarte de artículos de ≤40 palabras → 26,210 artículos) → **3. Embeddings** (Doc2Vec / BGE-M3 / BETO sobre el corpus limpio, viz 3D PCA + UMAP) → **4. Clustering** (HDBSCAN por método + métricas) → **5. AE/VAE** (exploratorio). Ver sección "Notebook eda.ipynb" abajo.
- `data/processed/`, `data/embeddings/` — datos derivados, ignorados por git.
- `scripts/sample_stratified_articles.py` — script de referencia que generó el CSV (ver sección abajo; no ejecutable tal cual aquí).
- `src/embeddings/beto.py` — embeddings con **BETO** (`dccuchile/bert-base-spanish-wwm-cased`, 768 dims) en PyTorch: chunking en ventanas solapadas de 512 tokens + mean pooling enmascarado intra-chunk + media inter-chunk. Ver sección "BETO" abajo.
- `scripts/embed_beto.py` — genera `data/embeddings/beto.npy` sobre la submuestra del EDA.
- `src/embeddings/nemotron.py` — embeddings con **Nemotron VL** (`nvidia/llama-nemotron-embed-vl-1b-v2`, 2048 dims) en modo solo texto. Ver sección "Nemotron VL" abajo.
- `scripts/embed_nemotron.py` — genera `data/embeddings/nemotron.npy` (con reanudación por bloques).
- `src/clustering/`, `src/rag/` — módulos del pipeline, por implementar.
- `src/llm_clients/` — config (`config.py`) y fábricas de clientes (`factory.py`) para los endpoints vLLM (chat/embeddings) del servidor H200 de la universidad, vía SDK `openai` (API compatible con OpenAI). Ver sección "LLMs vía vLLM" abajo.
- `scripts/check_milvus.py` — prueba de humo de la base vectorial.
- `scripts/check_vllm.py` — prueba de humo de los endpoints vLLM (chat y embeddings).

## Comandos

```bash
# Entorno (esta máquina usa micromamba ai_env; ver README para uv/conda)
micromamba activate ai_env && pip install -r requirements.txt

# Base vectorial
docker compose up -d              # levanta etcd + minio + milvus
docker compose --profile ui up -d # además Attu (UI en http://localhost:8000)
docker compose --profile embeddings run --rm embeddings   # job de embeddings (ver abajo)
docker compose ps                 # verificar que 'milvus' esté (healthy)
python scripts/check_milvus.py    # prueba de conectividad
docker compose down               # parar (agregar -v para borrar datos)

# LLMs vía vLLM (requiere VPN GlobalProtect activa, ver README)
pip install "openai>=1.40"        # extra embeddings-openai en pyproject.toml
python scripts/check_vllm.py      # prueba de conectividad (chat + embeddings)
```

## BETO (chunking + pooling)

`src/embeddings/beto.py` implementa embeddings de artículo completo con BETO pese a su ventana de 512 tokens: el texto se parte en ventanas solapadas (`stride_ratio=0.2`), cada chunk se resume con mean pooling enmascarado sobre la última capa, y los vectores de chunk se promedian.

**El pooling inter-chunk fue attention pooling y se simplificó a media tras medirlo.** Una atención sin parámetros (query = centroide de los chunks) resultó indistinguible de la media en este corpus: coseno 0.9999 entre ambos embeddings, r = 0.9988 entre matrices de distancia. La causa es que los artículos de prensa apenas exceden la ventana: **mediana de 1 chunk, 65% de artículos en un solo chunk**. No re-introducir attention pooling aquí sin volver a medir la distribución de chunks; si el corpus cambia a documentos largos, la decisión puede invertirse.

Corre en CPU (`torch` CPU-only en `ai_env`). Los chunks de todos los artículos se aplanan en un solo stream de batches (`batch_chunks`) porque la mayoría de artículos aporta solo 1-3 chunks.

Advertencia al interpretar resultados: BETO no tiene fine-tuning contrastivo de similitud (a diferencia de MiniLM y BGE-M3), y su espacio es anisotrópico — las cosenos entre documentos arbitrarios rondan 0.94, lo que comprime las distancias y tiende a penalizar el silhouette. Antes de concluir que el chunking o el pooling fallan, revisar la versión con whitening.

`hc_clustering.ipynb` incluye una función `whitening()` (centrado + `Σ^(-1/2)` truncada a 128 componentes, estilo Su et al. 2021) y evalúa "BETO (whitened)" como técnica aparte para que la comparación quede medida. La truncación a 128 no es cosmética: con ~1,512 documentos y 768 dims hay menos muestras que parámetros de covarianza, así que estimar `Σ` completa es ruidoso. Validado sobre BGE-M3: coseno medio entre documentos 0.340 → -0.001.

```bash
micromamba activate ai_env && python scripts/embed_beto.py   # ~10 min en CPU, cachea data/embeddings/beto.npy
```

## Nemotron VL (solo texto, sin chunking)

`src/embeddings/nemotron.py` usa `nvidia/llama-nemotron-embed-vl-1b-v2` (Eagle VLM: Llama 3.2 1B + SigLip2 400M, ~1.7 B parámetros, 2048 dims) por su rama de **texto** (`encode_document`) — el corpus no tiene imágenes de página, así que la rama de visión se carga pero nunca se ejecuta.

**No hay chunking, a diferencia de BETO, y es deliberado.** La ventana evaluada del modelo es de 10,240 tokens y el artículo más largo del corpus mide ~24k caracteres ≈ 6,900 tokens, así que **cada artículo entra completo en un solo forward pass** (verificado sobre el CSV, 0 artículos exceden la ventana). Eso lo convierte en la única técnica que ve el texto íntegro sin truncar (TF-IDF/MiniLM/BGE-M3 cortan a 2,000 caracteres) ni promediar chunks (BETO). No añadir chunking aquí sin volver a medir la distribución de longitudes.

Su valor analítico en la comparación es **separar cobertura de artefacto de pooling**: si en la sección de η² de `hc_clustering.ipynb` BETO sale alto y Nemotron VL bajo, el sesgo de longitud lo produce el promediado de chunks y no el hecho de leer el artículo completo.

Advertencias al interpretar resultados: (1) fue entrenado con objetivo contrastivo para *retrieval* consulta→página, no para similitud documento–documento, que es lo que pide el clustering; (2) sus 2048 dims son las más altas de la comparación, y sobre ~1,512 documentos eso tensiona las distancias de Ward y t-SNE.

**Costo.** Es de lejos el método más caro: en CPU tarda **horas** (frente a ~10 min de BETO). Se carga en `bfloat16` (~3.4 GB de pesos en vez de ~6.8) y los textos se ordenan por longitud para que el padding por lote no desperdicie cómputo. `scripts/embed_nemotron.py` persiste el avance en `nemotron_parcial.npy` cada 50 artículos para poder reanudar; al terminar lo borra y guarda `nemotron.npy`.

```bash
micromamba activate ai_env && python scripts/embed_nemotron.py   # horas en CPU, reanudable

# Recomendado: contenedor con GPU (ver docker/Dockerfile.embeddings)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
  --profile embeddings run --rm embeddings
```

**Esta máquina no puede correrlo tal cual**: `ai_env` tiene torch CPU-only, el disco está al
95% (~12 GB libres frente a ~9 GB de imagen + 3.4 GB de pesos) y quedan ~3 GB de RAM. Por eso
el job vive en Docker, pensado para ejecutarse en otro host. `data/embeddings/nemotron.npy`
**aún no existe**: las celdas que lo cargan en los tres notebooks fallarán con un
`FileNotFoundError` explícito hasta que se genere.

Requiere `transformers>=4.56` y `trust_remote_code=True` (el modelo trae código propio). Se fuerza `attn_implementation="sdpa"`: flash-attention no compila para CPU.

## Notebook `eda.ipynb`

Corre de principio a fin sobre `stratified_grid_2019_2026.csv`. Se ejecuta en un **servidor con GPU** (las secciones 3 y 4 no son viables en esta máquina). El orden de las secciones es la invariante: el EDA mide sobre texto **crudo** y no modifica `data`; a partir de la sección 2 sí.

**Sección canónica.** `mapear_seccion_canonica` + `_REGLAS` están **copiadas** de `scripts/sampling_for_clustering/sample_balanced_by_seccion.py`, no importadas: ese script depende de `config.get_sqlalchemy_url` y `src.utilities.text_utils`, que no existen aquí. Al cambiar las reglas hay que hacerlo en ambos sitios. Regenerar la canónica sobre la sección normalizada da **0 filas distintas** frente a la columna del CSV (verificado en el notebook); se recalcula para que la lógica sea auditable y el ~7% de `Otros` se pueda afinar sin volver a la BD.

**`seccion_canonica` es label ruidoso, no ground truth**: la asigna el periódico por criterio editorial/maquetación y el mapeo es por substring. Las métricas externas se reportan en dos vistas — 17 clases y solo las 13 temáticas (sin `Portada`/`Opinión`/`Actualidad`/`Otros`, que son formato o residuo). Un ARI bajo es ambiguo por diseño.

**Filtro de longitud:** se descartan artículos de ≤40 palabras (2,566, el 8.9%): teletipos y pies de foto. Corpus resultante **26,210** artículos, con `reset_index(drop=True)` porque la sección 4 indexa por posición.

**Cachés.** `cargar_cache_full`/`guardar_cache_full` validan contra `data["id_articulo"]` **y contra `huella_corpus(params)`**, un SHA-256 del contenido de `textos` más los parámetros del método, persistido en `{nombre}_full_meta.json`. Los ids solos no bastaban: cambiar `norm_text` o la tokenización de un método deja los ids intactos, con lo que el caché se daba por bueno y el notebook seguía con vectores obsoletos sin error visible. Los checkpoints `*_full_parcial.npy` llevan su propio sidecar `*_full_parcial_huella.txt` y se descartan solos si no coincide, así que ya **no** hay que borrarlos a mano.

Un caché generado antes de esta salvaguarda no tiene `_meta.json` y se ignora con un mensaje explícito. Si consta que el texto no cambió, `adoptar_cache_full(nombre, params)` le estampa la huella sin recalcularlo — es la única vía que puede sellar como válidos unos vectores obsoletos, así que usarla solo con esa certeza.

**Tokenización.** `_PATRON_TOKEN` (celda 50, el patrón por defecto de `TfidfVectorizer`) es el tokenizador compartido por Doc2Vec, c-TF-IDF y la coherencia de tópicos. BGE-M3 y BETO **no** lo usan: traen su tokenizador subpalabra y para ellos la puntuación es señal, por eso `norm_text` la conserva. No sustituir esto por `texto.split()`: deja la puntuación pegada al token (`"gobierno,"` ≠ `"gobierno"`), lo que duplica el vocabulario, hace que `min_count=5` descarte ~10% de los tokens de entrenamiento en vez de ~5% —penalizando a Doc2Vec por preprocesamiento— y desalinea el vocabulario entre los términos c-TF-IDF y los conteos con los que se calcula el NPMI.

**t-SNE está excluido a propósito** de la viz: con ~26k documentos son decenas de minutos por método y `sklearn.manifold.TSNE` es CPU puro (la GPU no lo acelera). Si se quiere recuperar, la vía es `cuml.TSNE(method='fft')`, que hasta donde se ha verificado solo implementa `n_components=2`. UMAP sí se ajusta con el corpus completo.

**Métricas de clustering:** el silhouette se calcula sobre el espacio UMAP-10 **y** sobre el embedding original en coseno, porque el primero está inflado por construcción (UMAP separa grupos aunque la separación no exista en el espacio original) y solo el segundo es comparable entre métodos de distinta dimensión. Silhouette y DBCV van sobre submuestra (`N_MUESTRA_METRICA = 8000`): necesitan la matriz de distancias completa, que con 26k puntos son ~5 GB.

Las métricas están **jerarquizadas en cuatro niveles** y el código y los plots respetan esa jerarquía (ver el markdown de la sección 4, que argumenta cada degradación):

- **Nivel 1, deciden:** AMI (vs `seccion` y `seccion_canonica`), DBCV, silhouette coseno, coherencia NPMI.
- **Nivel 2, contexto obligatorio:** `n_clusters`, `%_ruido`, `conf_media`, `pct_conf_baja`.
- **Nivel 3, se reportan:** ARI, NMI, V-measure, homogeneidad, completitud, coherencia $C_v$.
- **Nivel 4, anexo:** Davies-Bouldin, Calinski-Harabasz, silhouette sobre UMAP-10.

**AMI y no NMI/V-measure** porque estas dos no corrigen por azar y crecen con el número de clusters, lo que las hace engañosas en el barrido por cortes del árbol (premiarían siempre la partición más fragmentada). **ARI se conserva pero no decide**: la brecha entre AMI alto y ARI bajo es la evidencia de que el clustering encontró subtemas dentro de las secciones editoriales. **DB y CH al anexo** no por la dimensionalidad (se calculan sobre el UMAP-10, donde la euclídea sí es significativa) sino porque CH crece casi monótonamente con k, DB duplica la información del silhouette en ese mismo espacio, y HDBSCAN produce clusters no convexos. Se eliminó la subdivisión 17 vs 13 clases y la métrica de pureza.

**Toda la sección 4 vive en `analizar_espacio(emb, nombre, data, ...)`**, que corre clustering + barrido + soft clustering + métricas + plots sobre cualquier matriz `[N, D]` y devuelve un dict; `comparar_espacios(resultados)` apila los resúmenes en las tres tablas. La sección 5 la invoca sobre los latentes AE/VAE, que entran en la misma comparación. La coherencia solo se calcula sobre la partición EOM (no en cada nivel del barrido) por costo.

## LLMs vía vLLM (servidor H200)

La universidad expone `deepseek-v4-flash` (chat/razonamiento), `BGE-M3` (embeddings), `gemma-4-31B` y `glm-ocr` vía vLLM con API compatible con OpenAI (HTTP, sin TLS) en `172.28.230.10`. Requiere VPN GlobalProtect activa; sin ella las peticiones fallan por timeout/conexión, no por autenticación (no hay auth real todavía).

Los IDs de modelo reales reportados por `GET {base_url}/models` difieren de los nombres cortos de la documentación: `deepseek-ai/DeepSeek-V4-Flash` (no `deepseek-v4-flash`) y `BAAI/bge-m3` (no `BGE-M3`) — estos son los valores por defecto en `src/llm_clients/config.py`. Si vLLM cambia de versión, revalidar contra ese endpoint antes de fijar nombres.

`deepseek-ai/DeepSeek-V4-Flash` es un modelo de razonamiento: puede consumir el presupuesto de `max_tokens` en el campo `reasoning` antes de emitir `content`, dejando la respuesta vacía si el límite es muy bajo.

**BGE-M3 se usa aquí solo en su modalidad dense.** BGE-M3 soporta nativamente tres salidas (dense, sparse/lexical, multi-vector ColBERT-style), pero este proyecto la consume vía `cliente.embeddings.create(...)` contra `/v1/embeddings` de vLLM, que devuelve únicamente el vector pooled denso (1024 dims) — no expone sparse ni multi-vector. Confirmado contra el OpenAPI spec del servidor: `/v1/embeddings` acepta `EmbeddingCompletionRequest`/`EmbeddingChatRequest` y no tiene campos para esas otras salidas. El servidor sí expone `/pooling` (con `task: token_embed`, lo más cercano a multi-vector) y `/score`/`/rerank`, pero el código del proyecto no los usa. Si en la fase de RAG se quiere retrieval híbrido (denso + sparse), hay que usar `/pooling` con `task=token_embed` o correr BGE-M3 en local con `FlagEmbedding`, porque vLLM no replica el `encode()` nativo de BGE-M3 con sus tres salidas simultáneas.

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
