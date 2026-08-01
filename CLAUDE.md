# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Estado del repositorio

Repositorio de tesis de maestría USFQ. Objetivo del proyecto: **embeddings + clustering + RAG** sobre la muestra de artículos de prensa, usando **Milvus** como base vectorial. **Embeddings y clustering están implementados**, íntegramente dentro de `notebooks/pipeline.ipynb`; la fase de **RAG está pendiente** (el entorno y la base vectorial ya están listos).

**Todo el pipeline vive en un solo notebook, y es deliberado.** No hay módulos de embeddings en `src/`: cada modelo es una `EmbeddingSpec` (config inmutable) consumida por `EmbeddingRunner` dentro del notebook, con caché en `data/embeddings/`. `src/` conserva solo `llm_clients/` (que el notebook importa para el backend vLLM) y los paquetes vacíos `clustering/` y `rag/`. Al agregar un modelo nuevo, seguir el patrón del notebook — no reintroducir `src/embeddings/`.

Contenido relevante:

- `data/raw/stratified_grid_2019_2026.csv` — muestra grid (año × periódico con tope por celda + secciones aplanadas), ~28,777 artículos. **Es la única que usa el notebook**; incluye la columna `seccion_canonica`. Generada por `scripts/sampling_for_clustering/sample_grid_balanced.py`.
- `notebooks/pipeline.ipynb` — pipeline principal, en cinco secciones: **1. EDA** (sobre texto crudo) → **2. Data Wrangling** (normalización, dedupe, sección canónica, descarte de artículos de ≤40 palabras → 26,210 artículos) → **3. Embeddings** (BGE-M3 / Jina-v3 / E5-large / Qwen3-Embedding-0.6B / Jina-v2-ES sobre el corpus limpio, viz 3D PCA + UMAP) → **4. Clustering** (HDBSCAN por método + métricas) → **5. AE/VAE** (exploratorio, sobre los cinco). Ver sección "Embeddings (EmbeddingSpec / EmbeddingRunner)" abajo.
- `data/raw/stratified_sample_2019_2026.csv`, `stratified_by_seccion_all.csv`, `stratified_by_seccion_thematic.csv` — muestras de iteraciones anteriores. **Ya no las usa nada**; se conservan porque regenerarlas exige la BD MySQL de origen. No construir código nuevo sobre ellas.
- `data/embeddings/` — vectores cacheados por el notebook (`{spec.cache_name}_full.npy` + `_ids.npy` + `_meta.json`), versionados con Git LFS. `data/processed/` — datos derivados. **Los archivos actuales (`doc2vec_full*`, `beto_full*`, `bge_m3_full*` con el nombre corto viejo) son de la generación anterior de 3 modelos y no coinciden con los `cache_name` actuales** (`bge_m3_native`, `jina_v3_separation`, `multilingual_e5_large_query`, `qwen3_06b_topic_instruction`/`_no_instruction`, `jina_v2_base_es`); hay que regenerarlos.
- `src/clustering/`, `src/rag/` — módulos del pipeline, por implementar (vacíos).
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
docker compose ps                 # verificar que 'milvus' esté (healthy)
python scripts/check_milvus.py    # prueba de conectividad
docker compose down               # parar (agregar -v para borrar datos)

# LLMs vía vLLM (requiere VPN GlobalProtect activa, ver README)
pip install "openai>=1.40"        # extra embeddings-openai en pyproject.toml
python scripts/check_vllm.py      # prueba de conectividad (chat + embeddings)
```

## Embeddings (EmbeddingSpec / EmbeddingRunner)

Sección 3 del notebook. `EmbeddingSpec` (dataclass congelada) es el contrato completo de un método texto→vector — model_id, ventana de contexto, batch size, prefijo/prompt/task, backend — y **forma parte de la huella del caché** vía `spec.cache_params()`. `EmbeddingRunner` (context manager) carga el modelo, corre inferencia, guarda checkpoints reanudables y libera GPU al salir. El chunking y la agregación de chunks son funciones puras (`dividir_ids_con_cobertura`, `agregar_chunks`) compartidas por los cinco modelos.

`MODELOS_PRINCIPALES` (comparación principal):

| Modelo | `model_id` | Contexto | Tratamiento | Backend |
|---|---|---:|---|---|
| BGE-M3 | `BAAI/bge-m3` | 8192 | dense embedding nativo | `vllm` (servidor H200 remoto) |
| Jina-v3 | `jinaai/jina-embeddings-v3` | 8192 | `encode(..., task="separation")` | `sentence_transformers` (GPU local) |
| E5-large | `intfloat/multilingual-e5-large` | 512 | prefijo `"query: "` + chunking | `sentence_transformers` (GPU local) |
| Qwen3-instruct | `Qwen/Qwen3-Embedding-0.6B` | 32768 | instrucción temática neutral (`QWEN_CLUSTER_PROMPT`), `padding_side="left"` | `sentence_transformers` (GPU local) |
| Jina-v2-ES | `jinaai/jina-embeddings-v2-base-es` | 8192 | embedding bilingüe es-en, 768 dims | `sentence_transformers` (GPU local) |

`ABLACIONES_EMBEDDING` agrega **Qwen3-no-instruct** (mismo modelo sin el prompt, vía `dataclasses.replace`) como ablación separada — no compite como sexto candidato en la selección del ganador. `MODELOS_AE_VAE = list(MODELOS_PRINCIPALES)`: la sección 5 corre AE/VAE sobre los cinco, no sobre un subconjunto hard-coded.

**Solo BGE-M3 corre remoto** (vía `src/llm_clients.get_embedding_client`, contra el servidor vLLM); los otros cuatro cargan pesos localmente con `SentenceTransformer` y requieren GPU local — de ahí que su huella de caché fije `precision="remote"` para BGE-M3 (no debe cambiar si la máquina local tiene o no GPU) y `"float16"/"float32"` para los demás según `torch.cuda.is_available()`.

**Chunking y agregación son comunes a los cinco.** `dividir_ids_con_cobertura` parte los ids del artículo en ventanas solapadas (`overlap_tokens`, default 64) y devuelve, por chunk, cuántos tokens son *cobertura nueva* — no la longitud total de la ventana — para no pesar doble el overlap ni sobrerrepresentar el último chunk corto. `agregar_chunks` normaliza L2 cada vector de chunk y promedia ponderando por esa cobertura nueva. `_overhead_tokens` descuenta tokens especiales + prefijo/prompt del presupuesto antes de chunkear, con un margen fijo (`TOKEN_BUDGET_MARGIN = 4`) por si decode→retokenize desplaza la frontera.

`validar_embeddings` corta en seco si el shape no es `(n, spec.output_dim)`, si hay NaN/inf, o si las normas no son ≈1 (tolerancia `2e-3`) — así un fallo de pooling no pasa desapercibido como en la generación anterior. Al importar, el notebook aborta si `sentence-transformers < 3.1.0` (requerido por Jina v3) o `transformers < 4.51.0` (requerido por Qwen3 Embedding).

## Métodos evaluados y descartados

El pipeline actual compara **BGE-M3, Jina-v3, E5-large, Qwen3-Embedding-0.6B (+ ablación sin instrucción) y Jina-v2-ES** con **HDBSCAN**. Esto **reemplazó** la comparación anterior de **Doc2Vec, BGE-M3 y BETO** (commit `2873018`, "ejecución completa del pipeline con AE/VAE sobre los tres embeddings" — último commit con Doc2Vec/BETO intactos; el pivote a los cinco modelos actuales ocurrió en `711ce4b`). BETO usaba chunking con mean pooling enmascarado y ventana de 512 tokens; el pooling inter-chunk fue attention pooling y se simplificó a media tras medir que ambos daban coseno 0.9999 en este corpus (mediana de 1 chunk por artículo). Si hace falta recuperar Doc2Vec o BETO, están en `2873018`.

**Whitening tampoco está en el pipeline actual.** Notebooks previos a `2873018` evaluaban "BETO (whitened)" con una función `whitening()` (centrado + `Σ^(-1/2)` truncada a 128 componentes, estilo Su et al. 2021), validada sobre BGE-M3: coseno medio entre documentos 0.340 → -0.001. Esa comparación vive en el commit `4022f6d` (`notebooks/hc_clustering.ipynb`). Ojo al reusarla: la truncación a 128 se eligió para ~1,512 documentos, donde había menos muestras que parámetros de covarianza; con 26k el criterio hay que rehacerlo.

Iteraciones aún más antiguas (mismo commit `4022f6d`, notebooks `hc_clustering`/`gmm_clustering`/`*_autoencoder`, ya eliminados) también evaluaron **MiniLM**, **Nemotron VL** (`nvidia/llama-nemotron-embed-vl-1b-v2`, 2048 dims), **clustering jerárquico Ward** y **GMM**, sobre una submuestra de ~1,512 documentos. Nada de eso está en el repo hoy: al proponer alguno de esos métodos, tratarlo como decisión ya tomada y revisar primero ese commit en vez de reimplementarlo.

## Notebook `pipeline.ipynb`

Corre de principio a fin sobre `stratified_grid_2019_2026.csv`. Se ejecuta en un **servidor con GPU** (las secciones 3 y 4 no son viables en esta máquina). El orden de las secciones es la invariante: el EDA mide sobre texto **crudo** y no modifica `data`; a partir de la sección 2 sí.

**Sección canónica.** `mapear_seccion_canonica` + `_REGLAS` están **copiadas** de `scripts/sampling_for_clustering/sample_balanced_by_seccion.py`, no importadas: ese script depende de `config.get_sqlalchemy_url` y `src.utilities.text_utils`, que no existen aquí. Al cambiar las reglas hay que hacerlo en ambos sitios. Regenerar la canónica sobre la sección normalizada da **0 filas distintas** frente a la columna del CSV (verificado en el notebook); se recalcula para que la lógica sea auditable y el ~7% de `Otros` se pueda afinar sin volver a la BD.

**`seccion_canonica` es label ruidoso, no ground truth**: la asigna el periódico por criterio editorial/maquetación y el mapeo es por substring. Las métricas externas se reportan en dos vistas — 17 clases y solo las 13 temáticas (sin `Portada`/`Opinión`/`Actualidad`/`Otros`, que son formato o residuo). Un ARI bajo es ambiguo por diseño.

**Filtro de longitud:** se descartan artículos de ≤40 palabras (2,566, el 8.9%): teletipos y pies de foto. Corpus resultante **26,210** artículos, con `reset_index(drop=True)` porque la sección 4 indexa por posición.

**Cachés.** `cargar_cache_full`/`guardar_cache_full` validan contra `data["id_articulo"]` **y contra `huella_corpus(params)`**, un SHA-256 del contenido de `textos` más los parámetros del método, persistido en `{nombre}_full_meta.json`. Los ids solos no bastaban: cambiar `norm_text` o la tokenización de un método deja los ids intactos, con lo que el caché se daba por bueno y el notebook seguía con vectores obsoletos sin error visible. Los checkpoints `*_full_parcial.npy` llevan su propio sidecar `*_full_parcial_huella.txt` y se descartan solos si no coincide, así que ya **no** hay que borrarlos a mano.

Un caché generado antes de esta salvaguarda no tiene `_meta.json` y se ignora con un mensaje explícito. Si consta que el texto no cambió, `adoptar_cache_full(nombre, params)` le estampa la huella sin recalcularlo — es la única vía que puede sellar como válidos unos vectores obsoletos, así que usarla solo con esa certeza.

**Tokenización.** `_PATRON_TOKEN` (el patrón por defecto de `TfidfVectorizer`) es el tokenizador compartido por c-TF-IDF y la coherencia de tópicos (`tokens_coh`) — comparten vocabulario a propósito para que los términos de cada cluster y los conteos con los que se calcula el NPMI estén alineados. Ninguno de los cinco modelos de embedding lo usa: cada uno trae su tokenizador subpalabra propio (vía `AutoTokenizer.from_pretrained(spec.model_id, ...)`), y para ellos la puntuación es señal, por eso `norm_text` la conserva. No sustituir `_PATRON_TOKEN` por `texto.split()`: deja la puntuación pegada al token (`"gobierno,"` ≠ `"gobierno"`), lo que duplica el vocabulario y desalinea los términos c-TF-IDF de los conteos del NPMI.

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

## Punto importante: los scripts de `scripts/sampling_for_clustering/` no son ejecutables tal cual

Se conservan como documentación de cómo se generaron los CSV de `data/raw/` (ver el README de esa carpeta para la correspondencia script → CSV). `sample_stratified_articles.py` importa módulos que **no existen en este repositorio**:

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

## Esquema de los CSV de `data/raw/`

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

Los nombres de variables, docstrings, prints y comentarios en el código existente están en español — mantener esa convención al editar el notebook o cualquier otro código de este repo.
