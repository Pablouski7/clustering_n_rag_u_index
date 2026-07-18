# Sampling for clustering

Muestreo de artículos de la BD para hacer **clustering usando la sección como label**.

## Problema que resuelve

La sección es de **altísima cardinalidad**: la fuente (sobre todo El Universo desde 2020)
titula páginas temáticamente, así que la BD 2019-2026 tiene **~7,400 secciones crudas
distintas**, la mayoría con 1-3 artículos. Usar `seccion` cruda como label es inviable
(no se puede balancear ni entrenar sobre miles de clases casi vacías).

La solución tiene dos partes:

1. **Mapeo canónico** (`seccion → seccion_canonica`): normaliza y colapsa las ~7,400
   secciones a **~15 macro-secciones** limpias (Deportes, Economía, Política, Mundo,
   Seguridad, Sociedad, Cultura, Entretenimiento, Ciencia y Tecnología, Vida y Estilo,
   Local Guayaquil, Local Quito, Educación, + las no temáticas Portada, Opinión,
   Actualidad). Cobertura ~92.6%; el resto cae en `Otros`. El mapeo vive **solo en la capa
   de análisis** (no se escribe en la BD), coherente con
   `.agents/docs/seccion-contaminacion.md §6`.
2. **Muestreo balanceado** según distintos criterios (ver scripts).

Cada muestra conserva **ambas** columnas: `seccion` (original) y `seccion_canonica`.

## Scripts

| Script | Estrategia | Salida (en `data/clustering_samples/`) |
|---|---|---|
| `sample_stratified_articles.py` | Estratifica por **(año × periódico)**. No controla sección (hereda el desbalance de la fuente). | `stratified_sample_2019_2026.csv` |
| `sample_balanced_by_seccion.py` | Balancea por **sección canónica** (cuota igual por clase). Genera 2 muestras. | `stratified_by_seccion_all.csv` (17 clases, incl. no temáticas + `Otros`)<br>`stratified_by_seccion_thematic.csv` (13 clases temáticas) |
| `sample_grid_balanced.py` | Grid **(año × periódico) con tope por celda** + secciones aplanadas dentro de cada celda (round-robin). Añade una muestra de `Otros` para exploración. | `stratified_grid_2019_2026.csv` |

### ¿Cuál usar?

- **`sample_balanced_by_seccion`** → clustering donde importa que las clases de sección
  estén balanceadas (label plano). `all` incluye Portada/Opinión/Actualidad/`Otros`;
  `thematic` es solo temáticas limpias.
- **`sample_grid_balanced`** → dataset grande y representativo en el tiempo/periódico, con
  las secciones **suavizadas** (no perfectamente planas). Reduce el desbalance de sección de
  ~58× (pool crudo) a ~3×, manteniendo año/periódico parejos hasta donde el corpus permite
  (Diario Expreso 2019-2020 es intrínsecamente escaso, ~140-235 artículos).
- **`sample_stratified_articles`** → muestra representativa por año/periódico sin tocar
  sección (versión original, útil como referencia/auditoría).

## Uso

```bash
micromamba activate scraping
export PYTHONPATH=$PYTHONPATH:.

python scripts/sampling_for_clustering/sample_balanced_by_seccion.py
python scripts/sampling_for_clustering/sample_grid_balanced.py
python scripts/sampling_for_clustering/sample_stratified_articles.py
```

Todos leen la BD completa (`get_sqlalchemy_url`), aplican el mapeo canónico y calculan al
vuelo las flags ICOR (`icor_index`, `icor_v2_index`, `icor_v3_1_index`) replicando
`src/data_processing/data_processor.py`.

### Parámetros (constantes al inicio de cada script)

| Script | Constante | Default | Qué controla |
|---|---|---|---|
| `sample_balanced_by_seccion` | `TARGET_TOTAL` | 10,000 | Tamaño objetivo por muestra (cuota/clase = TARGET/n_clases) |
| `sample_grid_balanced` | `CELL_CAP` | 1,200 | Tope de artículos por celda año×periódico (~27k total) |
| `sample_grid_balanced` | `OTROS_SAMPLE` | 2,000 | Ejemplos de `Otros` añadidos para exploración (0 = ninguno) |
| `sample_stratified_articles` | `TARGET_TOTAL` / `MIN_PER_STRATUM` | 10,000 / 140 | Tamaño y mínimo por estrato (año×periódico) |

El mapeo canónico (`normalizar_seccion` + `_REGLAS`) vive en `sample_balanced_by_seccion.py`
y lo reutiliza `sample_grid_balanced.py`. Para afinar qué cae en `Otros`, editar `_REGLAS`.

## Columnas de salida

```
id_articulo, id_pressreader, titulo, texto, seccion, seccion_canonica,
fecha, id_periodico, nombre_periodico, anio,
incertidumbre, economic_uncertainty, political_uncertainty,
icor_index, icor_v2_index, icor_v3_1_index
```

## Notas

- `Otros` se **excluye** de los estratos balanceados en `sample_balanced_by_seccion`
  (excepto en la muestra `all`, donde entra como una clase más) y en el grid solo se añade la
  cantidad de `OTROS_SAMPLE` para exploración.
- **No concatenar** las muestras entre sí: `all` y `thematic` comparten ~76% de los
  artículos y usan cuotas distintas — concatenarlas duplica y rompe el balance. Para un
  dataset más grande, subir el parámetro de tamaño en **una** muestra.
