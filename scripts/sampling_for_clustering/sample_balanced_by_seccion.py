#!/usr/bin/env python3
"""
Muestra de artículos balanceada por *sección canónica* para clustering.

A diferencia de scripts/sample_stratified_articles.py (estratifica por año×periódico),
este script balancea por sección: la fuente titula páginas temáticamente y produce
miles de secciones distintas (~7,400 en la BD 2019-2026), inusables como label crudo.
Por eso se mapea cada sección a una *sección canónica* (~15 macro-secciones) y se
extrae una cuota igual por clase.

Se conservan en la salida TANTO la sección original (`seccion`) COMO la canónica
(`seccion_canonica`). Se generan DOS muestras:
  - stratified_by_seccion_all.csv      → incluye secciones no temáticas.
  - stratified_by_seccion_thematic.csv → excluye Portada/Opinión/Actualidad.

El mapeo vive en la capa de análisis (NO se escribe en la BD), consistente con
`.agents/docs/seccion-contaminacion.md §6`.
"""

import re
import sys
import unicodedata
from math import ceil
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from sqlalchemy import create_engine

from config import get_normalized_terms
from config.config import get_sqlalchemy_url
from src.utilities.text_utils import (
    limpiar_y_normalizar_texto,
    generate_mask_for_texto,
    generate_mask_with_exclusion,
)

START_YEAR = 2019
END_YEAR = 2026
TARGET_TOTAL = 10_000  # tamaño objetivo de cada muestra (cuota/clase se deriva)
MIN_PER_CLASS = 150  # avisa si una clase queda por debajo
RANDOM_STATE = 42

# Secciones canónicas que NO son temáticas (mezclan temas → ruido como label
# de clustering temático). Se excluyen en la muestra "thematic".
NO_TEMATICAS = {"Portada", "Opinión", "Actualidad"}

QUERY = f"""
SELECT a.id_articulo, a.id_pressreader, a.titulo, a.texto,
       s.nombre_seccion AS seccion,
       fe.fecha, p.id_periodico, p.nombre_periodico,
       a.incertidumbre, a.economic_uncertainty, a.political_uncertainty
FROM articulos a
JOIN fuentes f    ON a.fuente_id = f.id_fuente
JOIN fechas fe    ON f.fecha_id = fe.id_fecha
JOIN periodicos p ON f.periodico_id = p.id_periodico
LEFT JOIN secciones s ON a.seccion_id = s.id_seccion
WHERE fe.fecha >= '{START_YEAR}-01-01' AND fe.fecha < '{END_YEAR + 1}-01-01'
"""


# --------------------------------------------------------------------------- #
# Mapeo canónico de sección
# --------------------------------------------------------------------------- #
def normalizar_seccion(s: str) -> str:
    """Normaliza el nombre de sección para el matching de reglas.

    Quita soft-hyphens, aplica NFKC, unifica separadores (& → ' y ', '-' → ' '),
    colapsa espacios y pasa a minúsculas. Colapsa duplicados triviales como
    'Vida&estilo' / 'Vida - Estilo' / 'Cien\xadcia'.
    """
    if not isinstance(s, str):
        return ""
    s = s.replace("­", "")  # soft-hyphen
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("&", " y ").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


# Reglas por palabra clave (substring sobre el nombre normalizado). Orden =
# prioridad: la primera que matchea gana. Cubren la cabeza y la cola larga.
_REGLAS = [
    ("Deportes", ("deporte", "jugada", "marcador", "futbol", "fútbol",
                  "eliminatoria", "mundial", "olimp", "diversión", "diversion",
                  "juegos", "liga", "copa")),
    ("Economía", ("económ", "economia", "economía", "negocio", "entorno económico",
                  "enfoque económico", "mercado", "finanz")),
    ("Política", ("polít", "politic", "legislat", "elecc", "asamblea",
                  "coyuntura nacional", "acontecer nacional", "hechos del país",
                  "debate", "gobierno", "presidencial")),
    ("Seguridad", ("segurid", "suceso", "delict", "violent", "muerte", "crimen",
                   "expediente", "narco", "conflicto")),
    ("Ciencia y Tecnología", ("ciencia", "tecnolog", "tecno", "digital", "innovac")),
    ("Vida y Estilo", ("vida y estilo", "vida estilo", "vidayestilo", "estilo",
                       "gastronom", "moda", "salud", "bienestar", "en ruta",
                       "hogar", "familia")),
    ("Cultura", ("cultura", "música", "musica", "cine", "arte", "libro",
                 "literatura", "gente", "patrimonio")),
    ("Entretenimiento", ("entreten", "farándula", "farandula", "trending",
                         "qué ver", "que ver", "espectác", "espectac", "tv",
                         "televisión", "viral")),
    ("Mundo", ("mundo", "internacional", "panorama internacional", "el país",
               "el mundo", "global", "migra")),
    ("Local Guayaquil", ("guayaquil", "guayas", "gran guayaquil")),
    ("Local Quito", ("quito", "capital", "los valles", "pichincha")),
    ("Educación", ("educ", "universidad", "escolar")),
    ("Sociedad", ("sociedad", "comunidad", "intercultural", "información general",
                  "informacion general", "ecología", "ecologia", "ambiente",
                  "en la ciudad")),
    # No temáticas (se evalúan al final para no capturar prefijos temáticos).
    ("Portada", ("portada",)),
    ("Opinión", ("opinión", "opinion", "lectores", "editorial", "columna")),
    ("Actualidad", ("actualidad", "lo último", "lo ultimo", "hoy", "última hora")),
]


def mapear_seccion_canonica(seccion: str) -> str:
    """Mapea una sección cruda a su sección canónica (o 'Otros')."""
    norm = normalizar_seccion(seccion)
    if not norm:
        return "Otros"
    for canonica, claves in _REGLAS:
        if any(clave in norm for clave in claves):
            return canonica
    return "Otros"


# --------------------------------------------------------------------------- #
# Carga y auditoría
# --------------------------------------------------------------------------- #
def load_articles() -> pd.DataFrame:
    """Carga todos los artículos del rango con todas sus columnas de la BD."""
    print(f"🔄 Cargando artículos {START_YEAR}-{END_YEAR} desde la BD...")
    engine = create_engine(get_sqlalchemy_url())
    df = pd.read_sql(QUERY, engine)
    engine.dispose()

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["anio"] = df["fecha"].dt.year
    df["seccion_canonica"] = df["seccion"].apply(mapear_seccion_canonica)
    print(f"✅ {len(df):,} artículos cargados")
    return df


def auditar_mapeo(df: pd.DataFrame) -> None:
    """Imprime la distribución por sección canónica y la cobertura del mapeo."""
    total = len(df)
    counts = df["seccion_canonica"].value_counts()
    otros = counts.get("Otros", 0)
    cobertura = (total - otros) / total * 100 if total else 0

    print("\n📊 Distribución por sección canónica (pool completo):")
    print(counts.to_string())
    print(
        f"\n🧭 Cobertura fuera de 'Otros': {cobertura:.1f}% "
        f"({total - otros:,} de {total:,}); 'Otros' = {otros:,}"
    )

    otros_crudos = (
        df[df["seccion_canonica"] == "Otros"]["seccion"]
        .value_counts()
        .head(25)
    )
    if not otros_crudos.empty:
        print("\n🔎 Top secciones crudas que caen en 'Otros' (para afinar reglas):")
        print(otros_crudos.to_string())


# --------------------------------------------------------------------------- #
# Muestreo balanceado
# --------------------------------------------------------------------------- #
def muestra_balanceada(
    df: pd.DataFrame, etiqueta: str, incluir_otros: bool = False
) -> pd.DataFrame:
    """Extrae una cuota igual por sección canónica hasta ~TARGET_TOTAL.

    La cuota se deriva del nº de clases (TARGET_TOTAL / n_clases). Por defecto
    'Otros' se excluye (cajón de sastre); con incluir_otros=True se trata como
    una clase más.
    """
    pool = df.copy() if incluir_otros else df[df["seccion_canonica"] != "Otros"].copy()
    counts = pool["seccion_canonica"].value_counts()
    per_class = ceil(TARGET_TOTAL / len(counts))

    print(
        f"\n📦 [{etiqueta}] clases disponibles ({len(counts)}), "
        f"cuota {per_class}/clase (objetivo ~{TARGET_TOTAL:,}):"
    )
    print(counts.to_string())

    cortas = counts[counts < MIN_PER_CLASS]
    if not cortas.empty:
        print(f"\n⚠️  [{etiqueta}] clases por debajo de {MIN_PER_CLASS} (se toman todas):")
        print(cortas.to_string())

    sample = (
        pool.groupby("seccion_canonica", group_keys=False)
        .apply(
            lambda g: g.sample(n=min(per_class, len(g)), random_state=RANDOM_STATE)
        )
        .reset_index(drop=True)
    )
    print(f"🎲 [{etiqueta}] muestra extraída: {len(sample):,} artículos")
    return sample


def compute_icor_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula icor_index, icor_v2_index e icor_v3_1_index sobre la muestra.

    Replica src/data_processing/data_processor.py::process_articles_flags,
    conservando el texto original en la salida.
    """
    corruption = get_normalized_terms("corruption")
    context = get_normalized_terms("context")
    v2_n1 = get_normalized_terms("corruption_v2_n1")
    v2_c1 = get_normalized_terms("corruption_v2_c1")
    v2_excl = get_normalized_terms("corruption_v2_exclusion")
    v3_n1 = get_normalized_terms("corruption_v3_n1_fuerte") + get_normalized_terms(
        "corruption_v3_n1_debil"
    )
    v3_c1 = get_normalized_terms("corruption_v3_c1_v31")
    v3_excl = get_normalized_terms("corruption_v3_exclusion")

    print(f"🔄 Calculando flags ICOR sobre {len(df):,} artículos...")
    texto_norm = df["texto"].apply(limpiar_y_normalizar_texto)

    df = df.copy()
    df["icor_index"] = texto_norm.apply(
        lambda txt: generate_mask_for_texto(txt, corruption, context)
    ).astype(int)
    df["icor_v2_index"] = texto_norm.apply(
        lambda txt: generate_mask_with_exclusion(
            txt, v2_n1, v2_c1, exclusion_terms=v2_excl
        )
    ).astype(int)
    df["icor_v3_1_index"] = texto_norm.apply(
        lambda txt: generate_mask_with_exclusion(
            txt, v3_n1, v3_c1, exclusion_terms=v3_excl
        )
    ).astype(int)
    print("✅ Flags ICOR calculadas")
    return df


def save(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    column_order = [
        "id_articulo", "id_pressreader", "titulo", "texto",
        "seccion", "seccion_canonica",
        "fecha", "id_periodico", "nombre_periodico", "anio",
        "incertidumbre", "economic_uncertainty", "political_uncertainty",
        "icor_index", "icor_v2_index", "icor_v3_1_index",
    ]
    df = df.sort_values(["seccion_canonica", "nombre_periodico", "id_articulo"])
    df[column_order].to_csv(output_path, index=False, encoding="utf-8")
    print(f"💾 Muestra guardada en: {output_path}")


def procesar_y_guardar(
    df: pd.DataFrame, etiqueta: str, output_path: Path, incluir_otros: bool = False
) -> None:
    sample = muestra_balanceada(df, etiqueta, incluir_otros=incluir_otros)
    sample = compute_icor_flags(sample)
    save(sample, output_path)
    print(f"\n📊 [{etiqueta}] distribución final por sección canónica:")
    print(sample["seccion_canonica"].value_counts().to_string())


def main():
    print(
        f"📰 Muestra balanceada por sección canónica "
        f"({START_YEAR}-{END_YEAR}, objetivo ~{TARGET_TOTAL:,}/muestra)"
    )
    print("=" * 65)

    samples_dir = project_root / "data" / "clustering_samples"

    df = load_articles()
    if df.empty:
        print("❌ No se obtuvieron artículos de la base de datos")
        return

    auditar_mapeo(df)

    # Salida A: incluye secciones no temáticas.
    print("\n" + "=" * 65)
    print("▶️  Muestra ALL (incluye no temáticas + 'Otros')")
    procesar_y_guardar(
        df, "all", samples_dir / "stratified_by_seccion_all.csv", incluir_otros=True
    )

    # Salida B: excluye secciones no temáticas.
    print("\n" + "=" * 65)
    print(f"▶️  Muestra THEMATIC (excluye {sorted(NO_TEMATICAS)})")
    df_tem = df[~df["seccion_canonica"].isin(NO_TEMATICAS)]
    procesar_y_guardar(
        df_tem, "thematic", samples_dir / "stratified_by_seccion_thematic.csv"
    )

    print("\n✅ Proceso completado — dos muestras en data/clustering_samples/")


if __name__ == "__main__":
    main()
