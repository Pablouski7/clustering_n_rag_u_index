#!/usr/bin/env python3
"""
Script para extraer una muestra estratificada de artículos por periódico y año.

Estratos: año (2019-2026) × periódico (24 combinaciones). Cada estrato aporta
min(ceil(TARGET_TOTAL / n_estratos), disponibles) artículos, con un mínimo
obligatorio de MIN_PER_STRATUM por estrato (aborta si no se cumple).

La muestra incluye todas las columnas de la BD asociadas a cada artículo
(id, título, texto, sección, fecha, periódico y las flags almacenadas) más
las flags ICOR calculadas al vuelo (icor_index, icor_v2_index, icor_v3_1_index),
replicando la lógica de src/data_processing/data_processor.py.
"""

import sys
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
TARGET_TOTAL = 10_000
MIN_PER_STRATUM = 140
RANDOM_STATE = 42

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


def load_articles() -> pd.DataFrame:
    """Carga todos los artículos del rango con todas sus columnas de la BD."""
    print(f"🔄 Cargando artículos {START_YEAR}-{END_YEAR} desde la BD...")
    engine = create_engine(get_sqlalchemy_url())
    df = pd.read_sql(QUERY, engine)
    engine.dispose()

    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["anio"] = df["fecha"].dt.year
    print(f"✅ {len(df):,} artículos cargados")
    return df


def stratified_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Extrae la muestra estratificada por (año, periódico)."""
    counts = df.groupby(["anio", "nombre_periodico"]).size()
    n_strata = len(counts)
    quota = ceil(TARGET_TOTAL / n_strata)

    print(f"\n📊 Disponibles por estrato ({n_strata} estratos, cuota {quota}):")
    print(counts.to_string())

    short = counts[counts < MIN_PER_STRATUM]
    if not short.empty:
        print(f"\n❌ Estratos con menos de {MIN_PER_STRATUM} artículos:")
        print(short.to_string())
        sys.exit(1)

    for (anio, periodico), n in counts.items():
        if n < quota:
            print(
                f"⚠️  Estrato ({anio}, {periodico}) tiene {n} < cuota {quota}: "
                f"se toman todos"
            )

    sample = (
        df.groupby(["anio", "nombre_periodico"], group_keys=False)
        .apply(lambda g: g.sample(n=min(quota, len(g)), random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )
    print(f"\n🎲 Muestra extraída: {len(sample):,} artículos")
    return sample


def compute_icor_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula icor_index, icor_v2_index e icor_v3_1_index sobre la muestra.

    Replica src/data_processing/data_processor.py::process_articles_flags,
    pero conservando el texto original en la salida.
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

    print(f"\n🔄 Calculando flags ICOR sobre {len(df):,} artículos...")
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
        "id_articulo", "id_pressreader", "titulo", "texto", "seccion",
        "fecha", "id_periodico", "nombre_periodico", "anio",
        "incertidumbre", "economic_uncertainty", "political_uncertainty",
        "icor_index", "icor_v2_index", "icor_v3_1_index",
    ]
    df = df.sort_values(["anio", "nombre_periodico", "id_articulo"])
    df[column_order].to_csv(output_path, index=False, encoding="utf-8")
    print(f"💾 Muestra guardada en: {output_path}")


def main():
    print(
        f"📰 Muestra estratificada por periódico y año "
        f"({START_YEAR}-{END_YEAR}, objetivo ~{TARGET_TOTAL:,})"
    )
    print("=" * 65)

    output_path = project_root / "data" / "clustering_samples" / "stratified_sample_2019_2026.csv"

    df = load_articles()
    if df.empty:
        print("❌ No se obtuvieron artículos de la base de datos")
        return

    sample = stratified_sample(df)
    sample = compute_icor_flags(sample)
    save(sample, output_path)

    print(f"\n✅ Proceso completado — {len(sample):,} artículos en la muestra")
    print("\n📊 Distribución final por estrato:")
    print(sample.groupby(["anio", "nombre_periodico"]).size().to_string())
    print("\n📊 Tasa de positivos por flag:")
    flags = [
        "incertidumbre", "economic_uncertainty", "political_uncertainty",
        "icor_index", "icor_v2_index", "icor_v3_1_index",
    ]
    print(sample[flags].mean().round(4).to_string())


if __name__ == "__main__":
    main()
