#!/usr/bin/env python3
"""
Muestra grande estratificada por (año × periódico) con secciones aplanadas.

Complementa scripts/sample_balanced_by_seccion.py (que balancea SOLO por sección
y sacrifica año/periódico). Aquí el objetivo es el inverso suavizado:

  - Estratificar por celda (año × periódico) con un TOPE por celda (CELL_CAP):
    las celdas grandes se recortan, las chicas aportan todo. Esto empareja
    año/periódico tanto como el corpus permite (Diario Expreso 2019-2020 es
    intrínsecamente escaso: 142-235 artículos, no hay forma de balancearlo).
  - DENTRO de cada celda, muestreo round-robin entre secciones canónicas para
    aplanar la cola de secciones lo máximo que la celda permita.

'Otros' se excluye (cajón de sastre, ruido como label). Reutiliza el mapeo
canónico y las utilidades de sample_balanced_by_seccion.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from scripts.sampling_for_clustering.sample_balanced_by_seccion import (
    load_articles,
    compute_icor_flags,
    save,
)

CELL_CAP = 1200  # tope de artículos por celda (año × periódico)
OTROS_SAMPLE = 2000  # ejemplos de 'Otros' a incluir para exploración (0 = ninguno)
RANDOM_STATE = 42


def draw_balanceado_por_seccion(cell: pd.DataFrame, n: int) -> pd.DataFrame:
    """Toma n filas de la celda repartiéndolas lo más parejo posible entre
    secciones canónicas (round-robin).

    Baraja la celda, asigna a cada fila su posición dentro de su sección
    (cumcount) y toma las n de menor posición: todas las de posición 0 (una por
    sección) primero, luego las de posición 1, etc. Las secciones grandes solo
    aportan en posiciones altas, así que la distribución sale casi plana hasta
    donde alcanza la disponibilidad.
    """
    if len(cell) <= n:
        return cell
    barajada = cell.sample(frac=1, random_state=RANDOM_STATE)
    rank = barajada.groupby("seccion_canonica").cumcount()
    barajada = barajada.assign(_rank=rank).sort_values("_rank", kind="stable")
    return barajada.head(n).drop(columns="_rank")


def muestra_grid(df: pd.DataFrame) -> pd.DataFrame:
    pool = df[df["seccion_canonica"] != "Otros"].copy()

    print(f"\n📦 Grid (año × periódico), tope {CELL_CAP}/celda:")
    piv = pool.pivot_table(
        index="anio", columns="nombre_periodico",
        values="id_articulo", aggfunc="count", fill_value=0,
    )
    print(piv.to_string())

    partes = []
    for (anio, periodico), cell in pool.groupby(["anio", "nombre_periodico"]):
        n = min(CELL_CAP, len(cell))
        partes.append(draw_balanceado_por_seccion(cell, n))
    sample = pd.concat(partes).reset_index(drop=True)

    if OTROS_SAMPLE:
        otros_pool = df[df["seccion_canonica"] == "Otros"]
        n_otros = min(OTROS_SAMPLE, len(otros_pool))
        otros = otros_pool.sample(n=n_otros, random_state=RANDOM_STATE)
        sample = pd.concat([sample, otros]).reset_index(drop=True)
        print(f"➕ Añadidos {n_otros:,} ejemplos de 'Otros' para exploración")

    print(f"\n🎲 Muestra extraída: {len(sample):,} artículos")
    return sample


def main():
    print(f"📰 Muestra grid año×periódico + secciones aplanadas (tope {CELL_CAP}/celda)")
    print("=" * 65)

    output_path = (
        project_root / "data" / "clustering_samples" / "stratified_grid_2019_2026.csv"
    )

    df = load_articles()
    if df.empty:
        print("❌ No se obtuvieron artículos de la base de datos")
        return

    sample = muestra_grid(df)
    sample = compute_icor_flags(sample)
    save(sample, output_path)

    print("\n📊 Distribución final por periódico:")
    print(sample["nombre_periodico"].value_counts().to_string())
    print("\n📊 Distribución final por año:")
    print(sample["anio"].value_counts().sort_index().to_string())
    print("\n📊 Distribución final por sección canónica:")
    print(sample["seccion_canonica"].value_counts().to_string())
    print(f"\n✅ Proceso completado — {len(sample):,} artículos en {output_path.name}")


if __name__ == "__main__":
    main()
