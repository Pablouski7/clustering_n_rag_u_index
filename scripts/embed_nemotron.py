"""Calcula embeddings Nemotron VL de la submuestra del EDA y los cachea en `data/embeddings/`.

Reproduce exactamente la submuestra estratificada de ~1,500 artículos usada por
`notebooks/eda.ipynb`, `notebooks/hc_clustering.ipynb` y `notebooks/hc_autoencoder.ipynb`,
y verifica los `id_articulo` contra `muestra_ids.npy` para que los vectores sean comparables
con las demás técnicas.

Igual que BETO, **no se trunca el texto a 2,000 caracteres**: la ventana del modelo es de
10,240 tokens y el artículo más largo del corpus ronda los 6,900, así que cada artículo entra
completo en un solo forward pass (ver `src/embeddings/nemotron.py`).

**Este script tarda horas en CPU** (~1.7 B parámetros frente a los 110 M de BETO). Por eso
guarda el avance por bloques en `nemotron_parcial.npy`: si se interrumpe, la siguiente
ejecución retoma donde quedó.

Uso:
    micromamba activate ai_env && python scripts/embed_nemotron.py
"""

import re
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.nemotron import cargar_modelo, embed_articulos  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
EMB_DIR = RAIZ / "data" / "embeddings"
RANDOM_STATE = 42
N_MUESTRA = 1500
# Artículos por bloque: cada bloque completado se persiste, acotando lo que se pierde
# si el proceso muere a mitad de camino.
TAM_BLOQUE = 50


def norm_text(text: str) -> str:
    """Normaliza un texto: minúsculas y colapso de espacios en blanco."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def reproducir_submuestra() -> pd.DataFrame:
    """Replica la submuestra estratificada (año × periódico) del EDA."""
    data = pd.read_csv(RAIZ / "data" / "raw" / "stratified_sample_2019_2026.csv")
    data["anio"] = pd.to_datetime(data["fecha"]).dt.year
    estratos = data.groupby(["nombre_periodico", "anio"])
    cuota = ceil(N_MUESTRA / estratos.ngroups)
    return (
        pd.concat([g.sample(min(len(g), cuota), random_state=RANDOM_STATE) for _, g in estratos])
        .sort_values(["anio", "nombre_periodico", "id_articulo"])
        .reset_index(drop=True)
    )


def main() -> None:
    muestra = reproducir_submuestra()
    ids_muestra = muestra["id_articulo"].to_numpy()

    ruta_ids_eda = EMB_DIR / "muestra_ids.npy"
    if ruta_ids_eda.exists() and not np.array_equal(np.load(ruta_ids_eda), ids_muestra):
        sys.exit("La submuestra no coincide con la del EDA: recalcular embeddings en eda.ipynb")

    textos = (
        muestra["titulo"].fillna("").map(norm_text) + ". " + muestra["texto"].fillna("").map(norm_text)
    ).tolist()

    ruta_parcial = EMB_DIR / "nemotron_parcial.npy"
    parcial = np.load(ruta_parcial) if ruta_parcial.exists() else np.zeros((0, 2048), dtype=np.float32)
    if len(parcial) >= len(textos):
        sys.exit(f"Ya hay {len(parcial)} vectores en {ruta_parcial}: borrarlo para recalcular")

    print(f"Submuestra: {len(textos)} artículos (texto completo, sin truncar)")
    if len(parcial):
        print(f"Retomando desde el artículo {len(parcial)} ({ruta_parcial})")

    modelo = cargar_modelo()
    for inicio in range(len(parcial), len(textos), TAM_BLOQUE):
        bloque = textos[inicio : inicio + TAM_BLOQUE]
        print(f"Bloque {inicio}–{inicio + len(bloque)} de {len(textos)}")
        # normalizar=False: se normaliza una sola vez sobre la matriz completa, al final.
        parcial = np.vstack([parcial, embed_articulos(bloque, modelo=modelo, normalizar=False)])
        np.save(ruta_parcial, parcial)

    emb = parcial / np.linalg.norm(parcial, axis=1, keepdims=True).clip(min=1e-9)
    np.save(EMB_DIR / "nemotron.npy", emb)
    np.save(EMB_DIR / "nemotron_ids.npy", ids_muestra)
    ruta_parcial.unlink()
    print(f"Guardado: {EMB_DIR / 'nemotron.npy'} — {emb.shape}")


if __name__ == "__main__":
    main()
