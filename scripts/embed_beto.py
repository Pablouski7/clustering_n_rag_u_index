"""Calcula embeddings BETO de la submuestra del EDA y los cachea en `data/embeddings/`.

Reproduce exactamente la submuestra estratificada de ~1,500 artículos usada por
`notebooks/eda.ipynb` y `notebooks/hc_clustering.ipynb`, y verifica los `id_articulo`
contra `muestra_ids.npy` para que los vectores sean comparables con MiniLM y BGE-M3.

A diferencia de los otros métodos, aquí **no se trunca el texto a 2,000 caracteres**: el
punto de BETO en este proyecto es cubrir el artículo completo vía chunking + pooling
(ver `src/embeddings/beto.py`).

Uso:
    micromamba activate ai_env && python scripts/embed_beto.py
"""

import re
import sys
from math import ceil
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.embeddings.beto import embed_articulos  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
EMB_DIR = RAIZ / "data" / "embeddings"
RANDOM_STATE = 42
N_MUESTRA = 1500


def norm_text(text: str) -> str:
    """Normaliza un texto: minúsculas y colapso de espacios en blanco."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def reproducir_submuestra() -> pd.DataFrame:
    """Replica la submuestra estratificada (año × periódico) del EDA."""
    data = pd.read_csv(RAIZ / "data" / "raw" / "stratified_grid_2019_2026.csv")
    data["anio"] = pd.to_datetime(data["anio"], format="%Y").dt.year
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
    print(f"Submuestra: {len(textos)} artículos (texto completo, sin truncar)")

    emb = embed_articulos(textos)
    np.save(EMB_DIR / "beto.npy", emb)
    np.save(EMB_DIR / "beto_ids.npy", ids_muestra)
    print(f"Guardado: {EMB_DIR / 'beto.npy'} — {emb.shape}")


if __name__ == "__main__":
    main()
