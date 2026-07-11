#!/usr/bin/env python3
"""Prueba de humo de Milvus: conecta, crea una colección temporal, inserta,
busca y la elimina. Sirve para validar que el docker-compose está arriba.

Uso:
    micromamba activate base
    python scripts/check_milvus.py
"""

import os
import random

from dotenv import load_dotenv
from pymilvus import MilvusClient

load_dotenv()

URI = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION = "_smoke_test"
DIM = 8


def main() -> None:
    print(f"🔌 Conectando a Milvus en {URI} ...")
    client = MilvusClient(uri=URI)

    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    client.create_collection(collection_name=COLLECTION, dimension=DIM)
    print(f"✅ Colección temporal '{COLLECTION}' creada (dim={DIM})")

    data = [
        {"id": i, "vector": [random.random() for _ in range(DIM)]}
        for i in range(5)
    ]
    client.insert(collection_name=COLLECTION, data=data)
    print(f"✅ Insertados {len(data)} vectores")

    results = client.search(
        collection_name=COLLECTION,
        data=[data[0]["vector"]],
        limit=3,
        output_fields=["id"],
    )
    print(f"🔎 Búsqueda OK, vecinos más cercanos: {[hit['id'] for hit in results[0]]}")

    client.drop_collection(COLLECTION)
    print("🧹 Colección temporal eliminada")
    print("\n🎉 Milvus responde correctamente.")


if __name__ == "__main__":
    main()
