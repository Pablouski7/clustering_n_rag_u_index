# Configuración del Entorno de Desarrollo (Setup)

Este documento detalla los pasos para configurar el entorno local.

## Requisitos Previos
- Docker y Docker Compose (para Milvus y Attu).
- Micromamba, Conda o Python 3.10+ con venv.

## Instalación
1. **Entorno Python (Conda/Micromamba):**
   ```bash
   micromamba activate base && pip install -r requirements.txt
   ```
   En esta máquina en particular, usar el entorno dedicado `ai_env` en vez de `base` (ver `CLAUDE.md`).
2. **Variables de Entorno:**
   Copiar `.env.example` a `.env` y configurar las credenciales y URLs para el servidor vLLM.
   ```bash
   cp .env.example .env
   ```

## Ejecución de Servicios
```bash
docker compose up -d              # levanta etcd + minio + milvus
docker compose --profile ui up -d # además Attu (UI en http://localhost:8000)
docker compose ps                 # verificar que 'milvus' esté (healthy)
docker compose down               # parar (agregar -v para borrar datos)
```

## Pruebas de conectividad
```bash
python scripts/check_milvus.py    # Milvus (gRPC localhost:19530)
python scripts/check_vllm.py      # vLLM: chat + embeddings (requiere VPN GlobalProtect)
```
