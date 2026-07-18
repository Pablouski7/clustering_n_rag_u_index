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
2. **Variables de Entorno:**
   Copiar `.env.example` a `.env` y configurar las credenciales y URLs para el servidor vLLM.
   ```bash
   cp .env.example .env
   ```

## Ejecución de Servicios
```bash
docker compose up -d
```
o con el perfil de UI (Attu):
```bash
docker compose --profile ui up -d
```
