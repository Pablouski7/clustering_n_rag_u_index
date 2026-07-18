# AGENTS.md

Este es el punto de entrada para los agentes autónomos de IA que trabajan en este repositorio.

Repositorio de tesis de maestría de la USFQ enfocado en implementar un pipeline de embeddings, clustering y RAG (Generación Aumentada por Recuperación) sobre una muestra de artículos periodísticos ecuatorianos. Utiliza Milvus como base vectorial y modelos vLLM de la universidad (como deepseek-v4-flash y BAAI/bge-m3) a través de una interfaz compatible con OpenAI.

**Tecnologías clave:** Python, Milvus, vLLM (OpenAI SDK), Docker, Jupyter Notebooks.

<!-- BEGIN:MANUAL -->
## Índice de Documentación

| Documento | Descripción |
| --- | --- |
| [CLAUDE.md](file:///home/pablo-herrera/Documents/02_Estudio/USFQ%20Maestr%C3%ADa/clustering_n_rag_u_index/CLAUDE.md) | Comandos del proyecto, estado de infraestructura y notas de integración vLLM. |
| [architecture.md](file:///home/pablo-herrera/Documents/02_Estudio/USFQ%20Maestr%C3%ADa/clustering_n_rag_u_index/.agents/docs/architecture.md) | Guía detallada sobre la arquitectura del pipeline de embeddings, clustering y RAG. |
| [conventions.md](file:///home/pablo-herrera/Documents/02_Estudio/USFQ%20Maestr%C3%ADa/clustering_n_rag_u_index/.agents/docs/conventions.md) | Convenciones de código, nombres de variables y estándares lingüísticos. |
| [setup.md](file:///home/pablo-herrera/Documents/02_Estudio/USFQ%20Maestr%C3%ADa/clustering_n_rag_u_index/.agents/docs/setup.md) | Configuración del entorno de desarrollo, variables de entorno y comandos necesarios. |
| [adding-features.md](file:///home/pablo-herrera/Documents/02_Estudio/USFQ%20Maestr%C3%ADa/clustering_n_rag_u_index/.agents/docs/adding-features.md) | Guía paso a paso para añadir nuevos embeddings, algoritmos de clustering o flujos de RAG. |

## Reglas de Comportamiento Clave

1. **Idioma de Desarrollo:** Mantener el español para nombres de variables, docstrings, mensajes de log, comentarios en código y commits, siguiendo las convenciones del repositorio.
2. **Sincronización de Dependencias:** Al agregar una dependencia, actualizar simultáneamente `pyproject.toml` (uv/poetry), `requirements.txt` (pip) y `environment.yml` (conda/micromamba) para mantener los entornos sincronizados.
3. **Validación de Modelos vLLM:** Consultar siempre el endpoint `/v1/models` antes de realizar llamadas para revalidar que los nombres de los modelos (ej. `BAAI/bge-m3`, `deepseek-ai/DeepSeek-V4-Flash`) no hayan cambiado en el servidor vLLM.
4. **Infraestructura Local:** Asegurarse de que el contenedor de Docker para Milvus esté arriba antes de ejecutar pruebas en bases vectoriales.
<!-- END:MANUAL -->
