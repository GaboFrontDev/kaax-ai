# TODO - Hardening LangGraph + LangChain

Estado actual: funcional para pruebas locales/piloto, con mejoras pendientes antes de producción.

## Prioridad alta

- [ ] Filtrar/evitar salida de razonamiento interno (`<thinking>...</thinking>`) en respuestas al usuario.
- [ ] Extender idempotencia para `streamResponse=true` y evitar duplicados en reintentos.
- [ ] Definir estrategia de concurrencia por `thread_id` para UX (cola/retry en vez de solo `409`).

## Prioridad media

- [ ] Agregar pruebas E2E de resiliencia: caída de DB, caída de Bedrock, recuperación de checkpoints.
- [ ] Fortalecer observabilidad para SLO: latencia P95, lock contention, errores de tools, alertas.

## Seguridad

- [ ] Rotar y proteger secretos; evitar credenciales en archivos `.env` compartidos.
