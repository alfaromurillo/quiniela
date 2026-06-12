# TODO — Artículo quiniela WC 2026

## Inmediato (escribir antes de ~2026-07-03, fin de fase de grupos)
- [ ] Revisar y completar sección 1: Introducción
- [ ] Revisar y completar sección 2: Mercados de predicción
- [ ] Revisar y completar sección 3: Modelo histórico (3.1–3.4)
- [ ] Revisar y completar sección 4: Integración de Kalshi
- [ ] Revisar y completar sección 5: Optimización de puntos
- [ ] Revisar y completar sección 6: Implementación y Claude Code
- [ ] Generar figuras 1–4 en PNG: `python model/figures.py`
- [ ] Verificar refs.bib: DOIs correctos, títulos sin errores
- [ ] Subir preprint a arXiv (en español, con abstract en inglés)

## Al concluir la fase de grupos (~2026-07-03)
- [ ] Rellenar sección 7.1 con resultados reales de la fase de grupos
- [ ] Calcular métricas: calibración de Kalshi, puntos reales vs esperados
- [ ] Calcular evolución de δ por jornada y generar figura 4 actualizada
- [ ] Generar figura 5: puntos reales vs esperados por partido
- [ ] Completar sección 8 (Discusión) con datos reales
- [ ] Actualizar preprint en arXiv con versión completa de grupos
- [ ] Enviar a revisión a la Revista de Matemática (UCR/CIMPA)
  - Preparar carta de presentación (cover_letter.tex)
  - Incluir nota sobre preprint en arXiv en la carta

## Durante la fase eliminatoria (~2026-07-04 al ~2026-07-19)
- [ ] Rellenar sección 7.2 por ronda conforme avancen los partidos:
  - [ ] Ronda de 32 (~jul 4-7)
  - [ ] Octavos de final (~jul 9-12)
  - [ ] Cuartos de final (~jul 14-15)
  - [ ] Semifinales (~jul 17-18)
  - [ ] Final (~jul 19)

## Para responder comentarios de revisión
- [ ] Incorporar resultados completos del torneo en sección 7.2
- [ ] Responder comentarios de los revisores
- [ ] Ajustar formato al template oficial de la revista (LaTeX)
- [ ] Confirmar que todas las figuras estén en PNG a resolución adecuada

## Plazos clave
- **~2026-07-03**: fin fase de grupos → envío a arXiv + revista
- **~2026-07-19**: fin del Mundial → sección 7.2 completa
- **2026-2027**: revisión de la revista (6-10 meses estimados)
- **Meta**: número enero-junio 2027

## Datos que hay que reportar en sección 7 (calcular al final de cada fase)
- Puntos totales obtenidos / puntos máximos posibles
- Puntos por partido (tabla)
- Puntos esperados promedio del modelo vs puntos reales promedio
- Score de Brier de las probabilidades de Kalshi (calibración)
- Valor estimado de δ al final de la fase de grupos
- Valor estimado de α̂ al final de la fase de grupos (vs α₀ = 1.10 de Whelan 2024)
- Comparación vs líneas base:
  - Línea base 1: probabilidades uniformes (1/3, 1/3, 1/3)
  - Línea base 2: modelo histórico sin Kalshi
