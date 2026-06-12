# Figuras del artículo

Generar con: `python model/figures.py` desde la raíz del repositorio.
Salida: archivos PNG en este directorio.

| Archivo | Contenido | Cuándo generarla |
|---------|-----------|------------------|
| `fig1_historical_overview.png` | Panel 3×4 canónico (x=goles ganador, y=goles perdedor): datos históricos WC 2014/2018/2022 (proporciones) + estimado modelo antes del torneo (δ=0). Las 4 celdas inferiores del bloque derecho son placeholders: 2 se llenan al fin de grupos (~jul 3), 2 al fin del torneo (~jul 19). | Ya |
| `fig2_gamma_effect.png` | Comparación de distribución con γ=1 (sin decaimiento) vs γ≈0.84 (estimado) | Ya |
| `fig3_kalshi_reweight.png` | Ejemplo de reajuste de distribución antes/después de aplicar precios de Kalshi, para un partido concreto del WC 2026 | Ya (usar cualquier partido con datos de Kalshi en caché) |
| `fig4_delta_evolution.png` | Estimación de δ por jornada conforme llegan resultados del WC 2026 | Actualizar al final de cada jornada |
| `fig5_points_results.png` | Puntos reales vs esperados por partido, fase de grupos | Al concluir la fase de grupos (~2026-07-03) |
