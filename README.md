# Quiniela Mundial 2026

Predicciones de marcadores para el Mundial de Fútbol 2026,
optimizadas para maximizar los puntos esperados en una quiniela.

**Sitio web:** https://alfaromurillo.github.io/quiniela/

## Cómo funciona

El modelo combina dos fuentes de información:

1. **Probabilidades del mercado de predicciones Kalshi** — apuestas
   con dinero real sobre victoria/empate/derrota, total de goles y
   diferencia de goles por partido.
2. **Distribuciones históricas de marcadores** — datos ponderados de
   los mundiales 2014, 2018 y 2022 (ponderación γ ≈ 0.84), con una
   prior de producto Poisson para suavizado.

Para cada partido el modelo encuentra el marcador que **maximiza los
puntos esperados en la quiniela** (no simplemente el marcador más
probable).

A medida que se acumulan resultados del Mundial 2026, se estima un
peso δ específico del torneo mediante validación cruzada y se
incorpora automáticamente.

## Puntuación de la quiniela

**Fase de grupos** (se aplica la primera regla que coincida):

| Resultado | Puntos |
|-----------|--------|
| Marcador exacto | 5 |
| Ganador/empate correcto + goles correctos de un equipo | 3 |
| Solo ganador/empate correcto | 2 |
| Goles correctos de un equipo, ganador incorrecto | 1 |

**Eliminatorias** (90 + 30 min; penaltis → predecir empate):

| Resultado | Puntos |
|-----------|--------|
| Marcador exacto | 3 |
| Ganador/empate correcto (goles incorrectos) | 1 |

## Estructura del repositorio

```
model/          # Código del modelo en Python
  historical.py # Distribuciones de marcadores ponderadas
  kalshi.py     # Obtención y caché de datos de Kalshi
  optimizer.py  # Maximizador de puntos esperados
  predict.py    # Pipeline completo de predicción
  results.py    # Obtiene resultados de partidos desde ESPN
  learn.py      # Estimación de γ y δ por validación cruzada
  sanity.py     # Valida predictions.json antes de hacer commit
  figures.py    # Genera figuras del artículo

site/           # Sitio estático en GitHub Pages
  index.html    # Predicciones por jornada
  modelo.html   # Metodología (MathJax)
  data/
    predictions.json        # Predicciones actuales (actualización automática)
    locked_predictions.json # Congeladas 1.5 h antes del partido
    results.json            # Marcadores finales de partidos jugados

data/           # Datos históricos y calendario del mundial
articulo/       # Artículo científico (LaTeX)
```

## Ejecución local

```bash
pip install -r requirements.txt

# Actualizar predicciones (consulta Kalshi, escribe site/data/predictions.json)
python model/predict.py

# Obtener resultados de partidos jugados desde ESPN
python model/results.py

# Validar predicciones antes de hacer commit
python model/sanity.py

# Vista previa del sitio
cd site && python3 -m http.server 8765 --bind 127.0.0.1
# abrir http://127.0.0.1:8765/
```

Las predicciones se actualizan automáticamente cada hora mediante
GitHub Actions.
