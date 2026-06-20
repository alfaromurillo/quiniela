# Quiniela Mundial 2026

Score predictions for the 2026 FIFA World Cup, optimised to
maximise expected points in a quiniela (prediction pool).

**Live site:** https://alfaromurillo.github.io/quiniela/

## How it works

The model combines two sources of information:

1. **Kalshi prediction-market probabilities** — real-money market odds
   for win/draw/loss, total goals, and goal spread for each match.
2. **Historical World Cup scoreline distributions** — weighted data
   from WC 2014, 2018, and 2022 (γ-decay weighting, γ ≈ 0.84), with
   a Poisson product prior for smoothing.

For each match the model finds the scoreline that **maximises expected
quiniela points** (not just the most probable scoreline).

Once WC 2026 results accumulate, a tournament-specific weight δ is
estimated by cross-validation and blended in automatically.

## Quiniela scoring

**Group stage** (first matching rule wins):

| Result | Points |
|--------|--------|
| Exact score | 5 |
| Correct winner/draw + correct goals of one team | 3 |
| Correct winner/draw only | 2 |
| Correct goals of one team, wrong winner | 1 |

**Knockout** (90 + 30 min; penalty shootout → predict draw):

| Result | Points |
|--------|--------|
| Exact score | 3 |
| Correct winner/draw (wrong goals) | 1 |

## Repository layout

```
model/          # Python model code
  historical.py # Weighted scoreline distributions
  kalshi.py     # Kalshi market data fetcher / cache
  optimizer.py  # Expected-points maximiser
  predict.py    # Full prediction pipeline
  results.py    # Fetch completed match scores from ESPN
  learn.py      # Cross-validate γ and δ parameters
  sanity.py     # Validate predictions.json before commit
  figures.py    # Generate article figures

site/           # GitHub Pages static site
  index.html    # Predictions by matchday
  modelo.html   # Methodology (MathJax)
  data/
    predictions.json        # Current predictions (auto-updated)
    locked_predictions.json # Frozen 1.5 h before kickoff
    results.json            # Final scores for completed matches

data/           # Historical WC data and schedule
articulo/       # Scientific article (LaTeX)
```

## Running locally

```bash
pip install -r requirements.txt

# Update predictions (fetches Kalshi, writes site/data/predictions.json)
python model/predict.py

# Fetch completed match results from ESPN
python model/results.py

# Validate predictions before committing
python model/sanity.py

# Preview the site
cd site && python3 -m http.server 8765 --bind 127.0.0.1
# open http://127.0.0.1:8765/
```

Predictions are refreshed automatically every hour via GitHub Actions.
