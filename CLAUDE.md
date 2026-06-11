# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

World Cup 2026 score predictor for a quiniela (prediction pool). The model combines Kalshi prediction-market probabilities with WC 2022 historical scoreline distributions to find the scoreline that **maximises expected quiniela points** for each match.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full prediction pipeline (fetches Kalshi data, writes site/data/predictions.json)
python model/predict.py

# Quick model sanity check (no network, pure math)
python -c "
from model.historical import scoreline_probs
from model.optimizer import best_prediction
sp = scoreline_probs(0.50, 0.25, 0.25, 'group')
print(best_prediction(sp, 'group'))
"
```

## Scoring rules (quiniela)

**Group stage** (priority order — first matching rule wins):
| Result | Points |
|--------|--------|
| Exact score | 5 |
| Correct winner/draw **AND** correct goals of one team | 3 |
| Correct winner/draw only | 2 |
| Correct goals of one team, wrong winner | 1 |

**Knockout** (goals in 90+30 min; penalty shootout → predict draw):
| Result | Points |
|--------|--------|
| Exact score | 3 |
| Correct winner/draw (wrong goals) | 1 |

## Architecture

```
data/
  wc2022.json          # WC 2022 results from openfootball (source of truth for historical model)
  schedule.json        # WC 2026 full schedule (104 matches, generated from openfootball 2026 data)
  wc2026_raw.json      # Raw openfootball 2026 data (keep for reference)
  kalshi_cache.json    # Cached Kalshi API responses (TTL 1h, keyed by game event ticker)

model/
  historical.py        # Builds P(score | result, phase) from WC 2022 + Laplace smoothing
  kalshi.py            # Fetches win/draw/loss + total-goals + spread markets from Kalshi API
  optimizer.py         # E[pts] calculator and argmax over 0–5 goal grid
  predict.py           # Pipeline: schedule → Kalshi → scoreline_probs → best_prediction → JSON

site/
  data/predictions.json  # Output consumed by the frontend
  index.html             # Match predictions by date (TODO: build)
  modelo.html            # Model methodology page (TODO: build)
  assets/style.css       # (TODO)
  assets/main.js         # (TODO)

.github/workflows/update.yml  # Daily cron: runs predict.py, commits predictions.json (TODO)
```

## Kalshi API details

No authentication required for market data. Three market types per match:

| Type | Ticker pattern | Info |
|------|---------------|------|
| Game outcome | `KXWCGAME-{date}{HOME}{AWAY}` | `-HOME`, `-AWAY`, `-TIE` sub-markets |
| Total goals | `KXWCTOTAL-{date}{HOME}{AWAY}` | `-1` = over 0.5, `-2` = over 1.5, …, `-6` = over 5.5 |
| Spread | `KXWCSPREAD-{date}{HOME}{AWAY}` | `-HOME2/3/4`, `-AWAY2/3` = wins by over N.5 |

**Date code rule**: Kalshi uses Eastern Time (UTC-4) date in the ticker, derived from `time_utc` in schedule.json. Use `_date_code(time_utc)` in `kalshi.py`.

**Team codes** (Kalshi abbreviations, differ from FIFA/ISO in several cases):
- Haiti → `HTI`, Iran → `IRI`, Algeria → `DZA`, Turkey → `TUR`
- Saudi Arabia → `KSA`, Scotland → `SCO`, South Korea → `KOR`
- Full mapping in `model/kalshi.py` → `TEAM_CODES` dict

**Live/settled markets**: When a game is in progress, Kalshi prices become extreme (one outcome near 1.0). `_midprice()` rejects spreads > 0.30. For total-goals markets, `last_price_dollars` is used as fallback when bid/ask spread is too wide. Pre-game cache entry is preserved so predictions don't change mid-game.

## Known issues / TODO

1. **Date code bug**: `_date_code()` currently takes `date_str` (local date) but should take `time_utc` and convert to ET. Fix: change `predict.py` to pass `match["time_utc"]` and update `_date_code` signature.
2. **Missing team codes**: Some teams fall back to historical rates. Find correct codes by testing `KXWCGAME-{date}{CODE1}{CODE2}` via API.
3. **Frontend**: `site/index.html`, `site/modelo.html`, `site/assets/` not yet built.
4. **GitHub Actions**: `.github/workflows/update.yml` not yet created.
5. **Knockout TBD**: Knockout matches skip until bracket resolves. Once teams are known, predictions generate automatically on next run.

## Data pipeline flow

```
schedule.json  ──→  kalshi.py (fetch_match_probs)
                         │  P(home_win), P(draw), P(away_win)
                         │  total_goals_probs {0..6}
                         ↓
wc2022.json  ──→  historical.py (scoreline_probs)
                         │  P(home=a, away=b) for all a,b ∈ 0..5
                         │  weighted by Kalshi result probs
                         │  reweighted by Kalshi total-goals probs
                         ↓
                   optimizer.py (best_prediction)
                         │  argmax E[quiniela_pts | predict (a,b)]
                         ↓
              site/data/predictions.json
```

## Plan

Full implementation plan at: `~/.claude/plans/parallel-wiggling-oasis.md`

Remaining work:
- Fix date code and missing team codes in `model/kalshi.py`
- Build `site/index.html` + `site/modelo.html` + `site/assets/`
- Write `.github/workflows/update.yml`
- Create GitHub repo + enable Pages from `main`/`site`
