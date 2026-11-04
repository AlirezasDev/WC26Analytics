# WC26 Analytics - Project Summary

## Overview
WC26 Analytics is an advanced predictive machine learning engine designed to forecast international and club football match outcomes (Home Win, Draw, Away Win) and compute bookmaker-grade European Decimal and American Moneyline Odds.

The engine integrates full historical match datasets, dynamic Elo ratings seeded with FIFA priors, Dixon-Coles bivariate Poisson goal models, and comprehensive squad/lineup profiling with market valuation dynamics.

---

## What the System Does
1. **Multi-Class Outcome Forecasting**: Computes calibrated probabilities for 1X2 outcomes (Win / Draw / Loss).
2. **Professional Betting Odds Generation**:
   - European Decimal Odds ($1 / P$)
   - American Moneyline Odds ($+ / -$)
3. **Dynamic Elo Rating Engine**:
   - Chronological Elo tracking across all 88k+ matches with tournament-specific $K$-factors ($K=60$ for World Cup, $K=50$ for national competitions).
   - Pre-seeded with FIFA World Rankings.
4. **Dixon-Coles Bivariate Poisson Expected Goals ($xG$)**:
   - Models low-scoring adjustments (0-0, 1-0, 0-1, 1-1) and expected goals for home and away sides.
5. **Detailed Squad & Lineup Profiling**:
   - Distinguishes Starting XI from Substitutes.
   - Computes positional market values (Forwards, Midfielders, Defenders, Goalkeepers) and top-3 star-player valuations.
   - Tracks 1.89 million player appearances with rolling 5-match form metrics.
6. **Position-Aware Monte Carlo Simulations**:
   - Simulates 500+ match instances under player rotation and form stochasticity.

---

## Dataset Scale & Temporal Split
- **Total Historical Matches**: 88,958 games (covering both club and national team competitions).
- **Training Set**: 88,872 matches strictly prior to the 2026 World Cup (`date < 2026-06-11`).
- **Evaluation Set**: 72 unseen Group Stage matches of the 2026 World Cup (`2026-06-11` to `2026-06-28`, `competition_id == 'FIWC'`).
- **Feature Space**: 192 features (187 numeric + 5 categorical), including differential (`diff_*`) and ratio (`ratio_*`) features.

---

## Core Algorithms & Ensembling
- **Soft-Voting Ensemble**:
  - **LightGBM Classifier**: 300 estimators, multiclass log-loss objective.
  - **CatBoost Classifier**: 300 iterations, MultiClass loss function.
  - **Random Forest Classifier**: 250 trees, max depth 15, balanced subsampling.
- **Sample Weighting**: Applies a 3.5x multiplier to national team tournaments and exponential time decay (half-life of 5 years).
- **Leakage Prevention**: All features are computed strictly chronologically as-of the match date (`pd.merge_asof(direction="backward")`).

---

## World Cup 2026 Evaluation Performance
Evaluated on the 72 actual group-stage matches of World Cup 2026:
- **Accuracy**: **59.72%** (43 of 72 correct in a 3-way outcome space).
- **Log Loss**: **0.8747**
- **Brier Score**: **0.5234**
- **ROC-AUC (Macro OVR)**: **0.7112**
- **Win F1-Score**: **0.7013**
- **Loss F1-Score**: **0.7500**

---

## Command-Line Usage

```bash
# 1. Train model on 88k+ historical matches
python src/wc26_engine.py --mode train

# 2. Evaluate specifically on the 72 World Cup 2026 group stage games
python src/wc26_engine.py --mode test_wc26

# 3. Predict match odds between any two national teams
python src/wc26_engine.py --mode predict --home "Iran" --away "England"

# 4. Run Monte Carlo simulations on batch matches
python src/wc26_engine.py --mode simulate --input src/sample_predictions.csv --runs 500
```
