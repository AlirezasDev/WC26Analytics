# WC26 Analytics: World Cup 2026 Match Predictor & Betting Odds Engine

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/Ensemble-LightGBM%20%7C%20CatBoost%20%7C%20RandomForest-brightgreen.svg)]()
[![Model Accuracy](https://img.shields.io/badge/WC26%20Accuracy-59.72%25-success.svg)]()
[![Evaluation](https://img.shields.io/badge/Evaluation-72%20WC26%20Matches-orange.svg)]()

WC26 Analytics is an enterprise-grade football analytics and machine learning system engineered to model 3-way match probabilities (**Home Win, Draw, Away Win**) and compute bookmaker-grade **European Decimal** ($1/P$) and **American Moneyline** ($+/-$) Odds for international and club matches.

Originally a standalone Random Forest trained on a decimated 670-match sample, this engine has been completely overhauled into a high-performance **Soft-Voting Ensemble** leveraging **88,958 matches**, dynamic **Elo ratings**, **Dixon-Coles bivariate Poisson score distributions**, and **squad-level valuations** spanning **3.18 million lineups** and **1.89 million player appearances**.

---

## Key Highlights & Performance

- **Full Scale Learning**: Trains on **88,872 historical matches** up to June 10, 2026 with tournament importance weighting (3.5x for national team competitions) and time decay.
- **Empirical World Cup 2026 Benchmark**: Evaluated strictly on the **72 unseen Group Stage matches** of the 2026 World Cup (`2026-06-11` to `2026-06-28`):
  - **3-Way Accuracy**: **59.72%** (43 of 72 correct outcomes).
  - **Multi-Class Log Loss**: **0.8747**
  - **Brier Score**: **0.5234**
  - **ROC-AUC (Macro OVR)**: **0.7112**
  - **Win F1-Score**: **0.7013** | **Loss F1-Score**: **0.7500**
- **Dynamic Elo Rating Engine**: Chronological Elo tracking across all 88k+ matches with tournament-specific $K$-factors ($K=60$ for World Cup, $K=50$ for national competitions) and FIFA World Ranking prior seeding.
- **Dixon-Coles Bivariate Poisson Goal Model**: Computes expected goals ($xG$) and low-scoring corrections (0-0, 1-0, 0-1, 1-1).
- **Deep Lineup Profiling**: Distinguishes Starting XI from Substitutes, calculates positional market values (Forwards, Midfielders, Defenders, Goalkeepers), and tracks top-3 star player averages.
- **Ultra-Fast Vectorized Inference**: Sub-40ms prediction latency per match.

---

## Architecture Overview

```
                      [ 88,958 Historical Matches ]
                                    |
          +-------------------------+-------------------------+
          |                                                   |
[ Dynamic Elo Engine ]                             [ Lineup & Squad Profiler ]
- K=60 World Cup Weight                            - 3.18M Game Lineups
- FIFA Rank Prior Seeding                          - 1.89M Appearances (Minutes/Goals/Form)
- Goal Margin Multipliers                          - Starting XI vs Bench Separation
          |                                        - Positional Market Valuations
          v                                                   |
[ Dixon-Coles Poisson Engine ]                                |
- Expected Goals (xG)                                         |
- Low-Scoreline Correlation (rho)                             |
          |                                                   |
          +-------------------------+-------------------------+
                                    |
                                    v
                     [ 192 Differential & Ratio Features ]
                                    |
                                    v
                    [ Soft-Voting ML Ensemble Pipeline ]
                    - LightGBM (Multiclass, Weight: 0.45)
                    - CatBoost (MultiClass, Weight: 0.40)
                    - Random Forest (Weight: 0.15)
                                    |
                                    v
                       [ Calibrated Probabilities ]
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
          v                         v                         v
   [ 1X2 Probabilities ]     [ European Decimal Odds ]   [ American Moneyline Odds ]
   (e.g., 64.7% / 21.1% / 14.2%)  (e.g., 1.55 / 4.73 / 7.05)   (e.g., -183 / +373 / +605)
```

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/AlirezasDev/WC26Analytics.git
cd WC26Analytics

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Quickstart & Usage

### 1. Match Prediction & Betting Odds
Predict the outcome, expected goals, and betting odds between any two national teams:

```bash
python src/wc26_engine.py --mode predict --home "Iran" --away "England"
```

Output:
```text
=================================================================
      MATCH PREDICTION & ODDS: IRAN vs ENGLAND
=================================================================
 Elo Rating:         Iran: 1804.6  |  England: 2092.1
 Expected Goals:     Iran 0.78 - 1.72 England
-----------------------------------------------------------------
 Outcome            | Probability  | Decimal Odds   | American Odds
-----------------------------------------------------------------
 Iran Win           |      14.2%  |         7.05  |         +605
 Draw               |      21.1%  |         4.73  |         +373
 England Win        |      64.7%  |         1.55  |         -183
=================================================================
```

### 2. World Cup 2026 Evaluation
Evaluate the model against all 72 actual group-stage matches of World Cup 2026:

```bash
python src/wc26_engine.py --mode test_wc26
```

### 3. Full Model Retraining
Train the ensemble model across all 88k+ matches:

```bash
python src/wc26_engine.py --mode train
```

### 4. Batch Monte Carlo Lineup Simulation
Run 500 position-aware Monte Carlo simulation runs across a batch of matches:

```bash
python src/wc26_engine.py --mode simulate --input src/sample_predictions.csv --runs 500
```

### 5. Backward Compatible Interface
You can also run predictions using the legacy command-line arguments:

```bash
python wc26_random_forest.py --mode predict --predict-home-team "France" --predict-away-team "Argentina" --neutral-site
```

---

## Project Structure

```
WC26Analytics/
├── artifacts/                           # Benchmark reports and feature metadata
│   ├── wc26_group_stage_evaluation.csv  # 72 match-by-match WC26 actual vs predicted results
│   ├── batch_predictions_results.csv    # 500-run Monte Carlo simulation results
│   ├── evaluation_metrics.json          # Complete evaluation metrics
│   └── feature_columns.json             # 192 feature names used in training
├── src/                                 # Core modular codebase
│   ├── config.py                        # Constants, team aliases, position role mapping
│   ├── data_loader.py                   # Data ingestion, shootout score rectification, fuzzy matching
│   ├── elo_poisson.py                   # Dynamic Elo engine & Dixon-Coles Poisson model
│   ├── player_profiler.py               # Vectorized appearances & squad starting vs bench aggregations
│   ├── feature_engineering.py           # Team/manager history, differential (diff_*) features & sample weights
│   ├── models.py                        # Ensemble pipeline (LightGBM + CatBoost + RF) & odds converter
│   ├── wc26_engine.py                   # Main CLI engine (train, test_wc26, predict, simulate)
│   ├── wc26_random_forest.py            # Backward compatibility wrapper
│   └── sample_predictions.csv           # 72 WC 2026 group stage match schedule
├── wc26_random_forest.py                # Root entrypoint redirecting to engine
├── PROJECT_DOCUMENTATION.md             # Comprehensive technical & mathematical documentation
├── PROJECT_SUMMARY.md                   # Concise summary of improvements and benchmarks
├── requirements.txt                     # Python package dependencies
├── .gitignore                           # Git ignore rules for datasets and checkpoints
└── README.md                            # Project overview & documentation
```

---

## Resolved Architectural Bugs

1. **Inference Differential Feature Leakage**: In the original implementation, `build_diff_features()` was never called during inference, resulting in 100% `NaN` values for all 72 `diff_*` columns. Fixed with unified pipeline transforms.
2. **Valuation Index Collisions in `merge_asof`**: Column collisions resulted in `_x`/`_y` suffixes, zeroing all player market values. Re-engineered with clean backward as-of merges.
3. **Knockout Penalty Shootout Bias**: Penalty shootout goals were previously added to match regulation scores in Transfermarkt data (e.g. 0-0 recorded as 5-3), destroying draw predictions. Resolved by detecting `Shootout` events and rectifying regulation outcomes to Draw (1).
4. **Homonym Player Disambiguation**: Resolved homonym ambiguity (e.g. Rodri, Danilo, Ederson) through 4-stage ranking by active season, citizenship, market value, and international caps.
5. **Starting XI vs Bench Separation**: Previously averaged all 26 squad players together. Now separately segments Starting XI from bench players and computes star-power metrics.

---

## License

This project is open-source under the MIT License.
