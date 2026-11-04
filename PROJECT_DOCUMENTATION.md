# WC26 Analytics - Comprehensive Engine Documentation

## 1. Executive Summary

WC26 Analytics is an enterprise-grade football analytics and machine learning system engineered to model match probabilities (Home Win, Draw, Away Win) and compute betting odds (European Decimal and American Moneyline).

Originally conceived as a standalone Random Forest trained on a heavily decimated 670-match sample, the project has been overhauled into an ensemble pipeline leveraging **88,958 matches**, dynamic **Elo ratings**, **Dixon-Coles bivariate Poisson score distributions**, and **squad-level valuations** spanning **3.18 million lineups** and **1.89 million appearances**.

---

## 2. Problem Formulation & Modeling Objectives

International football prediction involves severe structural challenges:
1. **Low Scoring & High Variance**: Football matches feature few goals, resulting in frequent draws (~25-28%) and upsets.
2. **National Team Sparsity**: National teams play only 8-15 games per calendar year, far fewer than club teams.
3. **Severe Class Imbalance**: In 90-minute regulation time, outcomes follow approximately 45% Home Wins, 27% Draws, and 28% Away Wins.
4. **Betting Calibration Requirement**: Raw class predictions are insufficient for betting markets; the model must output well-calibrated posterior probabilities $P(y=k \mid \mathbf{x})$ to calculate fair odds:
   $$\text{Decimal Odds} = \frac{1}{P(y=k)}, \quad \text{American Odds} = \begin{cases} -\frac{100 \cdot P}{1 - P} & P \ge 0.5 \\ +\frac{100 \cdot (1 - P)}{P} & P < 0.5 \end{cases}$$

---

## 3. Dataset Architecture & Pipeline

The system integrates 12 relational CSV datasets updated through September 2026:

| Dataset | Records | Description |
| :--- | :--- | :--- |
| `games.csv` | 88,958 | Historical matches across 28 global competitions (club & national) |
| `game_lineups.csv` | 3,179,016 | Player lineups per match, distinguishing Starting XI vs. Substitutes |
| `appearances.csv` | 1,894,350 | Exact match minutes, goals, assists, and disciplinary cards |
| `game_events.csv` | 1,791,241 | Minute-by-minute events (goals, cards, substitutions, penalty shootouts) |
| `player_valuations.csv` | 656,301 | Chronological market value updates in EUR |
| `players.csv` | 30,531 | Static attributes (birth date, height, foot, citizenship, international caps) |
| `club_games.csv` | 177,916 | Team-game records (goals for, goals against, home/away indicator, manager) |
| `national_teams.csv` | 48 | Metadata for national teams participating in the 2026 World Cup |
| `competitions.csv` | 43 | Competition tiers, country IDs, confederations |
| `countries.csv` | 240 | FIFA country codes and confederation mappings |
| `clubs.csv` | 426 | Domestic club registry |
| `transfers.csv` | 74,484 | Historical player transfer fees and club transfers |

---

## 4. Key Mathematical & Physical Features

### 4.1 Dynamic Elo Rating Engine
Every team (club and national) is assigned a dynamic Elo rating $R$, updated chronologically match-by-match:
$$E_A = \frac{1}{1 + 10^{(R_B - R_A + H) / 400}}$$
where $H$ is home advantage ($H=0$ for neutral World Cup games, $H=65$ for domestic games).

The rating update incorporates competition importance ($K$) and goal margin multipliers ($G$):
$$R_A^{\text{new}} = R_A^{\text{old}} + K \cdot G \cdot (S_A - E_A)$$
- **$K$-factors**: $K=60$ for World Cup finals (`FIWC`), $K=50$ for continental national championships, $K=40$ for international cups, $K=30$ for domestic leagues.
- **Goal Margin Multiplier $G$**:
  $$G = \begin{cases} 1.0 & \text{if } |GD| \le 1 \\ 1.5 & \text{if } |GD| = 2 \\ \frac{11 + |GD|}{8} & \text{if } |GD| \ge 3 \end{cases}$$
- **FIFA Prior Seeding**: National teams without sufficient club history are initialized using their FIFA ranking: $R_0 = 2000 - 5 \cdot \text{Rank}$.

### 4.2 Dixon-Coles Bivariate Poisson Goal Model
Expected goals $\lambda_H$ and $\lambda_A$ are derived from Elo spreads:
$$\lambda_H = \text{clip}\left(1.35 + \frac{R_H - R_A}{500}, 0.2, 4.5\right), \quad \lambda_A = \text{clip}\left(1.15 - \frac{R_H - R_A}{500}, 0.2, 4.5\right)$$

Probabilities of exact scorelines $(x, y)$ are computed using the Dixon-Coles adjustment for low-scoring football matches ($\rho = -0.05$):
$$P(X=x, Y=y) = \tau_{\lambda_H, \lambda_A}(x, y) \cdot \frac{\lambda_H^x e^{-\lambda_H}}{x!} \cdot \frac{\lambda_A^y e^{-\lambda_A}}{y!}$$
where $\tau$ adjusts probabilities for scorelines (0-0, 1-0, 0-1, 1-1). Summing the lower triangle, diagonal, and upper triangle yields analytic prior probabilities for Win, Draw, and Loss.

### 4.3 High-Performance Squad Profiling
Rather than treating all 26 squad members uniformly, the feature pipeline strictly segments:
1. **Starting XI vs. Substitutes**:
   - `starter_market_value_sum`, `starter_market_value_mean`, `starter_market_value_max`
   - `starter_top3_val_mean` (Star Power index)
   - `starter_value_share` ($\text{Starters} / \text{Total Squad Value}$)
2. **Positional Valuations**:
   - Market value breakdown by role: `val_fw`, `val_mid`, `val_def`, `val_gk`.
3. **Appearance Momentum**:
   - Player minutes, goals, and assists over rolling 5 matches from `appearances.csv`.
4. **Differential & Ratio Features**:
   - Complete symmetry: for every numerical feature $f$, `diff_f = home_f - away_f`.
   - Key ratios: `ratio_starter_market_value`, `ratio_squad_market_value`, `ratio_elo`.

---

## 5. Machine Learning Ensemble Architecture

The model uses a soft-voting ensemble combining three complementary learning paradigms:

```
[ Input Match Vector (192 Features) ]
                 |
                 v
   [ ColumnTransformer Preprocessing ]
   - Numeric: Median Imputation
   - Categorical: One-Hot Encoding
                 |
                 v
    +------------------------------------------+
    |           Soft-Voting Ensemble           |
    |                                          |
    |  - LightGBM (Weight 0.45, lr=0.03)       |
    |  - CatBoost (Weight 0.40, depth=6)       |
    |  - Random Forest (Weight 0.15, depth=15) |
    +------------------------------------------+
                 |
                 v
      [ Calibrated Probabilities ]
                 |
                 +---> European Decimal Odds (1/P)
                 +---> American Moneyline Odds (+/-)
                 +---> Match Classification (1X2)
```

### Sample Weighting
To align training with World Cup inference without discarding the rich sample of club matches:
- **Tournament Importance**: National team matches receive a $3.5\times$ weight, international cups receive $2.0\times$.
- **Temporal Decay**: Exponential decay with a half-life of 5 years ($\lambda = \frac{1}{1825 \text{ days}}$) gives higher weight to recent form (2022-2026).

---

## 6. Audit of Resolved Codebase Defects

The complete audit identified and resolved seven critical bugs present in the original codebase:

1. **Inference Feature Leakage & Omission**:
   - *Original Bug*: `build_diff_features()` was omitted during inference in `build_inference_row()`. All 72 differential features were entirely `NaN` and replaced with median constants at test time.
   - *Resolution*: Implemented unified differential and ratio transformations executed identically across training and inference.

2. **`merge_asof` Column Collisions in Market Valuations**:
   - *Original Bug*: Historical valuation columns were pre-initialized to `NaN`, causing Pandas `merge_asof` to create `_x` and `_y` suffix collisions and zeroing out all market valuations.
   - *Resolution*: Direct `pd.merge_asof(direction="backward", allow_exact_matches=False)` with clean column resolution.

3. **Knockout Penalty Shootout Draw Distortion**:
   - *Original Bug*: Penalty shootout goals were added to match goals in Transfermarkt data (e.g. 0-0 knockout match recorded as 5-3). This severely degraded draw recall (down to 10.3%).
   - *Resolution*: Detects `Shootout` events from `game_events.csv` and rectifies regulation match outcomes to Draw (1).

4. **Homonym Player Disambiguation Failure**:
   - *Original Bug*: Player resolution picked `candidates.iloc[0]`, mapping modern stars (Rodri, Danilo, Ederson) to retired players from a decade prior.
   - *Resolution*: Multi-factor ranking: `[team_match, last_season, market_value_in_eur, international_caps]`.

5. **Flat Squad Aggregation**:
   - *Original Bug*: Averaged all 26 squad players together without distinction.
   - *Resolution*: Segmented aggregations distinguishing Starting XI, Substitutes, positional lines, and top-3 star player averages.

6. **Unused Appearances Dataset**:
   - *Original Bug*: `appearances.csv` (1.89M rows) was ignored, missing player match fitness and momentum.
   - *Resolution*: Built vector-optimized player appearance histories tracking rolling minutes, goals, assists, and rest days.

7. **Data Truncation**:
   - *Original Bug*: Inner-joins dropped 99.2% of matches (from 88,271 down to 670).
   - *Resolution*: Full relational joining retains 88,872 training matches and all 72 World Cup 2026 group stage games.

---

## 7. World Cup 2026 Empirical Evaluation

The model was evaluated strictly on the **72 unseen Group Stage matches** of the 2026 World Cup (`2026-06-11` to `2026-06-28`):

| Metric | Original RF (670 matches) | Upgraded Ensemble Engine |
| :--- | :--- | :--- |
| **Accuracy** | 39.3% (flat odds) | **59.72%** (43 of 72 correct) |
| **Multi-Class Log Loss** | 1.098 | **0.8747** |
| **Brier Score** | 0.667 | **0.5234** |
| **ROC-AUC (Macro OVR)** | 0.531 | **0.7112** |
| **Win F1-Score** | 0.482 | **0.7013** |
| **Loss F1-Score (Away Win)** | 0.441 | **0.7500** |
| **Inference Time per Match** | ~2.5s | **< 0.04s** |

### Sample Match Predictions vs. Actual Results

| Match | Actual Score | Actual | Predicted | Prob (Win/Draw/Loss) | Decimal Odds (1/X/2) | Correct? |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mexico vs. South Africa** | 2 - 0 | Win | Win | 64.3% / 25.6% / 10.1% | 1.55 / 3.91 / 9.92 | Yes |
| **United States vs. Paraguay** | 4 - 1 | Win | Win | 53.4% / 31.5% / 15.2% | 1.87 / 3.18 / 6.58 | Yes |
| **Haiti vs. Scotland** | 0 - 1 | Loss | Loss | 10.5% / 27.4% / 62.1% | 9.56 / 3.65 / 1.61 | Yes |
| **Germany vs. Curaçao** | 7 - 1 | Win | Win | 85.8% / 10.2% / 4.0% | 1.17 / 9.81 / 24.78 | Yes |
| **Sweden vs. Tunisia** | 5 - 1 | Win | Win | 69.3% / 21.8% / 8.8% | 1.44 / 4.58 / 11.30 | Yes |

---

## 8. Command-Line Reference

```bash
# Model training on full 88k+ match history
python src/wc26_engine.py --mode train

# Evaluate on the 72 World Cup 2026 matches
python src/wc26_engine.py --mode test_wc26

# Single match prediction & odds
python src/wc26_engine.py --mode predict --home "Brazil" --away "Germany"

# Batch Monte Carlo simulations (500 runs)
python src/wc26_engine.py --mode simulate --input src/sample_predictions.csv --runs 500

# Legacy compatibility wrapper
python src/wc26_random_forest.py --mode predict --predict-home-team "France" --predict-away-team "Argentina" --neutral-site
```
