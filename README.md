# WC26 Analytics

## Project Overview

This project implements a **Random Forest Classifier** to predict international football match outcomes (win, loss, or draw) based on comprehensive historical data about teams, players, managers, and match events. The model uses rich feature engineering to capture team form, player quality, managerial performance, and squad composition patterns.

## Dataset Composition

### Data Sources
The model integrates data from multiple CSV files covering:
- **Teams & Competitions**: Clubs, national teams, competitions, and confederations
- **Match Data**: Game lineups, team appearances, club games, game events
- **Player Data**: Player information, valuations, transfers
- **Match Events**: Goals, cards, substitutions, assists

### Dataset Scale
- **Total Matches in Dataset**: 670 matches across all available competitions
- **Training Records**: 520 matches (used for model training)
- **Test Records**: 150 matches (used for model evaluation)
- **Data Timeframe**: Historical matches up to January 2024
- **Match Types**: Both club and national team competitions included

### Supported Competitions
The `--competition-scope` parameter (default: "all") allows filtering to specific competition types:
- `all`: Includes all available competitions (club and national team matches)
- `national_team_competition`: National team matches only

**Note**: The current implementation uses `competition_type="all"` by default, so the model trains and evaluates on all available competition types without restriction.

## Train-Test Split Strategy

The dataset is divided using a **time-based (temporal) split** approach:

### Split Methodology
- **Sorting**: All matches sorted chronologically by date (and game_id for ties)
- **Split Ratio**: Test set represents approximately 20% of total data
- **Minimum Test Size**: Enforced minimum of 20 test games
- **Training Cutoff Date**: January 25, 2024 (last training match)
- **Test Start Date**: January 28, 2024 (first test match)

### Rationale
Time-based splitting is essential for sports prediction models because:
1. **Prevents Data Leakage**: Historical statistics can only use past matches, not future ones
2. **Realistic Evaluation**: Simulates actual prediction scenario where model forecasts unseen future matches
3. **Temporal Dynamics**: Captures changes in team form, player injuries, and squad compositions over time
4. **No Lookahead Bias**: Test data contains only matches chronologically after training data

## Feature Engineering & Model Architecture

### Feature Categories

#### 1. **Team Historical Performance Features**
Calculated from club_games.csv for each team before each match:
- `team_matches_before`: Cumulative match count
- `team_wins_before`, `team_draws_before`: Win/draw history
- `team_points_before`: Cumulative points (3 for win, 1 for draw, 0 for loss)
- `team_goals_for_before`, `team_goals_against_before`: Goal statistics
- `team_goal_diff_before`: Cumulative goal differential
- `team_win_rate_before`: Historical win percentage
- `team_points_per_game_before`: Average points per match
- `team_goal_diff_per_game_before`: Average goal differential per match
- `team_days_since_last_match`: Rest period before match
- **Recent Form Features**: `team_win_rate_last_5`, `team_points_last_5`, `team_goals_for_last_5`, `team_goals_against_last_5`, `team_goal_diff_last_5` (rolling 5-match averages)

#### 2. **Manager Performance Features**
Derived from team_history.csv using manager names:
- `manager_matches_before`: Cumulative matches under manager
- `manager_wins_before`, `manager_points_before`: Manager record
- `manager_goal_diff_before`: Goal differential under manager
- `manager_win_rate_before`: Managerial win percentage
- `manager_points_per_game_before`: Points per match average
- `manager_goal_diff_per_game_before`: Goal differential per match
- `manager_days_since_last_match`: Rest between matches
- `manager_clubs_before`: Number of clubs managed (for international managers)

#### 3. **Club Event History Features**
Aggregated from game_events.csv (goals, cards, substitutions):
- `club_event_matches_before`: Cumulative event match count
- `club_goal_events_before`: Cumulative goal events
- `club_card_events_before`: Cumulative card events
- **Recent Event Patterns**: `club_goal_events_last_5`, `club_card_events_last_5`, `club_sub_events_last_5` (rolling 5-match averages)
- `club_days_since_last_match`: Rest period

#### 4. **Player-Level Features**
From game_lineups.csv, matched against players.csv:
- **Demographic**: `age_years`, `height_in_cm`, `foot` (left/right/both/unknown)
- **Player History**: `appearance_before`, `days_since_last_appearance`, `start_before`, `sub_before`, `captain_before`
- **Event Record**: `goals_before`, `cards_before`, `subs_before`, `assists_before`
- **Player Event Matches**: `player_event_matches_before`
- **Recent Performance**: `goals_last_5`, `cards_last_5`, `assists_last_5`, `player_days_since_last_event_match`
- **Transfer History**: `transfer_count_before`, `days_since_last_transfer`
- **Market Value**: `market_value_before`, `valuation_count_before`, `days_since_last_valuation`
- **Role Classification**: One-hot encoded position categories (`role_goalkeeper`, `role_defender`, `role_midfielder`, `role_forward`, `role_unknown`)
- **Squad Status**: `is_captain`, `is_starting`, `is_substitute`, `unknown_player` (for missing player data)

#### 5. **Aggregated Lineup Features**
Computed from individual player features per team per match:
- **Squad Size**: `player_count`, `known_player_count`, `unknown_player_count`, `starter_count`, `sub_count`, `captain_count`
- **Age & Physical**: `age_mean`, `age_std`, `height_mean`, `height_std`
- **Squad Value**: `market_value_mean`, `market_value_sum`, `market_value_max`
- **Player History Aggregates**: `appearance_mean`, `days_since_last_appearance_mean`, `start_before_mean`, `sub_before_mean`, `captain_before_mean`
- **Performance Aggregates**: `goals_before_mean`, `cards_before_mean`, `assists_before_mean`, `goals_last_5_mean`, `cards_last_5_mean`, `assists_last_5_mean`
- **Role Distribution**: `role_goalkeeper_count`, `role_defender_count`, `role_midfielder_count`, `role_forward_count`, `role_unknown_count`
- **Foot Preference**: `foot_left_count`, `foot_right_count`, `foot_both_count`, `foot_unknown_count`

#### 6. **Match Context Features**
From games.csv:
- `competition_type`: Type of competition (national_team_competition, club league, cup, etc.)
- `round_group`: Round or group information
- `home_team_kind`: Team type (club or national_team)
- `away_team_kind`: Team type classification
- `home_team_confederation`: Continental confederation (UEFA, CONMEBOL, etc.)
- `away_team_confederation`: Confederation classification

#### 7. **Differential Features**
Computed as differences between home and away team features:
- `diff_*`: For all numeric features except leakage columns (e.g., `diff_team_win_rate_before`, `diff_manager_win_rate_before`, etc.)
- **Leakage Exclusions**: Features that would cause data leakage are explicitly excluded (`diff_game_id`, `diff_club_id`, `diff_club_goals`)

### Feature Pipeline Architecture

```
Raw Data (12 CSV files)
    ↓
[Team History] → Cumulative team stats, rolling averages
[Manager History] → Manager performance metrics
[Club Event History] → Event-based statistics
[Player Events] → Individual player goal/card/assist records
[Transfer History] → Player mobility patterns
[Valuation History] → Market value timeline
    ↓
[Lineup Features] → Per-player attributes merged with histories
    ↓
[Lineup Aggregation] → Aggregate squad-level statistics
    ↓
[Team Profile] → Merge all historical features at match level
    ↓
[Differential Features] → Home vs Away comparisons
    ↓
[Feature Matrix] → Remove leakage columns, categorize features
    ↓
[Preprocessing Pipeline]:
  - Numeric: Median imputation + missing indicator
  - Categorical: Most frequent imputation + OneHotEncoding
    ↓
[Model Input] → 520 training samples, ~150+ final features
```

## Model Architecture & Hyperparameters

### Algorithm: Random Forest Classifier

**Why Random Forest?**
- Handles both numeric and categorical features effectively
- Non-parametric, captures complex feature interactions without explicit feature engineering for interactions
- Robust to missing values and outliers
- Provides feature importance rankings
- Parallelizable for faster training

### Model Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 600 | Large ensemble for stable predictions |
| `max_depth` | 22 | Moderate depth to capture feature interactions without overfitting |
| `min_samples_split` | 10 | Prevent splitting on noise in small groups |
| `min_samples_leaf` | 7 | Ensure leaf nodes have sufficient samples for stable predictions |
| `max_features` | None | Consider all features at each split (default sqrt behavior) |
| `bootstrap` | True | Use bootstrap samples for diversity across trees |
| `class_weight` | "balanced_subsample" | Address class imbalance (wins are more common than losses/draws) |
| `random_state` | 42 | Ensure reproducibility |
| `n_jobs` | -1 | Use all available CPU cores for parallel tree building |

### Preprocessing Pipeline

#### Numeric Features
- **Strategy**: Median imputation
- **Add Indicator**: Yes (adds binary columns indicating originally missing values)
- **Rationale**: Median is robust to outliers; indicator captures missing pattern information

#### Categorical Features
- **Strategy 1**: Most frequent imputation
- **Strategy 2**: OneHotEncoder with `handle_unknown="ignore"` and `sparse_output=False`
- **Rationale**: Most frequent preserves category distribution; OneHotEncoder enables tree-based model to use categorical features; handle_unknown prevents errors on unseen categories

## Model Performance Metrics

### Evaluation Results (on 150 test matches)

| Metric | Value |
|--------|-------|
| **Overall Accuracy** | 59.33% |
| **Log Loss** | 0.9138 |
| **ROC-AUC (One-vs-Rest, Macro)** | 0.7332 |

### Per-Class Performance

| Outcome | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| Loss | 0.5397 | 0.6667 | 0.5965 | 51 |
| Draw | 0.3333 | 0.1034 | 0.1579 | 29 |
| Win | 0.6667 | 0.7429 | 0.7027 | 70 |

### Key Observations
- **Win Prediction**: Strong (74% recall) - model reliably identifies likely victories
- **Loss Prediction**: Moderate (67% recall) - decent at identifying likely defeats
- **Draw Prediction**: Weak (10% recall) - draws are hardest to predict (inherent difficulty in sports)
- **ROC-AUC**: Good discrimination (0.73) between match outcomes

## Data Processing Workflow

### 1. Data Loading & Validation
- Load 12 required CSV files (appearances, club_games, clubs, competitions, countries, game_events, game_lineups, games, national_teams, player_valuations, players, transfers)
- Validate all required files present; abort if missing

### 2. Historical Series Construction
- `build_team_history()`: Create cumulative statistics for each club/team per match
- `build_manager_history()`: Aggregate by manager across multiple clubs
- `build_club_event_history()`: Aggregate event statistics per team per match
- `build_player_event_history()`: Track goals/cards/assists per player
- `build_transfer_history()`: Count transfers and days since last transfer
- `build_valuation_history()`: Track market value timeline per player

### 3. Lineup Feature Assembly
- `build_lineup_player_features()`: Merge player demographics, histories, and valuations for each match lineup
- Join with player base data (age, height, foot)
- Merge with player event histories using `merge_asof()` for backward-looking matches
- Merge with transfer and valuation histories

### 4. Squad Aggregation
- `aggregate_lineup_features()`: Compute squad-level statistics (mean age, total market value, role distribution, etc.)
- Aggregate 30+ individual player features into squad-level metrics

### 5. Match-Level Profile Construction
- `build_team_profile()`: Assemble all team/manager/event/lineup features into match-level pairs
- Join home and away team profiles per match
- Merge with match metadata (competition type, round, result label)
- Fill missing confederations/country names

### 6. Differential Features
- `build_diff_features()`: Compute feature differences (home - away) for numeric columns
- Exclude leakage-prone columns (game_id, date, actual goals)

### 7. Feature Matrix Preparation
- `build_feature_matrix()`: Remove identifier columns and known-leakage columns
- Separate numeric and categorical features
- Remove all-NaN columns

### 8. Train-Test Splitting
- `time_based_split()`: Sort by date, split at 80/20 boundary, enforce minimum test size
- Return train/test features, labels, and date series for evaluation

### 9. Model Training & Evaluation
- Build preprocessing + classifier pipeline
- Fit on training data
- Generate predictions and probabilities on test set
- Calculate metrics (accuracy, log loss, ROC-AUC, per-class classification report)
- Save model, feature importance, ROC curves, and test predictions

## Usage

### Training Mode
```bash
python src/wc26_random_forest.py
```
Trains a new model and saves to `artifacts/`:
- `random_forest_model.joblib`: Trained model pipeline
- `feature_columns.json`: Feature names in training order
- `metrics.json`: Evaluation metrics
- `test_predictions.csv`: Predictions on test set
- `feature_importances.csv`: Top feature importance scores
- `random_forest_roc_curve.png`: ROC curves for each outcome

### Prediction Mode
```bash
python src/wc26_random_forest.py --mode predict \
  --predict-home-team "Argentina" \
  --predict-away-team "Brazil" \
  --predict-as-of-date "2024-02-01" \
  --predict-home-starting "player1,player2,..." \
  --predict-away-starting "player3,player4,..."
```

Or run interactively:
```bash
python src/wc26_random_forest.py
# Follow prompts to enter prediction details
```

Batch predictions from CSV:
```bash
python src/wc26_random_forest.py --prediction-csv sample_predictions.csv --simulation-runs 10
```
