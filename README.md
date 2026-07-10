# WC26 Analytics Summary

## What this project does
- Trains a **Random Forest classifier** to predict match outcomes: win, draw, or loss.
- Uses match history, team form, manager performance, lineup composition, player events, transfers, and valuations.
- Works on all available competitions by default (`competition_scope=all`).

## Data sources
- Primary match source: `games.csv`
- Auxiliary feature sources: `club_games.csv`, `game_lineups.csv`, `game_events.csv`, `players.csv`, `player_valuations.csv`, `transfers.csv`, plus teams/competitions metadata.

## Effective dataset used by the model
- **Final matches used**: 670
- **Training records**: 520
- **Test records**: 150
- The larger CSV `games.csv` contains **88,271 rows**, but only 670 matches survive the full feature engineering and matching process.

## Split method
- Uses a **time-based split** by match date.
- Training matches are the earlier subset, test matches are the later subset.
- Test set is about 20% of the final model dataset, with at least 20 games reserved for testing.
- Current split boundary in the saved run is January 25, 2024 for train and January 28, 2024 for test.

## Key feature groups
- Team form and cumulative performance history
- Manager win/points/goal differential history
- Club event statistics (goals, cards, substitutions)
- Player lineup stats, roles, age, height, and match history
- Aggregated squad-level metrics
- Match context and competition metadata
- Home vs away differential features

## Model details
- Algorithm: `RandomForestClassifier`
- Hyperparameters: 600 trees, max depth 22, min samples split 10, min samples leaf 7, balanced subsampling
- Preprocessing: median imputation for numeric values, one-hot encoding for categorical values, missing-value indicators

## Evaluation
- Accuracy: ~59.3%
- Log loss: ~0.91
- Macro ROC-AUC: ~0.73
- Draw prediction is weakest, while win/loss prediction is stronger.

## Important note
- The model currently does not use all 88,271 raw `games.csv` records directly; it only uses the 670 matches that have sufficient joined features from the auxiliary datasets.
