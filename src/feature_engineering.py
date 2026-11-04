"""
Feature engineering module: team history, manager form, differential/ratio features,
and sample weighting.
"""
import numpy as np
import pandas as pd

from data_loader import to_datetime, to_numeric_id

LEAKAGE_COLUMNS = {
    "game_id",
    "home_club_id",
    "away_club_id",
    "home_game_id",
    "away_game_id",
    "home_date",
    "away_date",
    "date",
    "round",
    "home_club_goals",
    "away_club_goals",
    "diff_game_id",
    "diff_club_id",
    "diff_club_goals",
}


def build_team_history(club_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate chronological cumulative and rolling 5-match statistics for each team.
    """
    frame = club_games.copy()
    frame["game_id"] = to_numeric_id(frame["game_id"])
    frame["club_id"] = to_numeric_id(frame["club_id"])
    frame = frame.merge(games[["game_id", "date"]], on="game_id", how="left")
    frame["date"] = to_datetime(frame["date"])
    frame["goals_for"] = to_numeric_id(frame["own_goals"]).fillna(0)
    frame["goals_against"] = to_numeric_id(frame["opponent_goals"]).fillna(0)
    frame["goal_diff"] = frame["goals_for"] - frame["goals_against"]
    frame["is_win"] = to_numeric_id(frame["is_win"]).fillna(0).astype(int)
    frame["is_draw"] = frame["goals_for"].eq(frame["goals_against"]).astype(int)
    frame["points"] = np.select([frame["is_win"].eq(1), frame["is_draw"].eq(1)], [3, 1], default=0)

    rows = []
    for club_id, group in frame.sort_values(["club_id", "date", "game_id"]).groupby("club_id", sort=False):
        g = group.sort_values(["date", "game_id"]).copy()
        g["team_matches_before"] = np.arange(len(g))
        g["team_wins_before"] = g["is_win"].shift(1).fillna(0).cumsum()
        g["team_draws_before"] = g["is_draw"].shift(1).fillna(0).cumsum()
        g["team_points_before"] = g["points"].shift(1).fillna(0).cumsum()
        g["team_goals_for_before"] = g["goals_for"].shift(1).fillna(0).cumsum()
        g["team_goals_against_before"] = g["goals_against"].shift(1).fillna(0).cumsum()
        g["team_goal_diff_before"] = g["goal_diff"].shift(1).fillna(0).cumsum()
        
        matches = g["team_matches_before"].replace(0, np.nan)
        g["team_win_rate_before"] = g["team_wins_before"] / matches
        g["team_points_per_game_before"] = g["team_points_before"] / matches
        g["team_goal_diff_per_game_before"] = g["team_goal_diff_before"] / matches
        g["team_days_since_last_match"] = g["date"].diff().dt.days

        g["team_win_rate_last_5"] = g["is_win"].shift(1).rolling(5, min_periods=1).mean()
        g["team_points_last_5"] = g["points"].shift(1).rolling(5, min_periods=1).mean()
        g["team_goals_for_last_5"] = g["goals_for"].shift(1).rolling(5, min_periods=1).mean()
        g["team_goals_against_last_5"] = g["goals_against"].shift(1).rolling(5, min_periods=1).mean()
        g["team_goal_diff_last_5"] = g["goal_diff"].shift(1).rolling(5, min_periods=1).mean()
        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def build_manager_history(team_history: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate managerial track record across all clubs and national teams.
    Ensures zero duplicate manager_name columns.
    """
    frame = team_history[["game_id", "club_id", "date", "own_manager_name", "points", "goal_diff", "is_win"]].copy()
    frame["manager_name"] = frame["own_manager_name"].fillna("Unknown").astype(str)

    rows = []
    for manager_name, group in frame.sort_values(["manager_name", "date", "game_id"]).groupby("manager_name", sort=False):
        g = group.sort_values(["date", "game_id"]).copy()
        g["manager_matches_before"] = np.arange(len(g))
        g["manager_wins_before"] = g["is_win"].shift(1).fillna(0).cumsum()
        g["manager_points_before"] = g["points"].shift(1).fillna(0).cumsum()
        g["manager_goal_diff_before"] = g["goal_diff"].shift(1).fillna(0).cumsum()

        matches = g["manager_matches_before"].replace(0, np.nan)
        g["manager_win_rate_before"] = g["manager_wins_before"] / matches
        g["manager_points_per_game_before"] = g["manager_points_before"] / matches
        g["manager_goal_diff_per_game_before"] = g["manager_goal_diff_before"] / matches
        g["manager_days_since_last_match"] = g["date"].diff().dt.days
        rows.append(g)

    manager = pd.concat(rows, ignore_index=True)
    cols = ["game_id", "club_id", "date", "manager_name"] + [
        c for c in manager.columns if c.startswith("manager_") and c != "manager_name"
    ]
    return manager[cols]


def build_match_profiles(
    games: pd.DataFrame,
    team_history: pd.DataFrame,
    manager_history: pd.DataFrame,
    squad_features: pd.DataFrame,
    team_dim: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assemble the complete match-level table combining Home & Away team form,
    manager stats, squad aggregations, and Elo/Poisson metrics.
    """
    team_safe = team_history[["game_id", "club_id", "date", *[c for c in team_history.columns if c.startswith("team_")]]].copy()
    manager_safe = manager_history.loc[:, ~manager_history.columns.duplicated()].copy()

    # Merge profiles per team-game
    profile = team_safe.merge(manager_safe, on=["game_id", "club_id", "date"], how="left")
    profile = profile.merge(squad_features, on=["game_id", "club_id"], how="left")

    home_lookup = team_dim.rename(
        columns={
            "team_id": "home_club_id",
            "team_kind": "home_team_kind",
            "team_confederation": "home_team_confederation",
            "team_country_name": "home_team_country_name",
        }
    )
    away_lookup = team_dim.rename(
        columns={
            "team_id": "away_club_id",
            "team_kind": "away_team_kind",
            "team_confederation": "away_team_confederation",
            "team_country_name": "away_team_country_name",
        }
    )

    pair = games.copy()
    pair = pair.merge(profile.add_prefix("home_"), left_on=["game_id", "home_club_id"], right_on=["home_game_id", "home_club_id"], how="left")
    pair = pair.merge(profile.add_prefix("away_"), left_on=["game_id", "away_club_id"], right_on=["away_game_id", "away_club_id"], how="left")
    pair = pair.merge(home_lookup, on="home_club_id", how="left")
    pair = pair.merge(away_lookup, on="away_club_id", how="left")

    pair["competition_type"] = pair["competition_type"].fillna("unknown")
    pair["home_team_kind"] = pair["home_team_kind"].fillna("unknown")
    pair["away_team_kind"] = pair["away_team_kind"].fillna("unknown")
    pair["home_team_confederation"] = pair["home_team_confederation"].fillna("unknown")
    pair["away_team_confederation"] = pair["away_team_confederation"].fillna("unknown")

    return pair


def build_diff_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Compute differential (Home - Away) and ratio (Home / Away) features.
    Guaranteed to run in BOTH training and inference pipelines.
    """
    pair = frame.copy()
    home_cols = [c for c in pair.columns if c.startswith("home_")]

    for h_col in home_cols:
        suffix = h_col[len("home_"):]
        a_col = f"away_{suffix}"
        if suffix in LEAKAGE_COLUMNS or a_col not in pair.columns:
            continue
        if pd.api.types.is_numeric_dtype(pair[h_col]) and pd.api.types.is_numeric_dtype(pair[a_col]):
            pair[f"diff_{suffix}"] = pair[h_col] - pair[a_col]

    # Specialized key ratios (Market Value, Elo, Goals)
    if "home_starter_market_value_sum" in pair.columns and "away_starter_market_value_sum" in pair.columns:
        pair["ratio_starter_market_value"] = (
            (pair["home_starter_market_value_sum"] + 1e5) / (pair["away_starter_market_value_sum"] + 1e5)
        )
    if "home_squad_market_value_sum" in pair.columns and "away_squad_market_value_sum" in pair.columns:
        pair["ratio_squad_market_value"] = (
            (pair["home_squad_market_value_sum"] + 1e5) / (pair["away_squad_market_value_sum"] + 1e5)
        )
    if "home_elo_before" in pair.columns and "away_elo_before" in pair.columns:
        pair["ratio_elo"] = pair["home_elo_before"] / pair["away_elo_before"].replace(0, 1500.0)

    return pair


def calculate_sample_weights(games: pd.DataFrame) -> np.ndarray:
    """
    Calculate sample weights for training:
    - Higher weight for national team tournaments (3.0x) and international cups (2.0x).
    - Time-decay weight favoring recent matches (2022-2026).
    """
    dates = to_datetime(games["date"])
    max_date = dates.max()
    
    # Recency weight: exponential decay with half-life ~ 5 years (1825 days)
    days_ago = (max_date - dates).dt.days.fillna(1825)
    recency_weight = np.exp(-days_ago / (5.0 * 365.25))

    comp_type = games["competition_type"].fillna("")
    comp_weight = np.where(
        comp_type == "national_team_competition",
        3.5,
        np.where(comp_type == "international_cup", 2.0, 1.0)
    )

    weights = recency_weight * comp_weight
    # Normalize to mean = 1.0
    return weights / np.mean(weights)
