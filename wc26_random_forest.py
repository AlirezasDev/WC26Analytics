from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, label_binarize


RESULT_LABELS = {0: "loss", 1: "draw", 2: "win"}
LABEL_TO_INT = {label: value for value, label in RESULT_LABELS.items()}
ROLE_MAP = {
    "Goalkeeper": "goalkeeper",
    "Centre-Back": "defender",
    "Left-Back": "defender",
    "Right-Back": "defender",
    "Defender": "defender",
    "Left Wing-Back": "defender",
    "Right Wing-Back": "defender",
    "Defensive Midfield": "midfielder",
    "Central Midfield": "midfielder",
    "Midfield": "midfielder",
    "Attacking Midfield": "midfielder",
    "Left Midfield": "midfielder",
    "Right Midfield": "midfielder",
    "Left Winger": "forward",
    "Right Winger": "forward",
    "Centre-Forward": "forward",
    "Forward": "forward",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="train a WC26 RandomForest model")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "artifacts")
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--min-test-games", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--competition-scope",
        type=str,
        default="national_team_competition",
        choices=["national_team_competition", "all"],
        help="Filter matches to a scope before feature engineering",
    )
    parser.add_argument("--predict-home-team", type=str, default=None)
    parser.add_argument("--predict-away-team", type=str, default=None)
    parser.add_argument("--predict-home-manager", type=str, default=None)
    parser.add_argument("--predict-away-manager", type=str, default=None)
    parser.add_argument("--predict-home-starting", type=str, default=None)
    parser.add_argument("--predict-home-subs", type=str, default=None)
    parser.add_argument("--predict-away-starting", type=str, default=None)
    parser.add_argument("--predict-away-subs", type=str, default=None)
    parser.add_argument("--predict-as-of-date", type=str, default=None)
    return parser.parse_args()


def load_csv(data_dir: Path, name: str, parse_dates: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(data_dir / name, low_memory=False, parse_dates=parse_dates)


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def to_numeric_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().lower().replace("-", " ").replace("_", " ").split())


def parse_name_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def prompt_text(prompt: str, default: str | None = None) -> str | None:
    suffix = f" [{default}]" if default else ""
    value = input(f"{prompt}{suffix}: ").strip()
    if value:
        return value
    return default


def prompt_name_list(prompt: str, default: str | None = None) -> list[str]:
    value = prompt_text(prompt, default)
    return parse_name_list(value)


def prompt_interactive_inference() -> dict[str, object] | None:
    answer = prompt_text("Run an inference example now? (y/n)", "y")
    if not answer or answer.lower() not in {"y", "yes"}:
        return None

    return {
        "home_team_name": prompt_text("Home team name"),
        "away_team_name": prompt_text("Away team name"),
        "home_manager_name": prompt_text("Home manager name", "") or None,
        "away_manager_name": prompt_text("Away manager name", "") or None,
        "home_starting": prompt_name_list("Home starting lineup (comma-separated, optional)", ""),
        "home_subs": prompt_name_list("Home substitutes (comma-separated, optional)", ""),
        "away_starting": prompt_name_list("Away starting lineup (comma-separated, optional)", ""),
        "away_subs": prompt_name_list("Away substitutes (comma-separated, optional)", ""),
        "as_of_date_text": prompt_text("Prediction as-of date (YYYY-MM-DD, optional)", "") or None,
    }


def role_from_position(position: object) -> str:
    if pd.isna(position):
        return "unknown"
    return ROLE_MAP.get(str(position), "unknown")


def prefix_non_keys(frame: pd.DataFrame, prefix: str, key_columns: list[str]) -> pd.DataFrame:
    renamed = frame.copy()
    rename_map = {column: f"{prefix}{column}" for column in renamed.columns if column not in key_columns}
    return renamed.rename(columns=rename_map)


def normalize_team_dim(
    clubs: pd.DataFrame,
    national_teams: pd.DataFrame,
    competitions: pd.DataFrame,
    countries: pd.DataFrame,
) -> pd.DataFrame:
    country_lookup = countries[["country_id", "country_code"]].copy()
    country_lookup["country_id"] = to_numeric_id(country_lookup["country_id"])

    club_dim = clubs.copy()
    club_dim["club_id"] = to_numeric_id(club_dim["club_id"])
    club_dim = club_dim.merge(
        competitions[["competition_id", "country_id", "country_name", "confederation"]],
        left_on="domestic_competition_id",
        right_on="competition_id",
        how="left",
    )
    club_dim = club_dim.merge(country_lookup, on="country_id", how="left")
    club_dim = club_dim.rename(
        columns={
            "club_id": "team_id",
            "name": "team_name",
            "country_name": "team_country_name",
            "confederation": "team_confederation",
        }
    )
    club_dim["team_kind"] = "club"

    national_dim = national_teams.copy()
    national_dim["national_team_id"] = to_numeric_id(national_dim["national_team_id"])
    national_dim = national_dim.merge(country_lookup, on="country_id", how="left")
    national_dim = national_dim.rename(
        columns={
            "national_team_id": "team_id",
            "name": "team_name",
            "country_name": "team_country_name",
            "confederation": "team_confederation",
            "country_id": "team_country_id",
        }
    )
    national_dim["team_kind"] = "national_team"

    keep_columns = ["team_id", "team_name", "team_kind", "team_confederation", "team_country_name", "country_code"]
    club_dim = club_dim.reindex(columns=keep_columns)
    national_dim = national_dim.reindex(columns=keep_columns)
    team_dim = pd.concat([club_dim, national_dim], ignore_index=True)
    return team_dim.drop_duplicates(subset=["team_id"]).reset_index(drop=True)


def build_team_history(club_games: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    frame = club_games.copy()
    frame["game_id"] = to_numeric_id(frame["game_id"])
    frame["club_id"] = to_numeric_id(frame["club_id"])
    frame = frame.merge(games[["game_id", "date"]], on="game_id", how="left")
    frame["date"] = to_datetime(frame["date"])
    frame["is_home"] = frame["hosting"].fillna("").str.lower().eq("home").astype(int)
    frame["goals_for"] = to_numeric_id(frame["own_goals"]).fillna(0)
    frame["goals_against"] = to_numeric_id(frame["opponent_goals"]).fillna(0)
    frame["goal_diff"] = frame["goals_for"] - frame["goals_against"]
    frame["points"] = np.select([frame["is_win"].eq(1), frame["goals_for"].eq(frame["goals_against"])], [3, 1], default=0)
    frame["is_draw"] = frame["goals_for"].eq(frame["goals_against"]).astype(int)

    rows = []
    for club_id, group in frame.sort_values(["club_id", "date", "game_id"]).groupby("club_id", sort=False):
        group = group.sort_values(["date", "game_id"]).copy()
        group["team_matches_before"] = np.arange(len(group))
        group["team_wins_before"] = group["is_win"].shift(1).fillna(0).cumsum()
        group["team_draws_before"] = group["is_draw"].shift(1).fillna(0).cumsum()
        group["team_points_before"] = group["points"].shift(1).fillna(0).cumsum()
        group["team_goals_for_before"] = group["goals_for"].shift(1).fillna(0).cumsum()
        group["team_goals_against_before"] = group["goals_against"].shift(1).fillna(0).cumsum()
        group["team_goal_diff_before"] = group["goal_diff"].shift(1).fillna(0).cumsum()
        group["team_win_rate_before"] = group["team_wins_before"] / group["team_matches_before"].replace(0, np.nan)
        group["team_points_per_game_before"] = group["team_points_before"] / group["team_matches_before"].replace(0, np.nan)
        group["team_goal_diff_per_game_before"] = group["team_goal_diff_before"] / group["team_matches_before"].replace(0, np.nan)
        group["team_days_since_last_match"] = group["date"].diff().dt.days
        shifted_win = group["is_win"].shift(1)
        shifted_points = group["points"].shift(1)
        shifted_gf = group["goals_for"].shift(1)
        shifted_ga = group["goals_against"].shift(1)
        shifted_gd = group["goal_diff"].shift(1)
        group["team_win_rate_last_5"] = shifted_win.rolling(5, min_periods=1).mean()
        group["team_points_last_5"] = shifted_points.rolling(5, min_periods=1).mean()
        group["team_goals_for_last_5"] = shifted_gf.rolling(5, min_periods=1).mean()
        group["team_goals_against_last_5"] = shifted_ga.rolling(5, min_periods=1).mean()
        group["team_goal_diff_last_5"] = shifted_gd.rolling(5, min_periods=1).mean()
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def build_manager_history(team_history: pd.DataFrame) -> pd.DataFrame:
    frame = team_history[["game_id", "club_id", "date", "own_manager_name", "points", "goal_diff", "is_win"]].copy()
    frame["manager_name"] = frame["own_manager_name"].fillna("Unknown").astype(str)
    rows = []
    for manager_name, group in frame.sort_values(["manager_name", "date", "game_id"]).groupby("manager_name", sort=False):
        group = group.sort_values(["date", "game_id"]).copy()
        group["manager_matches_before"] = np.arange(len(group))
        group["manager_wins_before"] = group["is_win"].shift(1).fillna(0).cumsum()
        group["manager_points_before"] = group["points"].shift(1).fillna(0).cumsum()
        group["manager_goal_diff_before"] = group["goal_diff"].shift(1).fillna(0).cumsum()
        group["manager_win_rate_before"] = group["manager_wins_before"] / group["manager_matches_before"].replace(0, np.nan)
        group["manager_points_per_game_before"] = group["manager_points_before"] / group["manager_matches_before"].replace(0, np.nan)
        group["manager_goal_diff_per_game_before"] = group["manager_goal_diff_before"] / group["manager_matches_before"].replace(0, np.nan)
        group["manager_days_since_last_match"] = group["date"].diff().dt.days
        group["manager_clubs_before"] = group["club_id"].expanding().apply(lambda s: pd.Series(s).nunique(), raw=False).shift(1)
        rows.append(group)
    manager = pd.concat(rows, ignore_index=True)
    keep_columns = ["game_id", "club_id", "date", "manager_name"] + [
        column for column in manager.columns if column.startswith("manager_")
    ]
    return manager[keep_columns]


def build_club_event_history(game_events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    events = game_events.copy()
    events["game_id"] = to_numeric_id(events["game_id"])
    events["club_id"] = to_numeric_id(events["club_id"])
    events["date"] = to_datetime(events["date"])
    events["goal_event"] = events["type"].eq("Goals").astype(int)
    events["card_event"] = events["type"].eq("Cards").astype(int)
    events["sub_event"] = events["type"].eq("Substitutions").astype(int)

    club_events = (
        events.groupby(["club_id", "game_id", "date"], as_index=False)
        .agg(
            club_goal_events=("goal_event", "sum"),
            club_card_events=("card_event", "sum"),
            club_sub_events=("sub_event", "sum"),
        )
        .sort_values(["club_id", "date", "game_id"])
    )

    rows = []
    for club_id, group in club_events.groupby("club_id", sort=False):
        group = group.sort_values(["date", "game_id"]).copy()
        group["club_event_matches_before"] = np.arange(len(group))
        group["club_goal_events_before"] = group["club_goal_events"].shift(1).fillna(0).cumsum()
        group["club_card_events_before"] = group["club_card_events"].shift(1).fillna(0).cumsum()
        group["club_sub_events_before"] = group["club_sub_events"].shift(1).fillna(0).cumsum()
        group["club_goal_events_last_5"] = group["club_goal_events"].shift(1).rolling(5, min_periods=1).mean()
        group["club_card_events_last_5"] = group["club_card_events"].shift(1).rolling(5, min_periods=1).mean()
        group["club_sub_events_last_5"] = group["club_sub_events"].shift(1).rolling(5, min_periods=1).mean()
        group["club_days_since_last_match"] = group["date"].diff().dt.days
        rows.append(group)
    events_out = pd.concat(rows, ignore_index=True)
    keep_columns = ["game_id", "club_id", "date"] + [
        column
        for column in events_out.columns
        if column.endswith("_before") or column.endswith("_last_5") or column in {"club_event_matches_before", "club_days_since_last_match"}
    ]
    return events_out[keep_columns]


def build_player_event_history(game_events: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    events = game_events.copy()
    events["game_id"] = to_numeric_id(events["game_id"])
    events["player_id"] = to_numeric_id(events["player_id"])
    events["date"] = to_datetime(events["date"])
    events["goal_event"] = events["type"].eq("Goals").astype(int)
    events["card_event"] = events["type"].eq("Cards").astype(int)
    events["sub_event"] = events["type"].eq("Substitutions").astype(int)
    events["assist_event"] = events["player_assist_id"].notna().astype(int)

    player_events = (
        events.groupby(["player_id", "game_id", "date"], as_index=False)
        .agg(
            goals_in_game=("goal_event", "sum"),
            cards_in_game=("card_event", "sum"),
            subs_in_game=("sub_event", "sum"),
            assists_in_game=("assist_event", "sum"),
        )
        .sort_values(["player_id", "date", "game_id"])
    )

    rows = []
    for player_id, group in player_events.groupby("player_id", sort=False):
        group = group.sort_values(["date", "game_id"]).copy()
        group["player_event_matches_before"] = np.arange(len(group))
        group["goals_before"] = group["goals_in_game"].shift(1).fillna(0).cumsum()
        group["cards_before"] = group["cards_in_game"].shift(1).fillna(0).cumsum()
        group["subs_before"] = group["subs_in_game"].shift(1).fillna(0).cumsum()
        group["assists_before"] = group["assists_in_game"].shift(1).fillna(0).cumsum()
        group["goals_last_5"] = group["goals_in_game"].shift(1).rolling(5, min_periods=1).mean()
        group["cards_last_5"] = group["cards_in_game"].shift(1).rolling(5, min_periods=1).mean()
        group["assists_last_5"] = group["assists_in_game"].shift(1).rolling(5, min_periods=1).mean()
        group["player_days_since_last_event_match"] = group["date"].diff().dt.days
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def build_transfer_history(transfers: pd.DataFrame) -> pd.DataFrame:
    frame = transfers.copy()
    frame["player_id"] = to_numeric_id(frame["player_id"])
    frame["transfer_date"] = to_datetime(frame["transfer_date"])
    frame = frame.sort_values(["player_id", "transfer_date"]).reset_index(drop=True)
    frame["transfer_count_before"] = frame.groupby("player_id").cumcount()
    frame["days_since_last_transfer"] = frame.groupby("player_id")["transfer_date"].diff().dt.days
    return frame[["player_id", "transfer_date", "transfer_count_before", "days_since_last_transfer"]]


def build_valuation_history(player_valuations: pd.DataFrame) -> pd.DataFrame:
    frame = player_valuations.copy()
    frame["player_id"] = to_numeric_id(frame["player_id"])
    frame["date"] = to_datetime(frame["date"])
    frame["market_value_in_eur"] = to_numeric_id(frame["market_value_in_eur"])
    frame = frame.sort_values(["player_id", "date"]).reset_index(drop=True)
    frame["valuation_count_before"] = frame.groupby("player_id").cumcount()
    frame["days_since_last_valuation"] = frame.groupby("player_id")["date"].diff().dt.days
    frame = frame.rename(columns={"market_value_in_eur": "market_value_before"})
    return frame[["player_id", "date", "market_value_before", "valuation_count_before", "days_since_last_valuation"]]


def merge_asof_player_safe(
    frame: pd.DataFrame,
    history: pd.DataFrame,
    right_date_col: str,
    value_columns: list[str],
) -> pd.DataFrame:
    left = frame.copy()
    right = history.copy()

    for column in value_columns:
        if column not in left.columns:
            left[column] = np.nan

    valid_left_mask = left["player_id"].notna() & left["date"].notna()
    valid_right_mask = right["player_id"].notna() & right[right_date_col].notna()

    left_valid = left.loc[valid_left_mask].copy()
    right_valid = right.loc[valid_right_mask].copy()

    left_valid["player_id"] = left_valid["player_id"].astype("Int64")
    right_valid["player_id"] = right_valid["player_id"].astype("Int64")

    left_valid = left_valid.sort_values(["date", "player_id", "game_id"])
    right_valid = right_valid.sort_values([right_date_col, "player_id"])

    if left_valid.empty or right_valid.empty:
        return left

    merged_valid = pd.merge_asof(
        left_valid,
        right_valid[["player_id", right_date_col, *value_columns]].rename(columns={right_date_col: "_history_date"}),
        by="player_id",
        left_on="date",
        right_on="_history_date",
        direction="backward",
        allow_exact_matches=False,
    )
    merged_valid = merged_valid.drop(columns=["_history_date"], errors="ignore")

    left.loc[valid_left_mask, :] = merged_valid.reindex(left.loc[valid_left_mask].index)
    return left


def build_lineup_player_features(
    game_lineups: pd.DataFrame,
    games: pd.DataFrame,
    players: pd.DataFrame,
    game_events: pd.DataFrame,
    player_valuations: pd.DataFrame,
    transfers: pd.DataFrame,
) -> pd.DataFrame:
    frame = game_lineups.copy()
    frame["game_id"] = to_numeric_id(frame["game_id"])
    frame["player_id"] = to_numeric_id(frame["player_id"])
    frame["club_id"] = to_numeric_id(frame["club_id"])
    frame["date"] = to_datetime(frame["date"])
    frame["is_starting"] = frame["type"].eq("starting_lineup").astype(int)
    frame["is_substitute"] = frame["type"].eq("substitutes").astype(int)
    frame["is_captain"] = to_numeric_id(frame["team_captain"]).fillna(0).astype(int)
    frame["role"] = frame["position"].apply(role_from_position)
    frame["unknown_player"] = frame["player_id"].isna().astype(int)

    player_base = players.copy()
    player_base["player_id"] = to_numeric_id(player_base["player_id"])
    player_base["date_of_birth"] = to_datetime(player_base["date_of_birth"])
    player_base["height_in_cm"] = to_numeric_id(player_base["height_in_cm"])
    player_base = player_base[["player_id", "date_of_birth", "height_in_cm", "foot"]]
    frame = frame.merge(player_base, on="player_id", how="left")
    frame["age_years"] = (frame["date"] - frame["date_of_birth"]).dt.days / 365.25

    frame["appearance_before"] = frame.sort_values(["player_id", "date", "game_id"]).groupby("player_id").cumcount()
    frame["days_since_last_appearance"] = frame.sort_values(["player_id", "date", "game_id"]).groupby("player_id")["date"].diff().dt.days
    frame["start_before"] = frame.groupby("player_id")["is_starting"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    frame["sub_before"] = frame.groupby("player_id")["is_substitute"].transform(lambda s: s.shift(1).fillna(0).cumsum())
    frame["captain_before"] = frame.groupby("player_id")["is_captain"].transform(lambda s: s.shift(1).fillna(0).cumsum())

    player_events = build_player_event_history(game_events, games)
    frame = frame.merge(
        player_events[
            [
                "player_id",
                "game_id",
                "goals_before",
                "cards_before",
                "subs_before",
                "assists_before",
                "player_event_matches_before",
                "goals_last_5",
                "cards_last_5",
                "assists_last_5",
                "player_days_since_last_event_match",
            ]
        ],
        on=["player_id", "game_id"],
        how="left",
    )

    valuation_history = build_valuation_history(player_valuations)
    frame = frame.sort_values(["date", "player_id", "game_id"]).reset_index(drop=True)
    frame = merge_asof_player_safe(
        frame=frame,
        history=valuation_history,
        right_date_col="date",
        value_columns=["market_value_before", "valuation_count_before", "days_since_last_valuation"],
    )

    transfer_history = build_transfer_history(transfers)
    frame = merge_asof_player_safe(
        frame=frame,
        history=transfer_history,
        right_date_col="transfer_date",
        value_columns=["transfer_count_before", "days_since_last_transfer"],
    )

    frame["team_captain"] = frame["is_captain"]
    frame["role_goalkeeper"] = frame["role"].eq("goalkeeper").astype(int)
    frame["role_defender"] = frame["role"].eq("defender").astype(int)
    frame["role_midfielder"] = frame["role"].eq("midfielder").astype(int)
    frame["role_forward"] = frame["role"].eq("forward").astype(int)
    frame["role_unknown"] = frame["role"].eq("unknown").astype(int)
    frame["foot"] = frame["foot"].fillna("unknown").astype(str).str.lower()
    for value in ["left", "right", "both", "unknown"]:
        frame[f"foot_{value}"] = frame["foot"].eq(value).astype(int)
    return frame


def aggregate_lineup_features(lineup_features: pd.DataFrame) -> pd.DataFrame:
    frame = lineup_features.copy()
    expected_columns = [
        "age_years",
        "height_in_cm",
        "market_value_before",
        "valuation_count_before",
        "days_since_last_valuation",
        "appearance_before",
        "days_since_last_appearance",
        "start_before",
        "sub_before",
        "captain_before",
        "goals_before",
        "cards_before",
        "subs_before",
        "assists_before",
        "player_event_matches_before",
        "goals_last_5",
        "cards_last_5",
        "assists_last_5",
        "player_days_since_last_event_match",
        "transfer_count_before",
        "days_since_last_transfer",
        "team_captain",
        "unknown_player",
        "is_starting",
        "is_substitute",
        "role_goalkeeper",
        "role_defender",
        "role_midfielder",
        "role_forward",
        "role_unknown",
        "foot_left",
        "foot_right",
        "foot_both",
        "foot_unknown",
    ]
    for column in expected_columns:
        if column not in frame.columns:
            frame[column] = np.nan

    agg_spec = {
        "player_count": ("player_id", "size"),
        "known_player_count": ("unknown_player", lambda s: int((1 - s.fillna(0)).sum())),
        "unknown_player_count": ("unknown_player", "sum"),
        "starter_count": ("is_starting", "sum"),
        "sub_count": ("is_substitute", "sum"),
        "captain_count": ("team_captain", "sum"),
        "age_mean": ("age_years", "mean"),
        "age_std": ("age_years", "std"),
        "height_mean": ("height_in_cm", "mean"),
        "height_std": ("height_in_cm", "std"),
        "market_value_mean": ("market_value_before", "mean"),
        "market_value_sum": ("market_value_before", "sum"),
        "market_value_max": ("market_value_before", "max"),
        "valuation_count_mean": ("valuation_count_before", "mean"),
        "days_since_last_valuation_mean": ("days_since_last_valuation", "mean"),
        "appearance_mean": ("appearance_before", "mean"),
        "days_since_last_appearance_mean": ("days_since_last_appearance", "mean"),
        "start_before_mean": ("start_before", "mean"),
        "sub_before_mean": ("sub_before", "mean"),
        "captain_before_mean": ("captain_before", "mean"),
        "goals_before_mean": ("goals_before", "mean"),
        "cards_before_mean": ("cards_before", "mean"),
        "assists_before_mean": ("assists_before", "mean"),
        "player_event_matches_before_mean": ("player_event_matches_before", "mean"),
        "goals_last_5_mean": ("goals_last_5", "mean"),
        "cards_last_5_mean": ("cards_last_5", "mean"),
        "assists_last_5_mean": ("assists_last_5", "mean"),
        "player_days_since_last_event_match_mean": ("player_days_since_last_event_match", "mean"),
        "transfer_count_before_mean": ("transfer_count_before", "mean"),
        "days_since_last_transfer_mean": ("days_since_last_transfer", "mean"),
        "role_goalkeeper_count": ("role_goalkeeper", "sum"),
        "role_defender_count": ("role_defender", "sum"),
        "role_midfielder_count": ("role_midfielder", "sum"),
        "role_forward_count": ("role_forward", "sum"),
        "role_unknown_count": ("role_unknown", "sum"),
        "foot_left_count": ("foot_left", "sum"),
        "foot_right_count": ("foot_right", "sum"),
        "foot_both_count": ("foot_both", "sum"),
        "foot_unknown_count": ("foot_unknown", "sum"),
    }

    grouped = frame.groupby(["game_id", "club_id"], as_index=False).agg(**agg_spec)
    grouped["unknown_player_share"] = grouped["unknown_player_count"] / grouped["player_count"].replace(0, np.nan)
    grouped["captain_share"] = grouped["captain_count"] / grouped["player_count"].replace(0, np.nan)
    grouped["market_value_per_player"] = grouped["market_value_sum"] / grouped["player_count"].replace(0, np.nan)
    grouped["defender_share"] = grouped["role_defender_count"] / grouped["player_count"].replace(0, np.nan)
    grouped["midfielder_share"] = grouped["role_midfielder_count"] / grouped["player_count"].replace(0, np.nan)
    grouped["forward_share"] = grouped["role_forward_count"] / grouped["player_count"].replace(0, np.nan)
    grouped["goalkeeper_share"] = grouped["role_goalkeeper_count"] / grouped["player_count"].replace(0, np.nan)
    return grouped


def build_team_profile(
    games: pd.DataFrame,
    team_history: pd.DataFrame,
    manager_history: pd.DataFrame,
    club_event_history: pd.DataFrame,
    lineup_summary: pd.DataFrame,
    team_dim: pd.DataFrame,
) -> pd.DataFrame:
    team_safe = team_history[["game_id", "club_id", "date", "own_manager_name", *[c for c in team_history.columns if c.startswith("team_")]]].copy()
    manager_safe = manager_history[
        ["game_id", "club_id", "date", *[c for c in manager_history.columns if c.startswith("manager_") and c not in {"game_id", "club_id", "date"}]]
    ].copy()
    event_safe = club_event_history[
        ["game_id", "club_id", "date", *[c for c in club_event_history.columns if c.startswith("club_") and c not in {"game_id", "club_id", "date"}]]
    ].copy()

    profile = team_safe.merge(manager_safe, on=["game_id", "club_id", "date"], how="left", suffixes=("", "_manager"))
    profile = profile.merge(event_safe, on=["game_id", "club_id", "date"], how="left", suffixes=("", "_club_event"))
    profile = profile.merge(lineup_summary, on=["game_id", "club_id"], how="left")

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

    games_meta = games[["game_id", "home_club_id", "away_club_id", "competition_type", "round", "date", "home_club_goals", "away_club_goals"]].copy()
    games_meta["home_club_id"] = to_numeric_id(games_meta["home_club_id"])
    games_meta["away_club_id"] = to_numeric_id(games_meta["away_club_id"])
    games_meta["date"] = to_datetime(games_meta["date"])
    games_meta["round_group"] = games_meta["round"].fillna("unknown").astype(str)
    games_meta["label"] = np.select(
        [games_meta["home_club_goals"] > games_meta["away_club_goals"], games_meta["home_club_goals"] == games_meta["away_club_goals"]],
        [LABEL_TO_INT["win"], LABEL_TO_INT["draw"]],
        default=LABEL_TO_INT["loss"],
    )

    pair = games_meta.merge(profile.add_prefix("home_"), left_on=["game_id", "home_club_id"], right_on=["home_game_id", "home_club_id"], how="left")
    pair = pair.merge(profile.add_prefix("away_"), left_on=["game_id", "away_club_id"], right_on=["away_game_id", "away_club_id"], how="left")
    pair = pair.merge(home_lookup, on="home_club_id", how="left")
    pair = pair.merge(away_lookup, on="away_club_id", how="left")

    pair["competition_type"] = pair["competition_type"].fillna("unknown")
    pair["home_team_kind"] = pair["home_team_kind"].fillna("unknown")
    pair["away_team_kind"] = pair["away_team_kind"].fillna("unknown")
    pair["home_team_confederation"] = pair["home_team_confederation"].fillna("unknown")
    pair["away_team_confederation"] = pair["away_team_confederation"].fillna("unknown")
    pair["home_team_country_name"] = pair["home_team_country_name"].fillna("unknown")
    pair["away_team_country_name"] = pair["away_team_country_name"].fillna("unknown")

    return pair


def build_diff_features(frame: pd.DataFrame) -> pd.DataFrame:
    pair = frame.copy()
    home_columns = [column for column in pair.columns if column.startswith("home_")]
    for home_column in home_columns:
        suffix = home_column[len("home_"):]
        away_column = f"away_{suffix}"
        if away_column in pair.columns and pd.api.types.is_numeric_dtype(pair[home_column]) and pd.api.types.is_numeric_dtype(pair[away_column]):
            pair[f"diff_{suffix}"] = pair[home_column] - pair[away_column]
    return pair


def build_feature_matrix(pair: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    features = pair.copy()
    features = features.loc[:, ~features.columns.duplicated()].copy()
    y = features.pop("label")
    drop_columns = [
        "game_id",
        "date",
        "round",
        "home_club_id",
        "away_club_id",
        "home_club_goals",
        "away_club_goals",
        "own_manager_name",
    ]
    for column in drop_columns:
        if column in features.columns:
            features = features.drop(columns=column)

    categorical_columns = [
        column
        for column in [
            "competition_type",
            "round_group",
            "home_team_kind",
            "away_team_kind",
            "home_team_confederation",
            "away_team_confederation",
        ]
        if column in features.columns
    ]
    numeric_columns = [column for column in features.columns if column not in categorical_columns and pd.api.types.is_numeric_dtype(features[column])]
    numeric_columns = [column for column in numeric_columns if features[column].notna().any()]
    drop_all_nan = [column for column in features.columns if column not in numeric_columns and column not in categorical_columns and features[column].isna().all()]
    if drop_all_nan:
        features = features.drop(columns=drop_all_nan)
    return features, y, numeric_columns, categorical_columns


def build_model(numeric_columns: list[str], categorical_columns: list[str], n_estimators: int, random_state: int) -> Pipeline:
    numeric_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy="median", add_indicator=True))])
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_columns),
            ("cat", categorical_pipe, categorical_columns),
        ]
    )
    classifier = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=None,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])


def time_based_split(features: pd.DataFrame, y: pd.Series, dates: pd.Series, test_fraction: float, min_test_games: int):
    ordered = features.copy()
    ordered["_date"] = to_datetime(dates)
    ordered["_label"] = y.values
    ordered = ordered.sort_values(["_date", "game_id"] if "game_id" in ordered.columns else ["_date"]).reset_index(drop=True)
    unique_dates = ordered["_date"].dropna().sort_values().unique()
    if len(unique_dates) == 0:
        raise ValueError("No valid dates available for splitting")
    cutoff_index = max(0, int(len(unique_dates) * (1 - test_fraction)) - 1)
    cutoff_date = unique_dates[cutoff_index]
    train_mask = ordered["_date"] <= cutoff_date
    test_mask = ordered["_date"] > cutoff_date
    if test_mask.sum() < min_test_games:
        cutoff_date = ordered["_date"].iloc[max(0, len(ordered) - min_test_games - 1)]
        train_mask = ordered["_date"] <= cutoff_date
        test_mask = ordered["_date"] > cutoff_date
    x_train = ordered.loc[train_mask].drop(columns=["_date", "_label"])
    x_test = ordered.loc[test_mask].drop(columns=["_date", "_label"])
    y_train = ordered.loc[train_mask, "_label"]
    y_test = ordered.loc[test_mask, "_label"]
    return x_train, x_test, y_train, y_test, ordered.loc[train_mask, "_date"], ordered.loc[test_mask, "_date"]


def save_curves(output_dir: Path, y_true: pd.Series, y_proba: np.ndarray) -> None:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    class_indices = list(RESULT_LABELS.keys())

    plt.figure(figsize=(9, 7))
    for class_index, class_label in RESULT_LABELS.items():
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_index], y_proba[:, class_index])
        auc_score = roc_auc_score(y_true_bin[:, class_index], y_proba[:, class_index])
        plt.step(fpr, tpr, where="post", alpha=0.75, label=f"{class_label} (AUC={auc_score:.3f})")

    fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_proba.ravel())
    roc_auc_micro = roc_auc_score(y_true_bin, y_proba, average="micro", multi_class="ovr")
    plt.step(fpr_micro, tpr_micro, where="post", color="black", linewidth=2.0, label=f"micro-average (AUC={roc_auc_micro:.3f})")

    all_fpr = np.unique(np.concatenate([roc_curve(y_true_bin[:, index], y_proba[:, index])[0] for index in class_indices]))
    mean_tpr = np.zeros_like(all_fpr)
    for index in class_indices:
        fpr, tpr, _ = roc_curve(y_true_bin[:, index], y_proba[:, index])
        mean_tpr += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= len(class_indices)
    roc_auc_macro = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
    plt.step(all_fpr, mean_tpr, where="post", color="#d62728", linewidth=2.0, label=f"macro-average (AUC={roc_auc_macro:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 7))
    for class_index, class_label in RESULT_LABELS.items():
        precision, recall, _ = precision_recall_curve(y_true_bin[:, class_index], y_proba[:, class_index])
        ap_score = average_precision_score(y_true_bin[:, class_index], y_proba[:, class_index])
        plt.step(recall, precision, where="post", alpha=0.75, label=f"{class_label} (AP={ap_score:.3f})")

    precision_micro, recall_micro, _ = precision_recall_curve(y_true_bin.ravel(), y_proba.ravel())
    ap_micro = average_precision_score(y_true_bin, y_proba, average="micro")
    plt.step(recall_micro, precision_micro, where="post", color="black", linewidth=2.0, label=f"micro-average (AP={ap_micro:.3f})")

    recall_grid = np.linspace(0, 1, 200)
    macro_precision = np.zeros_like(recall_grid)
    for index in class_indices:
        precision, recall, _ = precision_recall_curve(y_true_bin[:, index], y_proba[:, index])
        macro_precision += np.interp(recall_grid, recall[::-1], precision[::-1], left=1.0, right=precision[0])
    macro_precision /= len(class_indices)
    ap_macro = average_precision_score(y_true_bin, y_proba, average="macro")
    plt.step(recall_grid, macro_precision, where="post", color="#d62728", linewidth=2.0, label=f"macro-average (AP={ap_macro:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curve.png", dpi=160)
    plt.close()


def save_feature_importance(output_dir: Path, model: Pipeline, top_n: int = 40) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    classifier = model.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()
    importance = pd.DataFrame({"feature": feature_names, "importance": classifier.feature_importances_}).sort_values("importance", ascending=False)
    importance.to_csv(output_dir / "feature_importances.csv", index=False)
    subset = importance.head(top_n).sort_values("importance")
    plt.figure(figsize=(10, 10))
    plt.barh(subset["feature"], subset["importance"], color="#1f77b4")
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importances.png", dpi=160)
    plt.close()
    return importance


def resolve_team_id(team_dim: pd.DataFrame, team_name: str) -> float:
    normalized = normalize_name(team_name)
    matches = team_dim[team_dim["team_name"].fillna("").map(normalize_name).eq(normalized)]
    if matches.empty:
        return float("nan")
    return float(matches.iloc[0]["team_id"])


def resolve_player_row(players: pd.DataFrame, player_name: str) -> pd.Series | None:
    normalized = normalize_name(player_name)
    candidates = players[players["name"].fillna("").map(normalize_name).eq(normalized)]
    if candidates.empty:
        full_name = players["first_name"].fillna("").str.cat(players["last_name"].fillna(""), sep=" ").str.strip().map(normalize_name)
        candidates = players[full_name.eq(normalized)]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def build_custom_lineups(
    home_team_id: float,
    away_team_id: float,
    as_of_date: pd.Timestamp,
    home_starting: list[str],
    home_subs: list[str],
    away_starting: list[str],
    away_subs: list[str],
    players: pd.DataFrame,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def add_records(team_id: float, player_names: list[str], lineup_type: str) -> None:
        if not player_names:
            placeholder_count = 11 if lineup_type == "starting_lineup" else 5
            for index in range(1, placeholder_count + 1):
                records.append(
                    {
                        "game_id": -1,
                        "date": as_of_date,
                        "player_id": np.nan,
                        "club_id": team_id,
                        "player_name": f"Unknown {lineup_type} {index}",
                        "type": lineup_type,
                        "position": np.nan,
                        "number": index,
                        "team_captain": 1 if index == 1 and lineup_type == "starting_lineup" else 0,
                    }
                )
            return
        for index, player_name in enumerate(player_names, start=1):
            player_row = resolve_player_row(players, player_name)
            records.append(
                {
                    "game_id": -1,
                    "date": as_of_date,
                    "player_id": float(player_row["player_id"]) if player_row is not None else np.nan,
                    "club_id": team_id,
                    "player_name": player_name,
                    "type": lineup_type,
                    "position": str(player_row["sub_position"]) if player_row is not None and pd.notna(player_row.get("sub_position")) else (str(player_row["position"]) if player_row is not None and pd.notna(player_row.get("position")) else np.nan),
                    "number": index,
                    "team_captain": 1 if index == 1 and lineup_type == "starting_lineup" else 0,
                }
            )

    add_records(home_team_id, home_starting, "starting_lineup")
    add_records(home_team_id, home_subs, "substitutes")
    add_records(away_team_id, away_starting, "starting_lineup")
    add_records(away_team_id, away_subs, "substitutes")
    return pd.DataFrame.from_records(records)


def latest_row_before(frame: pd.DataFrame, key_column: str, key_value: float | str, as_of_date: pd.Timestamp) -> pd.DataFrame:
    if pd.isna(key_value):
        return pd.DataFrame()
    frame = frame.loc[:, ~frame.columns.duplicated()].copy()
    subset = frame[frame[key_column].eq(key_value)].copy()
    if "date" in subset.columns:
        subset = subset[subset["date"].lt(as_of_date)]
    if subset.empty:
        return pd.DataFrame()
    return subset.sort_values("date").tail(1)


def build_inference_row(
    data_dir: Path,
    team_dim: pd.DataFrame,
    team_history: pd.DataFrame,
    manager_history: pd.DataFrame,
    club_event_history: pd.DataFrame,
    players: pd.DataFrame,
    player_valuations: pd.DataFrame,
    transfers: pd.DataFrame,
    game_events: pd.DataFrame,
    games: pd.DataFrame,
    competition_scope: str,
    home_team_name: str,
    away_team_name: str,
    home_manager_name: str | None,
    away_manager_name: str | None,
    home_starting: list[str],
    home_subs: list[str],
    away_starting: list[str],
    away_subs: list[str],
    as_of_date_text: str | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    as_of_date = pd.to_datetime(as_of_date_text, errors="coerce") if as_of_date_text else pd.to_datetime(games["date"].dropna().max()) + pd.Timedelta(days=1)
    if pd.isna(as_of_date):
        as_of_date = pd.to_datetime(games["date"].dropna().max()) + pd.Timedelta(days=1)

    home_team_id = resolve_team_id(team_dim, home_team_name)
    away_team_id = resolve_team_id(team_dim, away_team_name)
    if pd.isna(home_team_id) or pd.isna(away_team_id):
        raise ValueError("Could not resolve one or both team names against the available datasets")

    historical_games = games[games["date"].lt(as_of_date)].copy()
    historical_game_ids = set(historical_games["game_id"].dropna().astype(int).tolist())
    historical_game_events = game_events[to_numeric_id(game_events["game_id"]).isin(historical_game_ids)].copy()
    historical_player_valuations = player_valuations[to_datetime(player_valuations["date"]).lt(as_of_date)].copy()
    historical_transfers = transfers[to_datetime(transfers["transfer_date"]).lt(as_of_date)].copy()

    custom_lineups = build_custom_lineups(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        as_of_date=as_of_date,
        home_starting=home_starting,
        home_subs=home_subs,
        away_starting=away_starting,
        away_subs=away_subs,
        players=players,
    )

    lineup_player_features = build_lineup_player_features(
        custom_lineups,
        historical_games,
        players,
        historical_game_events,
        historical_player_valuations,
        historical_transfers,
    )
    lineup_summary = aggregate_lineup_features(lineup_player_features)

    home_history = latest_row_before(team_history, "club_id", home_team_id, as_of_date)
    away_history = latest_row_before(team_history, "club_id", away_team_id, as_of_date)
    if home_history.empty or away_history.empty:
        raise ValueError("Could not find enough historical team data for one of the selected teams")

    if home_manager_name:
        home_manager_history = latest_row_before(manager_history, "manager_name", home_manager_name, as_of_date)
    else:
        home_manager_history = pd.DataFrame()
    if away_manager_name:
        away_manager_history = latest_row_before(manager_history, "manager_name", away_manager_name, as_of_date)
    else:
        away_manager_history = pd.DataFrame()

    if home_manager_history.empty:
        home_manager_history = latest_row_before(manager_history, "manager_name", home_history.iloc[0].get("own_manager_name"), as_of_date)
    if away_manager_history.empty:
        away_manager_history = latest_row_before(manager_history, "manager_name", away_history.iloc[0].get("own_manager_name"), as_of_date)

    home_events = latest_row_before(club_event_history, "club_id", home_team_id, as_of_date)
    away_events = latest_row_before(club_event_history, "club_id", away_team_id, as_of_date)

    home_lineup = lineup_summary[lineup_summary["club_id"].eq(home_team_id)].copy()
    away_lineup = lineup_summary[lineup_summary["club_id"].eq(away_team_id)].copy()
    if home_lineup.empty:
        home_lineup = pd.DataFrame([{column: np.nan for column in lineup_summary.columns}])
        home_lineup["club_id"] = home_team_id
        home_lineup["game_id"] = -1
    if away_lineup.empty:
        away_lineup = pd.DataFrame([{column: np.nan for column in lineup_summary.columns}])
        away_lineup["club_id"] = away_team_id
        away_lineup["game_id"] = -1

    base = pd.DataFrame(
        [
            {
                "game_id": -1,
                "date": as_of_date,
                "competition_type": competition_scope,
                "round": "prediction",
                "round_group": "prediction",
                "home_club_id": home_team_id,
                "away_club_id": away_team_id,
                "home_club_goals": np.nan,
                "away_club_goals": np.nan,
                "label": np.nan,
            }
        ]
    )

    def prefixed_single_row(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame([{}])
        frame = frame.loc[:, ~frame.columns.duplicated()].copy()
        return frame.tail(1).reset_index(drop=True).add_prefix(prefix)

    home_profile = prefixed_single_row(home_history, "home_")
    away_profile = prefixed_single_row(away_history, "away_")
    home_manager_profile = prefixed_single_row(home_manager_history, "home_")
    away_manager_profile = prefixed_single_row(away_manager_history, "away_")
    home_event_profile = prefixed_single_row(home_events, "home_")
    away_event_profile = prefixed_single_row(away_events, "away_")
    home_lineup_profile = prefixed_single_row(home_lineup, "home_")
    away_lineup_profile = prefixed_single_row(away_lineup, "away_")

    home_lookup = team_dim[team_dim["team_id"].eq(home_team_id)].tail(1).reset_index(drop=True).add_prefix("home_")
    away_lookup = team_dim[team_dim["team_id"].eq(away_team_id)].tail(1).reset_index(drop=True).add_prefix("away_")

    inference_row = base
    for part in [home_profile, away_profile, home_manager_profile, away_manager_profile, home_event_profile, away_event_profile, home_lineup_profile, away_lineup_profile, home_lookup, away_lookup]:
        inference_row = pd.concat([inference_row.reset_index(drop=True), part.reset_index(drop=True)], axis=1)

    inference_row = inference_row.loc[:, ~inference_row.columns.duplicated()].copy()

    return inference_row, {
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "as_of_date": str(pd.Timestamp(as_of_date).date()),
        "home_manager_name": home_manager_name,
        "away_manager_name": away_manager_name,
    }


def run_inference(
    model: Pipeline,
    feature_columns: list[str],
    data_dir: Path,
    team_dim: pd.DataFrame,
    team_history: pd.DataFrame,
    manager_history: pd.DataFrame,
    club_event_history: pd.DataFrame,
    players: pd.DataFrame,
    player_valuations: pd.DataFrame,
    transfers: pd.DataFrame,
    game_events: pd.DataFrame,
    games: pd.DataFrame,
    competition_scope: str,
    home_team_name: str,
    away_team_name: str,
    home_manager_name: str | None,
    away_manager_name: str | None,
    home_starting: list[str],
    home_subs: list[str],
    away_starting: list[str],
    away_subs: list[str],
    as_of_date_text: str | None,
    output_dir: Path,
) -> dict[str, object]:
    inference_row, metadata = build_inference_row(
        data_dir=data_dir,
        team_dim=team_dim,
        team_history=team_history,
        manager_history=manager_history,
        club_event_history=club_event_history,
        players=players,
        player_valuations=player_valuations,
        transfers=transfers,
        game_events=game_events,
        games=games,
        competition_scope=competition_scope,
        home_team_name=home_team_name,
        away_team_name=away_team_name,
        home_manager_name=home_manager_name,
        away_manager_name=away_manager_name,
        home_starting=home_starting,
        home_subs=home_subs,
        away_starting=away_starting,
        away_subs=away_subs,
        as_of_date_text=as_of_date_text,
    )

    inference_row = inference_row.loc[:, ~inference_row.columns.duplicated()].copy()
    inference_features = inference_row.reindex(columns=feature_columns, fill_value=np.nan)
    probabilities = model.predict_proba(inference_features)[0]
    result = {
        **metadata,
        "prob_loss": float(probabilities[0]),
        "prob_draw": float(probabilities[1]),
        "prob_win": float(probabilities[2]),
        "predicted_class": RESULT_LABELS[int(np.argmax(probabilities))],
    }
    with open(output_dir / "inference_prediction.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    return result


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading datasets...")
    games = load_csv(args.data_dir, "games.csv")
    club_games = load_csv(args.data_dir, "club_games.csv")
    clubs = load_csv(args.data_dir, "clubs.csv")
    competitions = load_csv(args.data_dir, "competitions.csv")
    countries = load_csv(args.data_dir, "countries.csv")
    national_teams = load_csv(args.data_dir, "national_teams.csv")
    players = load_csv(args.data_dir, "players.csv")
    player_valuations = load_csv(args.data_dir, "player_valuations.csv")
    transfers = load_csv(args.data_dir, "transfers.csv")
    game_lineups = load_csv(args.data_dir, "game_lineups.csv")
    game_events = load_csv(args.data_dir, "game_events.csv")

    games["game_id"] = to_numeric_id(games["game_id"])
    games["date"] = to_datetime(games["date"])
    games["home_club_id"] = to_numeric_id(games["home_club_id"])
    games["away_club_id"] = to_numeric_id(games["away_club_id"])
    games["competition_type"] = games["competition_type"].fillna("unknown")
    games["round"] = games["round"].fillna("unknown")

    if args.competition_scope != "all":
        games = games[games["competition_type"] == args.competition_scope].copy()

    valid_game_ids = set(games["game_id"].dropna().astype(int).tolist())
    club_games = club_games[to_numeric_id(club_games["game_id"]).isin(valid_game_ids)].copy()
    game_lineups = game_lineups[to_numeric_id(game_lineups["game_id"]).isin(valid_game_ids)].copy()
    game_events = game_events[to_numeric_id(game_events["game_id"]).isin(valid_game_ids)].copy()

    print("Building team and manager histories...")
    team_dim = normalize_team_dim(clubs, national_teams, competitions, countries)
    team_history = build_team_history(club_games, games)
    manager_history = build_manager_history(team_history)
    club_event_history = build_club_event_history(game_events, games)

    print("Building lineup and player profiles...")
    lineup_player_features = build_lineup_player_features(game_lineups, games, players, game_events, player_valuations, transfers)
    lineup_summary = aggregate_lineup_features(lineup_player_features)

    print("Building match-level feature table...")
    team_profile = build_team_profile(games, team_history, manager_history, club_event_history, lineup_summary, team_dim)
    pair = build_diff_features(team_profile)

    print("Splitting data by time and training RandomForest...")
    features, y, numeric_columns, categorical_columns = build_feature_matrix(pair)
    x_train, x_test, y_train, y_test, train_dates, test_dates = time_based_split(
        features, y, pair["date"], args.test_fraction, args.min_test_games
    )

    model = build_model(numeric_columns, categorical_columns, args.n_estimators, args.random_state)
    model.fit(x_train, y_train)

    print("Evaluating model and writing predictions...")
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_proba, labels=[0, 1, 2])),
        "roc_auc_ovr_macro": float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "train_last_date": str(pd.Series(train_dates).dropna().max().date()),
        "test_first_date": str(pd.Series(test_dates).dropna().min().date()),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=[RESULT_LABELS[index] for index in [0, 1, 2]],
            output_dict=True,
            zero_division=0,
        ),
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)

    prediction_frame = x_test.copy()
    prediction_frame["true_label"] = y_test.values
    prediction_frame["pred_label"] = y_pred
    prediction_frame["prob_loss"] = y_proba[:, 0]
    prediction_frame["prob_draw"] = y_proba[:, 1]
    prediction_frame["prob_win"] = y_proba[:, 2]
    prediction_frame.to_csv(args.output_dir / "test_predictions.csv", index=False)

    save_curves(args.output_dir, y_test, y_proba)
    importance = save_feature_importance(args.output_dir, model)

    try:
        import joblib

        joblib.dump(model, args.output_dir / "random_forest_model.joblib")
    except Exception:
        pass

    prediction_request = None
    if args.predict_home_team and args.predict_away_team:
        prediction_request = {
            "home_team_name": args.predict_home_team,
            "away_team_name": args.predict_away_team,
            "home_manager_name": args.predict_home_manager,
            "away_manager_name": args.predict_away_manager,
            "home_starting": parse_name_list(args.predict_home_starting),
            "home_subs": parse_name_list(args.predict_home_subs),
            "away_starting": parse_name_list(args.predict_away_starting),
            "away_subs": parse_name_list(args.predict_away_subs),
            "as_of_date_text": args.predict_as_of_date,
        }
    elif sys.stdin.isatty():
        prediction_request = prompt_interactive_inference()

    if prediction_request:
        inference_result = run_inference(
            model=model,
            feature_columns=list(x_train.columns),
            data_dir=args.data_dir,
            team_dim=team_dim,
            team_history=team_history,
            manager_history=manager_history,
            club_event_history=club_event_history,
            players=players,
            player_valuations=player_valuations,
            transfers=transfers,
            game_events=game_events,
            games=games,
            competition_scope=args.competition_scope,
            home_team_name=prediction_request["home_team_name"],
            away_team_name=prediction_request["away_team_name"],
            home_manager_name=prediction_request["home_manager_name"],
            away_manager_name=prediction_request["away_manager_name"],
            home_starting=prediction_request["home_starting"],
            home_subs=prediction_request["home_subs"],
            away_starting=prediction_request["away_starting"],
            away_subs=prediction_request["away_subs"],
            as_of_date_text=prediction_request["as_of_date_text"],
            output_dir=args.output_dir,
        )
        print(json.dumps(inference_result, indent=2, ensure_ascii=False))

    print("Done.")
    print("Training complete")
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print(f"Saved {len(importance)} feature importances to {args.output_dir}")


if __name__ == "__main__":
    main()
