"""
Player profiling, rolling performance aggregation, and squad-level feature extraction.
"""
import numpy as np
import pandas as pd

from config import ROLE_MAP
from data_loader import to_datetime, to_numeric_id


def role_from_position(position: object) -> str:
    if pd.isna(position) or position is None:
        return "unknown"
    return ROLE_MAP.get(str(position), "unknown")


def build_player_appearance_history(appearances: pd.DataFrame) -> pd.DataFrame:
    """
    Build cumulative and rolling performance statistics for every player
    chronologically across all matches in appearances.csv (1.89M rows).
    """
    app = appearances.copy()
    app["date"] = to_datetime(app["date"])
    app["player_id"] = to_numeric_id(app["player_id"]).fillna(0).astype("int64")
    app["minutes_played"] = to_numeric_id(app["minutes_played"]).fillna(0)
    app["goals"] = to_numeric_id(app["goals"]).fillna(0)
    app["assists"] = to_numeric_id(app["assists"]).fillna(0)
    app["yellow_cards"] = to_numeric_id(app["yellow_cards"]).fillna(0)
    app["red_cards"] = to_numeric_id(app["red_cards"]).fillna(0)

    app = app.sort_values(["player_id", "date"]).reset_index(drop=True)
    g = app.groupby("player_id")

    app["player_appearances_before"] = g.cumcount()
    app["player_days_since_last_match"] = g["date"].diff().dt.days

    shifted_goals = g["goals"].shift(1).fillna(0)
    shifted_assists = g["assists"].shift(1).fillna(0)
    shifted_mins = g["minutes_played"].shift(1).fillna(0)
    shifted_yellows = g["yellow_cards"].shift(1).fillna(0)
    shifted_reds = g["red_cards"].shift(1).fillna(0)

    app["player_goals_before"] = shifted_goals.groupby(app["player_id"]).cumsum()
    app["player_assists_before"] = shifted_assists.groupby(app["player_id"]).cumsum()
    app["player_minutes_before"] = shifted_mins.groupby(app["player_id"]).cumsum()
    app["player_yellow_cards_before"] = shifted_yellows.groupby(app["player_id"]).cumsum()
    app["player_red_cards_before"] = shifted_reds.groupby(app["player_id"]).cumsum()

    app["player_goals_last_5"] = g["goals"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    app["player_assists_last_5"] = g["assists"].shift(1).rolling(5, min_periods=1).mean().fillna(0)
    app["player_minutes_last_5"] = g["minutes_played"].shift(1).rolling(5, min_periods=1).mean().fillna(0)

    cols = [
        "player_id",
        "date",
        "player_appearances_before",
        "player_days_since_last_match",
        "player_goals_before",
        "player_assists_before",
        "player_minutes_before",
        "player_yellow_cards_before",
        "player_red_cards_before",
        "player_goals_last_5",
        "player_assists_last_5",
        "player_minutes_last_5",
    ]
    return app[cols].sort_values("date").reset_index(drop=True)


def build_valuation_history(player_valuations: pd.DataFrame) -> pd.DataFrame:
    """Prepare clean, date-sorted market valuations for backward merge_asof."""
    pv = player_valuations.copy()
    pv["player_id"] = to_numeric_id(pv["player_id"]).dropna().astype("int64")
    pv["date"] = to_datetime(pv["date"])
    pv["market_value_before"] = to_numeric_id(pv["market_value_in_eur"]).fillna(0)
    pv = pv.sort_values(["player_id", "date"]).reset_index(drop=True)

    g = pv.groupby("player_id")
    pv["valuation_count_before"] = g.cumcount()
    pv["days_since_last_valuation"] = g["date"].diff().dt.days

    cols = ["player_id", "date", "market_value_before", "valuation_count_before", "days_since_last_valuation"]
    return pv[cols].sort_values("date").reset_index(drop=True)


def build_lineup_features(
    game_lineups: pd.DataFrame,
    players: pd.DataFrame,
    player_history: pd.DataFrame,
    valuation_history: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge player lineups with player static attributes, appearance history,
    and historical market valuations as of the match date.
    """
    gl = game_lineups.copy()
    gl["game_id"] = to_numeric_id(gl["game_id"])
    gl["club_id"] = to_numeric_id(gl["club_id"])
    gl["player_id"] = to_numeric_id(gl["player_id"]).fillna(0).astype("int64")
    gl["date"] = to_datetime(gl["date"])
    gl["is_starting"] = gl["type"].eq("starting_lineup").astype(int)
    gl["is_substitute"] = gl["type"].eq("substitutes").astype(int)
    gl["is_captain"] = to_numeric_id(gl["team_captain"]).fillna(0).astype(int)
    gl["role"] = gl["position"].apply(role_from_position)

    # Base player attributes
    p_base = players[[
        "player_id",
        "date_of_birth",
        "height_in_cm",
        "market_value_in_eur",
        "highest_market_value_in_eur",
        "international_caps",
        "international_goals",
        "foot",
    ]].copy()
    p_base["player_id"] = to_numeric_id(p_base["player_id"]).fillna(0).astype("int64")
    p_base["date_of_birth"] = to_datetime(p_base["date_of_birth"])
    p_base["height_in_cm"] = to_numeric_id(p_base["height_in_cm"])
    p_base["market_value_static"] = to_numeric_id(p_base["market_value_in_eur"]).fillna(0)
    p_base["highest_market_value_in_eur"] = to_numeric_id(p_base["highest_market_value_in_eur"]).fillna(0)
    p_base["international_caps"] = to_numeric_id(p_base["international_caps"]).fillna(0)
    p_base["international_goals"] = to_numeric_id(p_base["international_goals"]).fillna(0)

    gl = gl.merge(p_base, on="player_id", how="left")
    gl["age_years"] = (gl["date"] - gl["date_of_birth"]).dt.days / 365.25

    # Backward as-of merge for historical valuations
    gl = gl.sort_values("date").reset_index(drop=True)
    gl = pd.merge_asof(
        gl,
        valuation_history,
        by="player_id",
        on="date",
        direction="backward",
        allow_exact_matches=False,
    )
    # Fallback to static market value if valuation prior to match date is missing
    gl["market_value_before"] = gl["market_value_before"].fillna(gl["market_value_static"]).fillna(0)

    # Backward as-of merge for historical appearance stats
    gl = pd.merge_asof(
        gl,
        player_history,
        by="player_id",
        on="date",
        direction="backward",
        allow_exact_matches=False,
    )
    for col in [
        "player_appearances_before",
        "player_goals_before",
        "player_assists_before",
        "player_minutes_before",
        "player_goals_last_5",
        "player_assists_last_5",
        "player_minutes_last_5",
    ]:
        gl[col] = gl[col].fillna(0)

    return gl


def aggregate_squad_features(lineup_features: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate individual lineup features into comprehensive team squad metrics,
    strictly distinguishing Starting XI from Substitutes and breaking down by position.
    """
    df = lineup_features.copy()
    
    # Pre-compute positional market values and boolean role flags
    df["val_fw"] = np.where(df["role"] == "forward", df["market_value_before"], 0.0)
    df["val_mid"] = np.where(df["role"] == "midfielder", df["market_value_before"], 0.0)
    df["val_def"] = np.where(df["role"] == "defender", df["market_value_before"], 0.0)
    df["val_gk"] = np.where(df["role"] == "goalkeeper", df["market_value_before"], 0.0)
    df["is_fw"] = (df["role"] == "forward").astype(int)
    df["is_mid"] = (df["role"] == "midfielder").astype(int)
    df["is_def"] = (df["role"] == "defender").astype(int)
    df["is_gk"] = (df["role"] == "goalkeeper").astype(int)
    
    # 1. Starting XI Aggregations
    starters = df[df["is_starting"] == 1].copy()
    starter_agg = starters.groupby(["game_id", "club_id"], as_index=False).agg(
        starter_count=("player_id", "count"),
        starter_market_value_sum=("market_value_before", "sum"),
        starter_market_value_mean=("market_value_before", "mean"),
        starter_market_value_max=("market_value_before", "max"),
        starter_age_mean=("age_years", "mean"),
        starter_age_std=("age_years", "std"),
        starter_height_mean=("height_in_cm", "mean"),
        starter_caps_mean=("international_caps", "mean"),
        starter_intl_goals_mean=("international_goals", "mean"),
        starter_goals_before_mean=("player_goals_before", "mean"),
        starter_assists_before_mean=("player_assists_before", "mean"),
        starter_minutes_before_mean=("player_minutes_before", "mean"),
        starter_goals_last_5_mean=("player_goals_last_5", "mean"),
        starter_assists_last_5_mean=("player_assists_last_5", "mean"),
        starter_minutes_last_5_mean=("player_minutes_last_5", "mean"),
        starter_val_fw_sum=("val_fw", "sum"),
        starter_val_mid_sum=("val_mid", "sum"),
        starter_val_def_sum=("val_def", "sum"),
        starter_val_gk=("val_gk", "sum"),
        starter_fw_count=("is_fw", "sum"),
        starter_mid_count=("is_mid", "sum"),
        starter_def_count=("is_def", "sum"),
        starter_gk_count=("is_gk", "sum"),
    )
    
    # Top 3 player market value average for starters (Star Power, vectorized C-level head(3))
    sorted_starters = starters.sort_values(["game_id", "club_id", "market_value_before"], ascending=[True, True, False])
    top3_starter = (
        sorted_starters.groupby(["game_id", "club_id"])
        .head(3)
        .groupby(["game_id", "club_id"], as_index=False)["market_value_before"]
        .mean()
        .rename(columns={"market_value_before": "starter_top3_val_mean"})
    )
    starter_agg = starter_agg.merge(top3_starter, on=["game_id", "club_id"], how="left")
    
    # 2. Substitutes Aggregations
    subs = df[df["is_substitute"] == 1].copy()
    sub_agg = subs.groupby(["game_id", "club_id"], as_index=False).agg(
        sub_count=("player_id", "count"),
        sub_market_value_sum=("market_value_before", "sum"),
        sub_market_value_mean=("market_value_before", "mean"),
        sub_age_mean=("age_years", "mean"),
        sub_caps_mean=("international_caps", "mean"),
        sub_goals_before_mean=("player_goals_before", "mean"),
        sub_minutes_last_5_mean=("player_minutes_last_5", "mean"),
    )
    
    # 3. Combine and calculate squad ratios
    squad = pd.merge(starter_agg, sub_agg, on=["game_id", "club_id"], how="outer")
    squad["starter_market_value_sum"] = squad["starter_market_value_sum"].fillna(0)
    squad["sub_market_value_sum"] = squad["sub_market_value_sum"].fillna(0)
    squad["squad_market_value_sum"] = squad["starter_market_value_sum"] + squad["sub_market_value_sum"]
    
    squad["starter_value_share"] = (
        squad["starter_market_value_sum"] / squad["squad_market_value_sum"].replace(0, np.nan)
    ).fillna(0.8)
    
    return squad
