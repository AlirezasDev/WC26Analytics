"""
Data loading, cleaning, normalization, and entity resolution module.
"""
from pathlib import Path
import unicodedata
import numpy as np
import pandas as pd

from config import (
    REQUIRED_CSV_FILES,
    TEAM_NAME_ALIASES,
    KNOWN_NATIONAL_TEAM_IDS,
    ROLE_MAP,
    LABEL_TO_INT,
)


def normalize_name(value: object) -> str:
    """Normalize names removing diacritics, accents, hyphens, and multiple spaces."""
    if pd.isna(value) or value is None:
        return ""
    text = str(value).strip().lower()
    # Normalize unicode (e.g., Curaçao -> Curacao, Müller -> Muller, Özil -> Ozil)
    nfkd = unicodedata.normalize("NFKD", text)
    cleaned = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = cleaned.replace("-", " ").replace("_", " ").replace("'", "").replace("’", "")
    return " ".join(cleaned.split())


def alias_team_name(team_name: str) -> str:
    """Map informal team aliases to canonical names."""
    norm = normalize_name(team_name)
    return TEAM_NAME_ALIASES.get(norm, norm)


def to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def to_numeric_id(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def load_csv(data_dir: Path, name: str) -> pd.DataFrame:
    path = data_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing required CSV file: {path}")
    return pd.read_csv(path, low_memory=False)


def validate_required_csv_files(data_dir: Path) -> None:
    missing = [name for name in REQUIRED_CSV_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files in {data_dir}: {', '.join(missing)}")


def normalize_team_dim(
    clubs: pd.DataFrame,
    national_teams: pd.DataFrame,
    competitions: pd.DataFrame,
    countries: pd.DataFrame,
    games: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build a comprehensive, unified team dimension table covering all clubs,
    all official national teams, and any additional teams appearing in games.csv.
    """
    country_lookup = countries[["country_id", "country_code", "country_name", "confederation"]].copy()
    country_lookup["country_id"] = to_numeric_id(country_lookup["country_id"])

    # 1. Clubs
    club_dim = clubs.copy()
    club_dim["club_id"] = to_numeric_id(club_dim["club_id"])
    club_dim = club_dim.merge(
        competitions[["competition_id", "country_id", "country_name", "confederation"]],
        left_on="domestic_competition_id",
        right_on="competition_id",
        how="left",
    )
    club_dim = club_dim.merge(country_lookup[["country_id", "country_code"]], on="country_id", how="left")
    club_dim = club_dim.rename(
        columns={
            "club_id": "team_id",
            "name": "team_name",
            "country_name": "team_country_name",
            "confederation": "team_confederation",
        }
    )
    club_dim["team_kind"] = "club"

    # 2. National Teams
    nat_dim = national_teams.copy()
    nat_dim["national_team_id"] = to_numeric_id(nat_dim["national_team_id"])
    nat_dim = nat_dim.merge(country_lookup[["country_id", "country_code"]], on="country_id", how="left")
    nat_dim = nat_dim.rename(
        columns={
            "national_team_id": "team_id",
            "name": "team_name",
            "country_name": "team_country_name",
            "confederation": "team_confederation",
        }
    )
    nat_dim["team_kind"] = "national_team"

    keep_columns = ["team_id", "team_name", "team_kind", "team_confederation", "team_country_name", "country_code"]
    club_dim = club_dim.reindex(columns=keep_columns)
    nat_dim = nat_dim.reindex(columns=keep_columns)
    team_dim = pd.concat([club_dim, nat_dim], ignore_index=True)

    # 3. Add any missing teams from games.csv (e.g. Cape Verde, Curacao, DR Congo, Haiti, Ivory Coast)
    if games is not None:
        home_g = games[["home_club_id", "home_club_name"]].dropna().rename(columns={"home_club_id": "team_id", "home_club_name": "team_name"})
        away_g = games[["away_club_id", "away_club_name"]].dropna().rename(columns={"away_club_id": "team_id", "away_club_name": "team_name"})
        games_teams = pd.concat([home_g, away_g], ignore_index=True).drop_duplicates(subset=["team_id"])
        games_teams["team_id"] = to_numeric_id(games_teams["team_id"])

        missing_teams = games_teams[~games_teams["team_id"].isin(team_dim["team_id"])].copy()
        if not missing_teams.empty:
            missing_teams["team_kind"] = "national_team"
            missing_teams["team_country_name"] = missing_teams["team_name"]
            missing_teams["team_confederation"] = "unknown"
            missing_teams["country_code"] = "unknown"
            team_dim = pd.concat([team_dim, missing_teams.reindex(columns=keep_columns)], ignore_index=True)

    # 4. Fill in known fallback IDs for national teams
    for nat_name, nat_id in KNOWN_NATIONAL_TEAM_IDS.items():
        if not team_dim["team_id"].eq(nat_id).any():
            new_row = pd.DataFrame([{
                "team_id": nat_id,
                "team_name": nat_name.title(),
                "team_kind": "national_team",
                "team_confederation": "unknown",
                "team_country_name": nat_name.title(),
                "country_code": "unknown",
            }])
            team_dim = pd.concat([team_dim, new_row], ignore_index=True)

    return team_dim.drop_duplicates(subset=["team_id"]).reset_index(drop=True)


def resolve_team_id(team_dim: pd.DataFrame, team_name: str) -> float:
    """Resolve a team name to its canonical team_id, using alias mapping and fallbacks."""
    if not team_name:
        return float("nan")
    norm = alias_team_name(team_name)
    
    # 1. Match against normalized team_name
    dim_norm = team_dim["team_name"].fillna("").map(normalize_name)
    matches = team_dim[dim_norm.eq(norm)]
    if not matches.empty:
        return float(matches.iloc[0]["team_id"])
    
    # 2. Match against team_country_name
    country_norm = team_dim["team_country_name"].fillna("").map(normalize_name)
    matches_c = team_dim[country_norm.eq(norm)]
    if not matches_c.empty:
        return float(matches_c.iloc[0]["team_id"])
    
    # 3. Check hardcoded known national team IDs
    if norm in KNOWN_NATIONAL_TEAM_IDS:
        return float(KNOWN_NATIONAL_TEAM_IDS[norm])
        
    return float("nan")


def resolve_player_row(players: pd.DataFrame, player_name: str, team_country_name: str | None = None) -> pd.Series | None:
    """
    Resolve a player name to their exact record in players.csv.
    Uses unicode normalization, full name matching, and smart disambiguation
    (nationality match, active season, and market value) to prevent homonym confusion.
    """
    if not player_name or pd.isna(player_name):
        return None
        
    norm = normalize_name(player_name)
    if "norm_name" not in players.columns:
        players["norm_name"] = players["name"].fillna("").map(normalize_name)
        players["norm_full_name"] = (
            players["first_name"].fillna("").str.cat(players["last_name"].fillna(""), sep=" ").str.strip().map(normalize_name)
        )
        
    cands = players[players["norm_name"].eq(norm)]
    if cands.empty:
        cands = players[players["norm_full_name"].eq(norm)]
    if cands.empty:
        parts = norm.split()
        if len(parts) >= 2:
            cands = players[
                players["norm_name"].str.contains(parts[-1], na=False) &
                players["norm_name"].str.contains(parts[0], na=False)
            ]
            
    if cands.empty:
        return None
    if len(cands) == 1:
        return cands.iloc[0]
        
    # Disambiguation for multiple homonyms
    cands = cands.copy()
    cands["team_match"] = 0
    if team_country_name:
        t_norm = alias_team_name(team_country_name)
        cands["team_match"] = cands["country_of_citizenship"].fillna("").map(normalize_name).eq(t_norm).astype(int)
        
    cands = cands.sort_values(
        ["team_match", "last_season", "market_value_in_eur", "international_caps"],
        ascending=[False, False, False, False],
    )
    return cands.iloc[0]


def rectify_knockout_scores(games: pd.DataFrame, game_events: pd.DataFrame) -> pd.DataFrame:
    """
    Rectify game scores in games.csv.
    Transfermarkt records penalty shootout goals as match goals in knockout matches.
    In standard betting markets (1X2 match odds), any match resolved by penalties is
    a DRAW (1) in 90-minute regulation time.
    """
    g = games.copy()
    g["game_id"] = to_numeric_id(g["game_id"])
    g["home_club_goals"] = to_numeric_id(g["home_club_goals"])
    g["away_club_goals"] = to_numeric_id(g["away_club_goals"])
    g["date"] = to_datetime(g["date"])
    
    shootout_events = game_events[game_events["type"] == "Shootout"]
    shootout_game_ids = set(to_numeric_id(shootout_events["game_id"]).dropna().astype(int))
    
    # Calculate regulation label
    # Win = 2, Draw = 1, Loss = 0
    g["is_shootout"] = g["game_id"].isin(shootout_game_ids).astype(int)
    
    conditions = [
        g["is_shootout"].eq(1),
        g["home_club_goals"] > g["away_club_goals"],
        g["home_club_goals"] == g["away_club_goals"],
    ]
    choices = [
        LABEL_TO_INT["draw"],
        LABEL_TO_INT["win"],
        LABEL_TO_INT["draw"],
    ]
    g["label"] = np.select(conditions, choices, default=LABEL_TO_INT["loss"])
    
    return g
