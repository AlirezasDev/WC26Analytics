"""
Dynamic Elo rating system and Dixon-Coles / Poisson expected goals and probabilities engine.
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from data_loader import to_datetime, to_numeric_id


def get_k_factor(competition_type: object, competition_id: object) -> float:
    """Assign match importance weight (K-factor) based on competition hierarchy."""
    c_id = str(competition_id)
    c_type = str(competition_type)
    if c_id == "FIWC":
        return 60.0
    if c_type == "national_team_competition":
        return 50.0
    if c_type == "international_cup":
        return 40.0
    return 30.0


def calculate_dixon_coles_probs(lambda_h: float, lambda_a: float, rho: float = -0.05, max_goals: int = 8) -> tuple[float, float, float]:
    """
    Calculate pure Win, Draw, Loss probabilities using a bivariate Poisson distribution
    with the Dixon-Coles correction factor for low-scoring matches (0-0, 1-0, 0-1, 1-1).
    """
    lambda_h = max(0.05, float(lambda_h))
    lambda_a = max(0.05, float(lambda_a))
    
    goals = np.arange(max_goals + 1)
    p_h = poisson.pmf(goals, lambda_h)
    p_a = poisson.pmf(goals, lambda_a)
    
    prob_matrix = np.outer(p_h, p_a)
    
    # Dixon-Coles adjustment for low scores
    prob_matrix[0, 0] *= max(0.0, 1.0 - lambda_h * lambda_a * rho)
    prob_matrix[0, 1] *= max(0.0, 1.0 + lambda_h * rho)
    prob_matrix[1, 0] *= max(0.0, 1.0 + lambda_a * rho)
    prob_matrix[1, 1] *= max(0.0, 1.0 - rho)
    
    # Normalize probabilities to sum to 1.0
    total = prob_matrix.sum()
    if total > 0:
        prob_matrix /= total
        
    p_win = float(np.sum(np.tril(prob_matrix, -1)))
    p_draw = float(np.sum(np.diag(prob_matrix)))
    p_loss = float(np.sum(np.triu(prob_matrix, 1)))
    
    return p_win, p_draw, p_loss


def calculate_dynamic_elo(games: pd.DataFrame, national_teams: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[float, float]]:
    """
    Calculate dynamic Elo ratings chronologically across all 88k+ matches.
    Pre-seeds national teams with FIFA ranking priors.
    Returns games DataFrame with added columns:
    [home_elo_before, away_elo_before, diff_elo] and the latest Elo rating per team.
    """
    g = games.copy()
    g["date"] = to_datetime(g["date"])
    g["home_club_id"] = to_numeric_id(g["home_club_id"])
    g["away_club_id"] = to_numeric_id(g["away_club_id"])
    g["home_club_goals"] = to_numeric_id(g["home_club_goals"])
    g["away_club_goals"] = to_numeric_id(g["away_club_goals"])
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)

    # Initialize Elo ratings
    elo: dict[float, float] = {}
    
    # Pre-seed national teams if national_teams metadata is present
    if national_teams is not None:
        for _, row in national_teams.iterrows():
            tid = float(row.get("national_team_id") or 0)
            if tid <= 0:
                continue
            fifa_rank = float(row.get("fifa_ranking") or 80)
            # Map FIFA rank 1 -> ~1950, rank 50 -> ~1600, rank 150 -> ~1250
            seed_elo = 2000.0 - 5.0 * fifa_rank
            elo[tid] = max(1100.0, min(2100.0, seed_elo))

    n = len(g)
    home_elos = np.zeros(n, dtype=float)
    away_elos = np.zeros(n, dtype=float)

    h_ids = g["home_club_id"].values
    a_ids = g["away_club_id"].values
    h_goals = g["home_club_goals"].values
    a_goals = g["away_club_goals"].values
    c_types = g["competition_type"].values
    c_ids = g["competition_id"].values

    for i in range(n):
        h = h_ids[i]
        a = a_ids[i]
        hg = h_goals[i]
        ag = a_goals[i]

        h_elo = elo.get(h, 1500.0)
        a_elo = elo.get(a, 1500.0)

        home_elos[i] = h_elo
        away_elos[i] = a_elo

        if pd.isna(hg) or pd.isna(ag):
            continue

        # Neutral site detection for national competitions
        is_neutral = c_types[i] == "national_team_competition" and c_ids[i] == "FIWC"
        home_adv = 0.0 if is_neutral else 65.0
        diff = h_elo - a_elo + home_adv

        e_home = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
        e_away = 1.0 - e_home

        if hg > ag:
            s_home, s_away = 1.0, 0.0
        elif hg == ag:
            s_home, s_away = 0.5, 0.5
        else:
            s_home, s_away = 0.0, 1.0

        gd = abs(int(hg) - int(ag))
        if gd <= 1:
            g_mult = 1.0
        elif gd == 2:
            g_mult = 1.5
        else:
            g_mult = (11.0 + gd) / 8.0

        k = get_k_factor(c_types[i], c_ids[i])
        elo[h] = h_elo + k * g_mult * (s_home - e_home)
        elo[a] = a_elo + k * g_mult * (s_away - e_away)

    g["home_elo_before"] = home_elos
    g["away_elo_before"] = away_elos
    g["diff_elo"] = home_elos - away_elos

    return g, elo


def compute_poisson_features(games: pd.DataFrame) -> pd.DataFrame:
    """
    Compute expected goals (lambda_home, lambda_away) and Dixon-Coles probabilities
    from Elo difference and rolling goal averages.
    """
    df = games.copy()
    diff_elo = df["diff_elo"].fillna(0).values
    
    # Expected goals formula based on Elo spread
    # Baseline expected goals: 1.4 home, 1.15 away
    # Elo difference of +200 shifts goals roughly +0.5 for favorite, -0.3 for underdog
    lambda_home = np.clip(1.35 + (diff_elo / 500.0), 0.2, 4.5)
    lambda_away = np.clip(1.15 - (diff_elo / 500.0), 0.2, 4.5)
    
    n = len(df)
    prob_win = np.zeros(n, dtype=float)
    prob_draw = np.zeros(n, dtype=float)
    prob_loss = np.zeros(n, dtype=float)
    
    for i in range(n):
        pw, pd_val, pl = calculate_dixon_coles_probs(lambda_home[i], lambda_away[i])
        prob_win[i] = pw
        prob_draw[i] = pd_val
        prob_loss[i] = pl
        
    df["expected_home_goals"] = lambda_home
    df["expected_away_goals"] = lambda_away
    df["expected_goal_diff"] = lambda_home - lambda_away
    df["expected_total_goals"] = lambda_home + lambda_away
    df["poisson_prob_win"] = prob_win
    df["poisson_prob_draw"] = prob_draw
    df["poisson_prob_loss"] = prob_loss
    
    return df
