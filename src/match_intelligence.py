"""
Deep Match Intelligence, Sportsbook-Grade Market Matrix, H2H, 15-Minute Goal Timing,
Player Goalscorer Projections, and Top-5 European Leagues Analytics Module.

Powered directly by the WC26Analytics historical database (games.csv, clubs.csv,
game_events.csv, players.csv, appearances.csv) and calibrated Bivariate Poisson / ML matrices.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import poisson

from config import DATA_DIR, OUTPUT_DIR, TOP5_EUROPEAN_LEAGUES
from data_loader import to_numeric_id
from models import prob_to_decimal_odds, prob_to_american_odds


def _fmt_market_outcome(prob: float) -> dict[str, Any]:
    p = float(np.clip(prob, 0.002, 0.996))
    return {
        "prob": round(p * 100.0, 2),
        "decimal_odds": prob_to_decimal_odds(p),
        "american_odds": prob_to_american_odds(p),
    }


def compute_full_match_markets(
    score_matrix: np.ndarray,
    lambda_home: float,
    lambda_away: float,
    minute: int = 0,
    current_hg: int = 0,
    current_ag: int = 0,
    expected_corners_home: float | None = None,
    expected_corners_away: float | None = None,
    expected_cards_home: float | None = None,
    expected_cards_away: float | None = None,
) -> dict[str, Any]:
    """
    Compute the complete sportsbook-grade analytical breakdown from a calibrated joint goal matrix:
    - 1X2, Double Chance, Draw No Bet
    - Over/Under Goals (0.5 .. 5.5), Team Totals, BTTS, Clean Sheet, Win to Nil, Odd/Even
    - Result + BTTS & Result + Over/Under 2.5 Combos
    - Asian Handicaps (-2.5 .. +2.5) & 3-Way European Handicaps
    - 1st Half & 2nd Half Markets + HT/FT 9-way Matrix
    - Exact Score Matrix (0-0 .. 5-5) & Top 12 Most Likely Scorelines
    - Corners, Cards & Shots Statistical Markets
    """
    mat = np.asarray(score_matrix, dtype=float)
    total = mat.sum()
    if total > 0:
        mat = mat / total
    n_rows, n_cols = mat.shape
    hg_idx, ag_idx = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    tot_goals = hg_idx + ag_idx
    goal_diff = hg_idx - ag_idx

    # 1. Main 1X2, Double Chance, Draw No Bet
    p_home = float(mat[hg_idx > ag_idx].sum())
    p_draw = float(mat[hg_idx == ag_idx].sum())
    p_away = float(mat[hg_idx < ag_idx].sum())

    non_draw = max(1e-6, p_home + p_away)
    dnb_home = p_home / non_draw
    dnb_away = p_away / non_draw

    # 2. Over / Under Total Goals
    ou_lines = {}
    for line in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5):
        p_over = float(mat[tot_goals > line].sum())
        ou_lines[str(line)] = {
            "over": _fmt_market_outcome(p_over),
            "under": _fmt_market_outcome(1.0 - p_over),
        }

    # 3. Team Total Goals Over / Under
    home_team_ou = {}
    away_team_ou = {}
    for line in (0.5, 1.5, 2.5):
        ph_o = float(mat[hg_idx > line].sum())
        pa_o = float(mat[ag_idx > line].sum())
        home_team_ou[str(line)] = {
            "over": _fmt_market_outcome(ph_o),
            "under": _fmt_market_outcome(1.0 - ph_o),
        }
        away_team_ou[str(line)] = {
            "over": _fmt_market_outcome(pa_o),
            "under": _fmt_market_outcome(1.0 - pa_o),
        }

    # 4. BTTS, Clean Sheet, Win to Nil, Odd/Even
    p_btts_yes = float(mat[(hg_idx >= 1) & (ag_idx >= 1)].sum())
    p_cs_home = float(mat[ag_idx == 0].sum())
    p_cs_away = float(mat[hg_idx == 0].sum())
    p_wtn_home = float(mat[(hg_idx > ag_idx) & (ag_idx == 0)].sum())
    p_wtn_away = float(mat[(ag_idx > hg_idx) & (hg_idx == 0)].sum())
    p_odd = float(mat[(tot_goals % 2) == 1].sum())

    # 5. Combo Markets (Result + BTTS & Result + O/U 2.5)
    combos = {
        "home_and_btts_yes": _fmt_market_outcome(float(mat[(hg_idx > ag_idx) & (ag_idx >= 1)].sum())),
        "home_and_btts_no": _fmt_market_outcome(p_wtn_home),
        "draw_and_btts_yes": _fmt_market_outcome(float(mat[(hg_idx == ag_idx) & (hg_idx >= 1)].sum())),
        "away_and_btts_yes": _fmt_market_outcome(float(mat[(ag_idx > hg_idx) & (hg_idx >= 1)].sum())),
        "away_and_btts_no": _fmt_market_outcome(p_wtn_away),
        "home_and_over_2_5": _fmt_market_outcome(float(mat[(hg_idx > ag_idx) & (tot_goals > 2.5)].sum())),
        "home_and_under_2_5": _fmt_market_outcome(float(mat[(hg_idx > ag_idx) & (tot_goals < 2.5)].sum())),
        "away_and_over_2_5": _fmt_market_outcome(float(mat[(ag_idx > hg_idx) & (tot_goals > 2.5)].sum())),
        "away_and_under_2_5": _fmt_market_outcome(float(mat[(ag_idx > hg_idx) & (tot_goals < 2.5)].sum())),
    }

    # 6. Asian Handicap & European 3-Way Handicap
    asian_handicap = {}
    for hcap in (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5):
        p_h_cov = float(mat[(goal_diff + hcap) > 0].sum())
        label = f"{hcap:+.1f}"
        asian_handicap[label] = {
            "home": _fmt_market_outcome(p_h_cov),
            "away": _fmt_market_outcome(1.0 - p_h_cov),
        }
    for int_hcap in (-1, 0, 1):
        p_win_hc = float(mat[(goal_diff + int_hcap) > 0].sum())
        p_lose_hc = float(mat[(goal_diff + int_hcap) < 0].sum())
        non_push = max(1e-6, p_win_hc + p_lose_hc)
        label = f"{int_hcap:+.1f}"
        asian_handicap[label] = {
            "home": _fmt_market_outcome(p_win_hc / non_push),
            "away": _fmt_market_outcome(p_lose_hc / non_push),
        }

    euro_handicap = {}
    for h_give, a_give in ((0, 1), (0, 2), (1, 0), (2, 0)):
        adj_diff = (hg_idx + h_give) - (ag_idx + a_give)
        euro_handicap[f"{h_give}:{a_give}"] = {
            "home": _fmt_market_outcome(float(mat[adj_diff > 0].sum())),
            "draw": _fmt_market_outcome(float(mat[adj_diff == 0].sum())),
            "away": _fmt_market_outcome(float(mat[adj_diff < 0].sum())),
        }

    # 7. 1st Half & 2nd Half Distributions (44% 1H / 56% 2H empirical European split)
    lh_1h, la_1h = max(0.05, lambda_home * 0.44), max(0.05, lambda_away * 0.44)
    lh_2h, la_2h = max(0.05, lambda_home * 0.56), max(0.05, lambda_away * 0.56)
    g_range = np.arange(6)
    mat_1h = np.outer(poisson.pmf(g_range, lh_1h), poisson.pmf(g_range, la_1h))
    mat_1h /= max(1e-9, mat_1h.sum())
    mat_2h = np.outer(poisson.pmf(g_range, lh_2h), poisson.pmf(g_range, la_2h))
    mat_2h /= max(1e-9, mat_2h.sum())

    i1, j1 = np.meshgrid(g_range, g_range, indexing="ij")
    p1h_h = float(mat_1h[i1 > j1].sum())
    p1h_d = float(mat_1h[i1 == j1].sum())
    p1h_a = float(mat_1h[i1 < j1].sum())

    p2h_h = float(mat_2h[i1 > j1].sum())
    p2h_d = float(mat_2h[i1 == j1].sum())
    p2h_a = float(mat_2h[i1 < j1].sum())

    # HT / FT 9 combinations via conditional transition approximation
    ht_probs = {"1": p1h_h, "X": p1h_d, "2": p1h_a}
    ft_probs = {"1": p_home, "X": p_draw, "2": p_away}
    ht_ft = {}
    raw_htft = {}
    for hk, hp in ht_probs.items():
        for fk, fp in ft_probs.items():
            affinity = 1.45 if hk == fk else (0.95 if hk == "X" else 0.35)
            raw_htft[f"{hk}/{fk}"] = hp * fp * affinity
    htft_sum = max(1e-9, sum(raw_htft.values()))
    for k, val in raw_htft.items():
        ht_ft[k] = _fmt_market_outcome(val / htft_sum)

    # 8. Exact Scorelines (0-0 .. 5-5 grid + Top 12 sorted)
    score_grid = []
    flat_scores = []
    for i in range(min(6, n_rows)):
        row_items = []
        for j in range(min(6, n_cols)):
            pr = float(mat[i, j])
            entry = {
                "score": f"{i}-{j}",
                "home_goals": i,
                "away_goals": j,
                **_fmt_market_outcome(pr),
            }
            row_items.append(entry)
            if i >= current_hg and j >= current_ag:
                flat_scores.append(entry)
        score_grid.append(row_items)
    flat_scores.sort(key=lambda x: x["prob"], reverse=True)

    # 9. Corners, Cards & Shots Projections
    rem_ratio = 1.0 if minute <= 0 else max(0.08, (92.0 - minute) / 92.0)
    ec_h = (expected_corners_home if expected_corners_home is not None else (3.6 + 1.35 * lambda_home)) * rem_ratio
    ec_a = (expected_corners_away if expected_corners_away is not None else (3.1 + 1.25 * lambda_away)) * rem_ratio
    ec_tot = ec_h + ec_a

    corners_ou = {}
    for cline in (7.5, 8.5, 9.5, 10.5, 11.5):
        p_co = float(1.0 - poisson.cdf(int(cline), max(0.5, ec_tot)))
        corners_ou[str(cline)] = {
            "over": _fmt_market_outcome(p_co),
            "under": _fmt_market_outcome(1.0 - p_co),
        }

    ey_h = (expected_cards_home if expected_cards_home is not None else 1.95) * rem_ratio
    ey_a = (expected_cards_away if expected_cards_away is not None else 2.15) * rem_ratio
    ey_tot = ey_h + ey_a
    cards_ou = {}
    for yline in (2.5, 3.5, 4.5, 5.5):
        p_yo = float(1.0 - poisson.cdf(int(yline), max(0.3, ey_tot)))
        cards_ou[str(yline)] = {
            "over": _fmt_market_outcome(p_yo),
            "under": _fmt_market_outcome(1.0 - p_yo),
        }

    exp_shots_h = round(6.2 + 5.1 * lambda_home, 1)
    exp_shots_a = round(5.4 + 4.9 * lambda_away, 1)
    exp_sot_h = round(1.8 + 2.15 * lambda_home, 1)
    exp_sot_a = round(1.5 + 2.05 * lambda_away, 1)

    return {
        "main_1x2": {
            "home": _fmt_market_outcome(p_home),
            "draw": _fmt_market_outcome(p_draw),
            "away": _fmt_market_outcome(p_away),
        },
        "double_chance": {
            "1X": _fmt_market_outcome(p_home + p_draw),
            "12": _fmt_market_outcome(p_home + p_away),
            "X2": _fmt_market_outcome(p_draw + p_away),
        },
        "draw_no_bet": {
            "home": _fmt_market_outcome(dnb_home),
            "away": _fmt_market_outcome(dnb_away),
        },
        "goals_over_under": ou_lines,
        "team_goals": {
            "home": home_team_ou,
            "away": away_team_ou,
        },
        "btts": {
            "yes": _fmt_market_outcome(p_btts_yes),
            "no": _fmt_market_outcome(1.0 - p_btts_yes),
        },
        "clean_sheet": {
            "home_yes": _fmt_market_outcome(p_cs_home),
            "home_no": _fmt_market_outcome(1.0 - p_cs_home),
            "away_yes": _fmt_market_outcome(p_cs_away),
            "away_no": _fmt_market_outcome(1.0 - p_cs_away),
        },
        "win_to_nil": {
            "home": _fmt_market_outcome(p_wtn_home),
            "away": _fmt_market_outcome(p_wtn_away),
        },
        "odd_even_goals": {
            "odd": _fmt_market_outcome(p_odd),
            "even": _fmt_market_outcome(1.0 - p_odd),
        },
        "combos": combos,
        "asian_handicap": asian_handicap,
        "european_handicap": euro_handicap,
        "halves": {
            "first_half_1x2": {
                "home": _fmt_market_outcome(p1h_h),
                "draw": _fmt_market_outcome(p1h_d),
                "away": _fmt_market_outcome(p1h_a),
            },
            "first_half_over_under": {
                "0.5": {
                    "over": _fmt_market_outcome(float(mat_1h[(i1 + j1) > 0.5].sum())),
                    "under": _fmt_market_outcome(float(mat_1h[(i1 + j1) < 0.5].sum())),
                },
                "1.5": {
                    "over": _fmt_market_outcome(float(mat_1h[(i1 + j1) > 1.5].sum())),
                    "under": _fmt_market_outcome(float(mat_1h[(i1 + j1) < 1.5].sum())),
                },
            },
            "second_half_1x2": {
                "home": _fmt_market_outcome(p2h_h),
                "draw": _fmt_market_outcome(p2h_d),
                "away": _fmt_market_outcome(p2h_a),
            },
            "ht_ft": ht_ft,
        },
        "exact_scores": {
            "top_scores": flat_scores[:12],
            "grid_6x6": score_grid,
        },
        "corners": {
            "expected_home": round(ec_h, 2),
            "expected_away": round(ec_a, 2),
            "expected_total": round(ec_tot, 2),
            "over_under": corners_ou,
        },
        "cards": {
            "expected_home_yellow": round(ey_h, 2),
            "expected_away_yellow": round(ey_a, 2),
            "expected_total_yellow": round(ey_tot, 2),
            "over_under": cards_ou,
        },
        "shots_projection": {
            "expected_shots_home": exp_shots_h,
            "expected_shots_away": exp_shots_a,
            "expected_sot_home": exp_sot_h,
            "expected_sot_away": exp_sot_a,
        },
    }


class HistoricalIntelligenceStore:
    """
    Fast, lazy-loaded in-memory analytics index over WC26Analytics CSV datasets
    for H2H, form splits, 15-minute goal timing, squad valuations, key players,
    and Top-5 European league hubs.
    """

    def __init__(self, data_dir: Path = DATA_DIR) -> None:
        self.data_dir = data_dir
        self._loaded = False
        self.clubs_df: pd.DataFrame | None = None
        self.games_df: pd.DataFrame | None = None
        self.events_recent_df: pd.DataFrame | None = None
        self.players_df: pd.DataFrame | None = None
        self.appearances_recent_df: pd.DataFrame | None = None

    def ensure_loaded(self) -> None:
        if self._loaded:
            return
        clubs_path = self.data_dir / "clubs.csv"
        games_path = self.data_dir / "games.csv"
        events_path = self.data_dir / "game_events.csv"
        players_path = self.data_dir / "players.csv"
        apps_path = self.data_dir / "appearances.csv"

        if clubs_path.is_file():
            self.clubs_df = pd.read_csv(clubs_path, low_memory=False)
            self.clubs_df["club_id"] = to_numeric_id(self.clubs_df["club_id"])

        if games_path.is_file():
            self.games_df = pd.read_csv(
                games_path,
                usecols=[
                    "game_id", "competition_id", "season", "round", "date",
                    "home_club_id", "away_club_id", "home_club_goals", "away_club_goals",
                    "home_club_manager_name", "away_club_manager_name", "stadium",
                    "attendance", "referee", "home_club_formation", "away_club_formation",
                    "home_club_name", "away_club_name", "competition_type",
                ],
                low_memory=False,
            )
            self.games_df["home_club_id"] = to_numeric_id(self.games_df["home_club_id"])
            self.games_df["away_club_id"] = to_numeric_id(self.games_df["away_club_id"])
            self.games_df["home_club_goals"] = to_numeric_id(self.games_df["home_club_goals"])
            self.games_df["away_club_goals"] = to_numeric_id(self.games_df["away_club_goals"])
            self.games_df = self.games_df.dropna(subset=["home_club_goals", "away_club_goals"]).sort_values("date")

        if players_path.is_file():
            self.players_df = pd.read_csv(
                players_path,
                usecols=[
                    "player_id", "name", "last_season", "current_club_id",
                    "country_of_citizenship", "position", "sub_position",
                    "foot", "height_in_cm", "current_club_domestic_competition_id",
                    "current_club_name", "market_value_in_eur",
                ],
                low_memory=False,
            )
            self.players_df["current_club_id"] = to_numeric_id(self.players_df["current_club_id"])
            self.players_df["market_value_in_eur"] = to_numeric_id(self.players_df["market_value_in_eur"]).fillna(0.0)

        # Load recent events (2022+) for fast 15-min goal & card distributions
        if events_path.is_file():
            ev = pd.read_csv(
                events_path,
                usecols=["date", "game_id", "minute", "type", "club_id", "description"],
                low_memory=False,
            )
            ev = ev[ev["date"].astype(str) >= "2022-07-01"].copy()
            ev["club_id"] = to_numeric_id(ev["club_id"])
            ev["minute"] = to_numeric_id(ev["minute"]).fillna(45)
            self.events_recent_df = ev

        # Load recent appearances (2023+) for top scorers/assisters
        if apps_path.is_file():
            ap = pd.read_csv(
                apps_path,
                usecols=[
                    "player_id", "player_club_id", "date", "player_name",
                    "competition_id", "yellow_cards", "red_cards", "goals", "assists", "minutes_played",
                ],
                low_memory=False,
            )
            ap = ap[ap["date"].astype(str) >= "2023-07-01"].copy()
            ap["player_club_id"] = to_numeric_id(ap["player_club_id"])
            self.appearances_recent_df = ap

        self._loaded = True

    def get_club_profile(self, team_id: float, fallback_name: str = "") -> dict[str, Any]:
        self.ensure_loaded()
        tid = float(team_id)
        squad_mv = 0.0
        if self.players_df is not None:
            active_pl = self.players_df[
                (self.players_df["current_club_id"] == tid) & (self.players_df["last_season"] >= 2024)
            ]
            squad_mv = float(active_pl["market_value_in_eur"].sum())

        if self.clubs_df is not None:
            row_df = self.clubs_df[self.clubs_df["club_id"] == tid]
            if not row_df.empty:
                r = row_df.iloc[0]
                return {
                    "team_id": int(tid),
                    "name": str(r.get("name") or fallback_name),
                    "domestic_league": str(r.get("domestic_competition_id") or ""),
                    "squad_size": int(r.get("squad_size") or 25) if pd.notna(r.get("squad_size")) else 25,
                    "average_age": round(float(r.get("average_age") or 26.0), 1) if pd.notna(r.get("average_age")) else 26.0,
                    "foreigners_pct": round(float(r.get("foreigners_percentage") or 0.0), 1) if pd.notna(r.get("foreigners_percentage")) else 0.0,
                    "national_team_players": int(r.get("national_team_players") or 0) if pd.notna(r.get("national_team_players")) else 0,
                    "stadium_name": str(r.get("stadium_name") or ""),
                    "stadium_seats": int(r.get("stadium_seats") or 0) if pd.notna(r.get("stadium_seats")) else 0,
                    "squad_market_value_eur": round(squad_mv, 0),
                    "squad_market_value_m": round(squad_mv / 1_000_000.0, 1),
                }
        return {
            "team_id": int(tid),
            "name": fallback_name,
            "domestic_league": "",
            "squad_size": 25,
            "average_age": 26.2,
            "foreigners_pct": 0.0,
            "national_team_players": 0,
            "stadium_name": "",
            "stadium_seats": 0,
            "squad_market_value_eur": round(squad_mv, 0),
            "squad_market_value_m": round(squad_mv / 1_000_000.0, 1),
        }

    def get_team_recent_form(self, team_id: float, n: int = 10, venue_filter: str | None = None) -> dict[str, Any]:
        self.ensure_loaded()
        if self.games_df is None:
            return {"matches": [], "form_sequence": [], "ppg": 0.0}
        tid = float(team_id)
        g = self.games_df
        if venue_filter == "home":
            sub = g[g["home_club_id"] == tid].tail(n)
        elif venue_filter == "away":
            sub = g[g["away_club_id"] == tid].tail(n)
        else:
            sub = g[(g["home_club_id"] == tid) | (g["away_club_id"] == tid)].tail(n)

        if sub.empty:
            return {"matches": [], "form_sequence": [], "ppg": 0.0}

        records = []
        seq = []
        pts = 0
        gf_tot, ga_tot, cs, btts_cnt, o25_cnt = 0, 0, 0, 0, 0
        last_formation = ""
        last_manager = ""

        for _, r in sub.iloc[::-1].iterrows():
            is_h = float(r["home_club_id"]) == tid
            hg, ag = int(r["home_club_goals"]), int(r["away_club_goals"])
            gf = hg if is_h else ag
            ga = ag if is_h else hg
            opp = str(r["away_club_name"] if is_h else r["home_club_name"])
            if not last_formation:
                fm = r["home_club_formation"] if is_h else r["away_club_formation"]
                if pd.notna(fm):
                    last_formation = str(fm)
            if not last_manager:
                mg = r["home_club_manager_name"] if is_h else r["away_club_manager_name"]
                if pd.notna(mg):
                    last_manager = str(mg)

            if gf > ga:
                res_char = "W"
                pts += 3
            elif gf == ga:
                res_char = "D"
                pts += 1
            else:
                res_char = "L"

            seq.append(res_char)
            gf_tot += gf
            ga_tot += ga
            if ga == 0:
                cs += 1
            if gf >= 1 and ga >= 1:
                btts_cnt += 1
            if (gf + ga) >= 3:
                o25_cnt += 1

            records.append(
                {
                    "date": str(r["date"])[:10],
                    "competition_id": str(r["competition_id"]),
                    "opponent": opp,
                    "is_home": is_h,
                    "goals_for": gf,
                    "goals_against": ga,
                    "score": f"{hg}-{ag}",
                    "result": res_char,
                }
            )

        cnt = len(records)
        return {
            "matches": records,
            "form_sequence": seq,
            "played": cnt,
            "ppg": round(pts / cnt, 2),
            "avg_gf": round(gf_tot / cnt, 2),
            "avg_ga": round(ga_tot / cnt, 2),
            "clean_sheet_pct": round((cs / cnt) * 100.0, 1),
            "btts_pct": round((btts_cnt / cnt) * 100.0, 1),
            "over_2_5_pct": round((o25_cnt / cnt) * 100.0, 1),
            "recent_formation": last_formation or "4-3-3",
            "recent_manager": last_manager,
        }

    def get_head_to_head(self, home_id: float, away_id: float, limit: int = 10) -> dict[str, Any]:
        self.ensure_loaded()
        if self.games_df is None:
            return {"total_matches": 0, "matches": []}
        hid, aid = float(home_id), float(away_id)
        g = self.games_df
        h2h = g[
            ((g["home_club_id"] == hid) & (g["away_club_id"] == aid))
            | ((g["home_club_id"] == aid) & (g["away_club_id"] == hid))
        ]
        if h2h.empty:
            return {
                "total_matches": 0,
                "home_team_wins": 0,
                "draws": 0,
                "away_team_wins": 0,
                "avg_total_goals": 0.0,
                "btts_pct": 0.0,
                "over_2_5_pct": 0.0,
                "matches": [],
            }

        h_wins, draws, a_wins = 0, 0, 0
        tot_g, btts_c, o25_c = 0, 0, 0
        recent_list = []

        for _, r in h2h.iloc[::-1].iterrows():
            hg, ag = int(r["home_club_goals"]), int(r["away_club_goals"])
            row_hid = float(r["home_club_id"])
            if hg == ag:
                draws += 1
            elif (hg > ag and row_hid == hid) or (ag > hg and row_hid == aid):
                h_wins += 1
            else:
                a_wins += 1

            tot_g += hg + ag
            if hg >= 1 and ag >= 1:
                btts_c += 1
            if (hg + ag) >= 3:
                o25_c += 1

            if len(recent_list) < limit:
                recent_list.append(
                    {
                        "date": str(r["date"])[:10],
                        "competition_id": str(r["competition_id"]),
                        "home_team": str(r["home_club_name"]),
                        "away_team": str(r["away_club_name"]),
                        "home_goals": hg,
                        "away_goals": ag,
                        "score": f"{hg}-{ag}",
                        "stadium": str(r["stadium"]) if pd.notna(r["stadium"]) else "",
                    }
                )

        n = len(h2h)
        return {
            "total_matches": n,
            "home_team_wins": h_wins,
            "draws": draws,
            "away_team_wins": a_wins,
            "avg_total_goals": round(tot_g / n, 2),
            "btts_pct": round((btts_c / n) * 100.0, 1),
            "over_2_5_pct": round((o25_c / n) * 100.0, 1),
            "matches": recent_list,
        }

    def get_goal_timing_and_cards(self, team_id: float) -> dict[str, Any]:
        """Compute 15-minute goal timing distribution and yellow/red card per-game rates from game_events.csv."""
        self.ensure_loaded()
        buckets = ["0-15", "16-30", "31-45+", "46-60", "61-75", "76-90+"]
        default_pcts = [13.5, 15.0, 16.5, 16.5, 17.5, 21.0]
        if self.events_recent_df is None or self.games_df is None:
            return {
                "buckets": [{"interval": b, "pct": p, "count": 0} for b, p in zip(buckets, default_pcts)],
                "avg_yellow_cards": 1.95,
                "late_goal_pct": 21.0,
            }

        tid = float(team_id)
        ev = self.events_recent_df[self.events_recent_df["club_id"] == tid]
        goals = ev[ev["type"] == "Goals"]
        cards = ev[ev["type"] == "Cards"]

        recent_games_cnt = len(
            self.games_df[
                (self.games_df["date"].astype(str) >= "2022-07-01")
                & ((self.games_df["home_club_id"] == tid) | (self.games_df["away_club_id"] == tid))
            ]
        )
        avg_yc = round(len(cards) / max(1, recent_games_cnt), 2) if recent_games_cnt >= 5 else 1.95
        avg_yc = float(np.clip(avg_yc, 1.1, 3.4))

        if len(goals) < 8:
            return {
                "buckets": [{"interval": b, "pct": p, "count": 0} for b, p in zip(buckets, default_pcts)],
                "avg_yellow_cards": avg_yc,
                "late_goal_pct": 21.0,
            }

        mins = goals["minute"].values
        counts = [
            int(np.sum((mins >= 0) & (mins <= 15))),
            int(np.sum((mins >= 16) & (mins <= 30))),
            int(np.sum((mins >= 31) & (mins <= 45))),
            int(np.sum((mins >= 46) & (mins <= 60))),
            int(np.sum((mins >= 61) & (mins <= 75))),
            int(np.sum(mins >= 76)),
        ]
        tot = max(1, sum(counts))
        pcts = [round((c / tot) * 100.0, 1) for c in counts]
        return {
            "buckets": [
                {"interval": b, "pct": p, "count": c}
                for b, p, c in zip(buckets, pcts, counts)
            ],
            "avg_yellow_cards": avg_yc,
            "late_goal_pct": pcts[-1],
        }

    def get_key_players(self, team_id: float, team_expected_goals: float = 1.35, top_n: int = 6) -> list[dict[str, Any]]:
        """Return top scorers, assisters, and highest-valued squad stars with Anytime Goalscorer odds."""
        self.ensure_loaded()
        tid = float(team_id)
        if self.players_df is None:
            return []

        squad = self.players_df[
            (self.players_df["current_club_id"] == tid) & (self.players_df["last_season"] >= 2024)
        ].copy()

        stats_df = pd.DataFrame()
        if self.appearances_recent_df is not None:
            ap = self.appearances_recent_df[self.appearances_recent_df["player_club_id"] == tid]
            if not ap.empty:
                stats_df = (
                    ap.groupby(["player_id", "player_name"], as_index=False)
                    .agg(
                        apps=("minutes_played", "count"),
                        minutes=("minutes_played", "sum"),
                        goals=("goals", "sum"),
                        assists=("assists", "sum"),
                        yellow_cards=("yellow_cards", "sum"),
                    )
                )

        if not squad.empty and not stats_df.empty:
            merged = squad.merge(stats_df.drop(columns=["player_name"]), on="player_id", how="left")
        elif not squad.empty:
            merged = squad.copy()
            for col in ("apps", "minutes", "goals", "assists", "yellow_cards"):
                merged[col] = 0
        elif not stats_df.empty:
            merged = stats_df.rename(columns={"player_name": "name"}).copy()
            merged["position"] = "Attack"
            merged["sub_position"] = "Forward"
            merged["market_value_in_eur"] = 0.0
        else:
            return []

        for col in ("apps", "minutes", "goals", "assists", "yellow_cards"):
            merged[col] = merged[col].fillna(0).astype(int)

        merged["impact_score"] = (
            merged["goals"] * 4.0 + merged["assists"] * 2.5 + (merged["market_value_in_eur"] / 10_000_000.0)
        )
        merged = merged.sort_values("impact_score", ascending=False).head(top_n)

        total_team_goals = max(8.0, float(merged["goals"].sum()) * 1.35)
        out = []
        for _, r in merged.iterrows():
            pos = str(r.get("sub_position") or r.get("position") or "")
            g_cnt = int(r["goals"])
            share = max(0.04, min(0.42, (g_cnt + 1.2) / (total_team_goals + 5.0)))
            if "Goalkeeper" in pos:
                share = 0.005
            elif "Back" in pos or "Defender" in pos:
                share = min(share, 0.08)
            lam_p = max(0.01, team_expected_goals * share)
            anytime_prob = 1.0 - np.exp(-lam_p)
            out.append(
                {
                    "player_id": int(r["player_id"]),
                    "name": str(r["name"]),
                    "position": pos,
                    "apps": int(r["apps"]),
                    "minutes": int(r["minutes"]),
                    "goals": g_cnt,
                    "assists": int(r["assists"]),
                    "market_value_m": round(float(r.get("market_value_in_eur") or 0.0) / 1_000_000.0, 1),
                    "anytime_scorer": _fmt_market_outcome(anytime_prob),
                }
            )
        return out

    def get_top5_league_summary(self, comp_code: str, elo_dict: dict[float, float] | None = None) -> dict[str, Any]:
        """Return specialized analytical hub for any of Europe's Big-5 leagues (GB1, ES1, L1, IT1, FR1)."""
        self.ensure_loaded()
        code = comp_code.upper()
        meta_info = TOP5_EUROPEAN_LEAGUES.get(code, {
            "code": code,
            "name_en": code,
            "name_fa": code,
            "country_en": "",
            "country_fa": "",
        })

        league_stats = {
            "avg_goals_per_match": 2.75,
            "home_win_pct": 44.5,
            "draw_pct": 24.5,
            "away_win_pct": 31.0,
            "btts_pct": 53.0,
            "over_2_5_pct": 52.5,
            "recent_matches_analyzed": 0,
        }
        if self.games_df is not None:
            lg = self.games_df[
                (self.games_df["competition_id"] == code)
                & (self.games_df["date"].astype(str) >= "2023-08-01")
            ]
            if not lg.empty:
                n = len(lg)
                hg = lg["home_club_goals"].values
                ag = lg["away_club_goals"].values
                tot = hg + ag
                league_stats = {
                    "avg_goals_per_match": round(float(np.mean(tot)), 2),
                    "home_win_pct": round(float(np.mean(hg > ag)) * 100.0, 1),
                    "draw_pct": round(float(np.mean(hg == ag)) * 100.0, 1),
                    "away_win_pct": round(float(np.mean(hg < ag)) * 100.0, 1),
                    "btts_pct": round(float(np.mean((hg >= 1) & (ag >= 1))) * 100.0, 1),
                    "over_2_5_pct": round(float(np.mean(tot >= 3)) * 100.0, 1),
                    "recent_matches_analyzed": int(n),
                }

        top_clubs = []
        if self.clubs_df is not None:
            c_sub = self.clubs_df[
                (self.clubs_df["domestic_competition_id"] == code) & (self.clubs_df["last_season"] >= 2024)
            ]
            for _, cr in c_sub.iterrows():
                cid = float(cr["club_id"])
                prof = self.get_club_profile(cid, str(cr["name"]))
                elo_val = round(float((elo_dict or {}).get(cid, 1650.0)), 1)
                prof["elo"] = elo_val
                top_clubs.append(prof)
            top_clubs.sort(key=lambda x: (x["elo"], x["squad_market_value_m"]), reverse=True)

        top_players = []
        if self.appearances_recent_df is not None:
            ap_lg = self.appearances_recent_df[self.appearances_recent_df["competition_id"] == code]
            if not ap_lg.empty:
                agg = (
                    ap_lg.groupby(["player_id", "player_name"], as_index=False)
                    .agg(
                        goals=("goals", "sum"),
                        assists=("assists", "sum"),
                        apps=("minutes_played", "count"),
                        minutes=("minutes_played", "sum"),
                    )
                )
                agg["ga"] = agg["goals"] + agg["assists"]
                agg = agg.sort_values(["goals", "ga"], ascending=False).head(12)
                for _, pr in agg.iterrows():
                    top_players.append(
                        {
                            "player_id": int(pr["player_id"]),
                            "name": str(pr["player_name"]),
                            "goals": int(pr["goals"]),
                            "assists": int(pr["assists"]),
                            "goal_contributions": int(pr["ga"]),
                            "apps": int(pr["apps"]),
                            "minutes": int(pr["minutes"]),
                        }
                    )

        return {
            "league": meta_info,
            "analytical_profile": league_stats,
            "clubs": top_clubs[:20],
            "top_scorers_and_creators": top_players,
        }


_GLOBAL_INTEL_STORE: HistoricalIntelligenceStore | None = None


def get_intelligence_store(data_dir: Path = DATA_DIR) -> HistoricalIntelligenceStore:
    global _GLOBAL_INTEL_STORE
    if _GLOBAL_INTEL_STORE is None:
        _GLOBAL_INTEL_STORE = HistoricalIntelligenceStore(data_dir=data_dir)
    return _GLOBAL_INTEL_STORE
