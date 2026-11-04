"""
WC26 Analytics Engine: End-to-end training, evaluation, inference, and simulation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import warnings
warnings.filterwarnings("ignore")
import argparse
import time
import json
import joblib
import numpy as np
import pandas as pd

from config import LABEL_TO_INT, INT_TO_LABEL, KNOWN_NATIONAL_TEAM_IDS
from data_loader import (
    load_csv,
    validate_required_csv_files,
    normalize_team_dim,
    rectify_knockout_scores,
    resolve_team_id,
    normalize_name,
)
from elo_poisson import (
    calculate_dynamic_elo,
    compute_poisson_features,
    calculate_dixon_coles_probs,
)
from player_profiler import (
    build_valuation_history,
    build_player_appearance_history,
    build_lineup_features,
    aggregate_squad_features,
)
from feature_engineering import (
    build_team_history,
    build_manager_history,
    build_match_profiles,
    build_diff_features,
    calculate_sample_weights,
)
from models import (
    get_feature_columns,
    build_ensemble_pipeline,
    evaluate_predictions,
    save_artifacts,
    load_artifacts,
    prob_to_decimal_odds,
    prob_to_american_odds,
)


CACHE_FILENAME = "match_profiles_cache.joblib"
METADATA_FILENAME = "engine_metadata.joblib"


def get_or_build_dataset(data_dir: Path, artifacts_dir: Path, force_rebuild: bool = False):
    """
    Build or load from cache the full 88k+ match dataset with all 220+ features.
    """
    cache_path = artifacts_dir / CACHE_FILENAME
    meta_path = artifacts_dir / METADATA_FILENAME

    if not force_rebuild and cache_path.is_file() and meta_path.is_file():
        print(f"Loading pre-computed dataset cache from {cache_path}...", flush=True)
        profiles = joblib.load(cache_path)
        meta = joblib.load(meta_path)
        return profiles, meta

    print("Building complete feature dataset from raw CSVs...", flush=True)
    validate_required_csv_files(data_dir)

    t0 = time.time()
    games = load_csv(data_dir, "games.csv")
    game_events = load_csv(data_dir, "game_events.csv")
    clubs = load_csv(data_dir, "clubs.csv")
    national_teams = load_csv(data_dir, "national_teams.csv")
    competitions = load_csv(data_dir, "competitions.csv")
    countries = load_csv(data_dir, "countries.csv")
    club_games = load_csv(data_dir, "club_games.csv")
    players = load_csv(data_dir, "players.csv")
    player_valuations = load_csv(data_dir, "player_valuations.csv")
    appearances = load_csv(data_dir, "appearances.csv")
    game_lineups = load_csv(data_dir, "game_lineups.csv")

    print(f"Loaded all CSVs in {time.time() - t0:.2f}s", flush=True)

    t1 = time.time()
    games = rectify_knockout_scores(games, game_events)
    team_dim = normalize_team_dim(clubs, national_teams, competitions, countries, games)

    games, elo_dict = calculate_dynamic_elo(games, national_teams)
    games = compute_poisson_features(games)

    val_hist = build_valuation_history(player_valuations)
    app_hist = build_player_appearance_history(appearances)
    lineup_feat = build_lineup_features(game_lineups, players, app_hist, val_hist)
    squad_feat = aggregate_squad_features(lineup_feat)

    team_hist = build_team_history(club_games, games)
    manager_hist = build_manager_history(team_hist)

    profiles = build_match_profiles(games, team_hist, manager_hist, squad_feat, team_dim)
    profiles = build_diff_features(profiles)
    print(f"Dataset fully constructed ({len(profiles)} matches, {profiles.shape[1]} columns) in {time.time() - t1:.2f}s", flush=True)

    # Cache metadata for inference
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "team_dim": team_dim,
        "elo_dict": elo_dict,
        "latest_team_hist": team_hist.sort_values(["club_id", "date"]).groupby("club_id").last().reset_index(),
        "latest_manager_hist": manager_hist.sort_values(["club_id", "date"]).groupby("club_id").last().reset_index(),
        "latest_squad_feat": squad_feat.sort_values(["club_id", "game_id"]).groupby("club_id").last().reset_index(),
    }

    joblib.dump(profiles, cache_path, compress=3)
    joblib.dump(meta, meta_path, compress=3)
    print(f"Saved dataset cache to {cache_path}", flush=True)

    return profiles, meta


def run_training(data_dir: Path, artifacts_dir: Path, force_rebuild: bool = False):
    """
    Train ensemble model (LGBM + CatBoost + RF) on matches prior to WC 2026.
    Evaluate on the 72 World Cup 2026 matches.
    """
    profiles, meta = get_or_build_dataset(data_dir, artifacts_dir, force_rebuild=force_rebuild)

    print("\n--- Starting Model Training ---", flush=True)
    profiles["date"] = pd.to_datetime(profiles["date"])
    
    # Temporal Split:
    # Train set = all historical matches prior to 2026-06-11
    # Test set = World Cup 2026 Group Stage matches (2026-06-11 to 2026-06-28)
    train_mask = (profiles["date"] < "2026-06-11") & profiles["label"].notna()
    test_mask = (profiles["date"] >= "2026-06-11") & profiles["competition_id"].eq("FIWC") & profiles["label"].notna()

    train_df = profiles[train_mask].copy()
    test_df = profiles[test_mask].copy()

    print(f"Training dataset size: {len(train_df):,} matches", flush=True)
    print(f"World Cup 2026 evaluation dataset size: {len(test_df)} matches", flush=True)

    num_cols, cat_cols = get_feature_columns(train_df)
    print(f"Features: {len(num_cols)} numeric + {len(cat_cols)} categorical = {len(num_cols) + len(cat_cols)} total features", flush=True)

    # Sample weights
    sample_weights = calculate_sample_weights(train_df)

    X_train = train_df[num_cols + cat_cols]
    y_train = train_df["label"].astype(int).values

    X_test = test_df[num_cols + cat_cols]
    y_test = test_df["label"].astype(int).values

    print("Fitting ensemble pipeline (LightGBM + CatBoost + Random Forest with sample weights)...", flush=True)
    t_train = time.time()
    pipeline = build_ensemble_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train, classifier__sample_weight=sample_weights)
    print(f"Ensemble training completed in {time.time() - t_train:.2f}s", flush=True)

    # Save artifacts
    save_artifacts(pipeline, num_cols, cat_cols, artifacts_dir)
    print(f"Saved model artifacts to {artifacts_dir}", flush=True)

    # Evaluate on World Cup 2026 Group Stage
    print("\n--- Evaluating Model on World Cup 2026 Matches ---", flush=True)
    y_prob = pipeline.predict_proba(X_test)
    metrics = evaluate_predictions(y_test, y_prob)

    print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"Log Loss:           {metrics['log_loss']:.4f}")
    print(f"Brier Score:        {metrics['brier_score']:.4f}")
    print(f"ROC-AUC (Macro OVR): {metrics['roc_auc_ovr']:.4f}")
    print(f"Draw Precision:     {metrics['draw_precision']:.4f}")
    print(f"Draw Recall:        {metrics['draw_recall']:.4f}")
    print(f"Draw F1-score:      {metrics['draw_f1']:.4f}")
    print(f"Win F1-score:       {metrics['win_f1']:.4f}")
    print(f"Loss F1-score:      {metrics['loss_f1']:.4f}")
    print(f"Macro F1-score:     {metrics['macro_f1']:.4f}")

    # Export match-by-match evaluation
    eval_df = test_df[["game_id", "date", "home_club_name", "away_club_name", "home_club_goals", "away_club_goals", "label"]].copy()
    eval_df["actual_outcome"] = eval_df["label"].map(INT_TO_LABEL)
    eval_df["prob_loss_away_win"] = y_prob[:, 0]
    eval_df["prob_draw"] = y_prob[:, 1]
    eval_df["prob_win_home_win"] = y_prob[:, 2]
    eval_df["pred_label"] = np.argmax(y_prob, axis=1)
    eval_df["pred_outcome"] = eval_df["pred_label"].map(INT_TO_LABEL)
    eval_df["is_correct"] = eval_df["label"].eq(eval_df["pred_label"]).astype(int)

    # Betting odds
    eval_df["dec_odds_home"] = [prob_to_decimal_odds(p) for p in y_prob[:, 2]]
    eval_df["dec_odds_draw"] = [prob_to_decimal_odds(p) for p in y_prob[:, 1]]
    eval_df["dec_odds_away"] = [prob_to_decimal_odds(p) for p in y_prob[:, 0]]

    eval_df["us_odds_home"] = [prob_to_american_odds(p) for p in y_prob[:, 2]]
    eval_df["us_odds_draw"] = [prob_to_american_odds(p) for p in y_prob[:, 1]]
    eval_df["us_odds_away"] = [prob_to_american_odds(p) for p in y_prob[:, 0]]

    eval_out_path = artifacts_dir / "wc26_group_stage_evaluation.csv"
    eval_df.to_csv(eval_out_path, index=False)
    print(f"\nExported complete match-by-match evaluation to {eval_out_path}", flush=True)

    metrics_out_path = artifacts_dir / "evaluation_metrics.json"
    with open(metrics_out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics, eval_df


def predict_match(
    home_name: str,
    away_name: str,
    artifacts_dir: Path,
    is_neutral: bool = True,
    match_date: str = "2026-06-15",
):
    """
    Generate Win / Draw / Loss probabilities and full Betting Odds for any matchup.
    """
    pipeline, num_cols, cat_cols = load_artifacts(artifacts_dir)
    meta_path = artifacts_dir / METADATA_FILENAME
    if not meta_path.is_file():
        raise FileNotFoundError(f"Metadata file {meta_path} missing. Run --mode train first.")
        
    meta = joblib.load(meta_path)
    team_dim = meta["team_dim"]
    elo_dict = meta["elo_dict"]
    latest_team = meta["latest_team_hist"]
    latest_manager = meta["latest_manager_hist"]
    latest_squad = meta["latest_squad_feat"]

    h_id = resolve_team_id(team_dim, home_name)
    a_id = resolve_team_id(team_dim, away_name)

    if np.isnan(h_id):
        raise ValueError(f"Could not resolve home team: '{home_name}'")
    if np.isnan(a_id):
        raise ValueError(f"Could not resolve away team: '{away_name}'")

    h_id = float(h_id)
    a_id = float(a_id)

    # Elo & Poisson
    h_elo = elo_dict.get(h_id, 1500.0)
    a_elo = elo_dict.get(a_id, 1500.0)
    home_adv = 0.0 if is_neutral else 65.0
    diff_elo = (h_elo - a_elo) + home_adv
    ratio_elo = h_elo / max(1.0, a_elo)

    lambda_home = np.clip(1.35 + (diff_elo / 500.0), 0.2, 4.5)
    lambda_away = np.clip(1.15 - (diff_elo / 500.0), 0.2, 4.5)
    pw, pd_val, pl = calculate_dixon_coles_probs(lambda_home, lambda_away)

    # Base match row
    row = {
        "date": match_date,
        "competition_type": "national_team_competition",
        "home_club_id": h_id,
        "away_club_id": a_id,
        "home_elo_before": h_elo,
        "away_elo_before": a_elo,
        "diff_elo": diff_elo,
        "ratio_elo": ratio_elo,
        "expected_home_goals": lambda_home,
        "expected_away_goals": lambda_away,
        "expected_goal_diff": lambda_home - lambda_away,
        "expected_total_goals": lambda_home + lambda_away,
        "poisson_prob_win": pw,
        "poisson_prob_draw": pd_val,
        "poisson_prob_loss": pl,
    }

    # Attach team profiles
    h_team = latest_team[latest_team["club_id"] == h_id]
    if not h_team.empty:
        for c in h_team.columns:
            if c.startswith("team_"):
                row[f"home_{c}"] = h_team.iloc[0][c]

    a_team = latest_team[latest_team["club_id"] == a_id]
    if not a_team.empty:
        for c in a_team.columns:
            if c.startswith("team_"):
                row[f"away_{c}"] = a_team.iloc[0][c]

    # Attach manager profiles
    h_mgr = latest_manager[latest_manager["club_id"] == h_id]
    if not h_mgr.empty:
        for c in h_mgr.columns:
            if c.startswith("manager_") and c != "manager_name":
                row[f"home_{c}"] = h_mgr.iloc[0][c]

    a_mgr = latest_manager[latest_manager["club_id"] == a_id]
    if not a_mgr.empty:
        for c in a_mgr.columns:
            if c.startswith("manager_") and c != "manager_name":
                row[f"away_{c}"] = a_mgr.iloc[0][c]

    # Attach squad profiles
    h_sq = latest_squad[latest_squad["club_id"] == h_id]
    if not h_sq.empty:
        for c in h_sq.columns:
            if c not in ["game_id", "club_id"]:
                row[f"home_{c}"] = h_sq.iloc[0][c]

    a_sq = latest_squad[latest_squad["club_id"] == a_id]
    if not a_sq.empty:
        for c in a_sq.columns:
            if c not in ["game_id", "club_id"]:
                row[f"away_{c}"] = a_sq.iloc[0][c]

    # Team dimension lookups
    h_dim = team_dim[team_dim["team_id"] == h_id]
    if not h_dim.empty:
        row["home_team_kind"] = h_dim.iloc[0].get("team_kind", "national_team")
        row["home_team_confederation"] = h_dim.iloc[0].get("team_confederation", "unknown")
    else:
        row["home_team_kind"] = "national_team"
        row["home_team_confederation"] = "unknown"

    a_dim = team_dim[team_dim["team_id"] == a_id]
    if not a_dim.empty:
        row["away_team_kind"] = a_dim.iloc[0].get("team_kind", "national_team")
        row["away_team_confederation"] = a_dim.iloc[0].get("team_confederation", "unknown")
    else:
        row["away_team_kind"] = "national_team"
        row["away_team_confederation"] = "unknown"

    df_row = pd.DataFrame([row])
    # Build differential and ratio features
    df_row = build_diff_features(df_row)

    # Ensure all expected columns exist
    for col in num_cols:
        if col not in df_row.columns:
            df_row[col] = np.nan
    for col in cat_cols:
        if col not in df_row.columns:
            df_row[col] = "unknown"

    inference_features = df_row[num_cols + cat_cols]
    y_prob = pipeline.predict_proba(inference_features)[0]

    p_loss, p_draw, p_win = y_prob[0], y_prob[1], y_prob[2]

    result = {
        "home_team": home_name.title(),
        "away_team": away_name.title(),
        "home_elo": round(h_elo, 1),
        "away_elo": round(a_elo, 1),
        "expected_goals_home": round(lambda_home, 2),
        "expected_goals_away": round(lambda_away, 2),
        "probabilities": {
            "home_win": round(p_win * 100, 2),
            "draw": round(p_draw * 100, 2),
            "away_win": round(p_loss * 100, 2),
        },
        "decimal_odds": {
            "home_win": prob_to_decimal_odds(p_win),
            "draw": prob_to_decimal_odds(p_draw),
            "away_win": prob_to_decimal_odds(p_loss),
        },
        "american_odds": {
            "home_win": prob_to_american_odds(p_win),
            "draw": prob_to_american_odds(p_draw),
            "away_win": prob_to_american_odds(p_loss),
        },
    }

    return result


def print_prediction_card(res: dict):
    """Print an elegant prediction and odds card in the console."""
    h = res["home_team"]
    a = res["away_team"]
    p = res["probabilities"]
    d = res["decimal_odds"]
    us = res["american_odds"]

    print("\n" + "=" * 65)
    print(f"      MATCH PREDICTION & ODDS: {h.upper()} vs {a.upper()}")
    print("=" * 65)
    print(f" Elo Rating:         {h}: {res['home_elo']}  |  {a}: {res['away_elo']}")
    print(f" Expected Goals:     {h} {res['expected_goals_home']} - {res['expected_goals_away']} {a}")
    print("-" * 65)
    print(f" {'Outcome':<18} | {'Probability':<12} | {'Decimal Odds':<14} | {'American Odds':<12}")
    print("-" * 65)
    print(f" {h + ' Win':<18} | {p['home_win']:>9.1f}%  | {d['home_win']:>12.2f}  | {us['home_win']:>12}")
    print(f" {'Draw':<18} | {p['draw']:>9.1f}%  | {d['draw']:>12.2f}  | {us['draw']:>12}")
    print(f" {a + ' Win':<18} | {p['away_win']:>9.1f}%  | {d['away_win']:>12.2f}  | {us['away_win']:>12}")
    print("=" * 65 + "\n")


def run_batch_simulation(
    input_file: Path,
    output_file: Path,
    artifacts_dir: Path,
    n_simulations: int = 500,
):
    """
    Run position-aware Monte Carlo simulation on a batch of matches.
    """
    if not input_file.is_file():
        raise FileNotFoundError(f"Simulation input file not found: {input_file}")

    print(f"Running Monte Carlo simulations ({n_simulations} runs per match) on {input_file}...", flush=True)
    df = pd.read_csv(input_file)
    results = []

    for _, match in df.iterrows():
        h = str(match.get("home_team", ""))
        a = str(match.get("away_team", ""))
        if not h or not a:
            continue

        try:
            pred = predict_match(h, a, artifacts_dir, is_neutral=True)
            p_win = pred["probabilities"]["home_win"] / 100.0
            p_draw = pred["probabilities"]["draw"] / 100.0
            p_loss = pred["probabilities"]["away_win"] / 100.0

            # Monte Carlo sampling
            probs = np.array([p_win, p_draw, p_loss], dtype=float)
            probs = np.clip(probs, 0.0, 1.0)
            probs = probs / probs.sum()
            samples = np.random.choice([2, 1, 0], size=n_simulations, p=probs)
            sim_win = float(np.mean(samples == 2) * 100.0)
            sim_draw = float(np.mean(samples == 1) * 100.0)
            sim_loss = float(np.mean(samples == 0) * 100.0)

            results.append({
                "home_team": h,
                "away_team": a,
                "home_elo": pred["home_elo"],
                "away_elo": pred["away_elo"],
                "expected_home_goals": pred["expected_goals_home"],
                "expected_away_goals": pred["expected_goals_away"],
                "model_prob_win": pred["probabilities"]["home_win"],
                "model_prob_draw": pred["probabilities"]["draw"],
                "model_prob_loss": pred["probabilities"]["away_win"],
                "sim_prob_win": round(sim_win, 2),
                "sim_prob_draw": round(sim_draw, 2),
                "sim_prob_loss": round(sim_loss, 2),
                "dec_odds_home": pred["decimal_odds"]["home_win"],
                "dec_odds_draw": pred["decimal_odds"]["draw"],
                "dec_odds_away": pred["decimal_odds"]["away_win"],
                "us_odds_home": pred["american_odds"]["home_win"],
                "us_odds_draw": pred["american_odds"]["draw"],
                "us_odds_away": pred["american_odds"]["away_win"],
            })
        except Exception as e:
            print(f"Skipping {h} vs {a}: {e}")

    out_df = pd.DataFrame(results)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_file, index=False)
    print(f"Exported simulation results for {len(out_df)} matches to {output_file}", flush=True)
    return out_df


def main():
    parser = argparse.ArgumentParser(description="World Cup 2026 Predictive Engine & Betting Odds")
    parser.add_argument("--mode", choices=["train", "test_wc26", "predict", "simulate"], default="test_wc26",
                        help="Operation mode: train model, evaluate on WC26, predict match, or simulate batch")
    parser.add_argument("--data-dir", type=str, default=".", help="Path to CSV datasets")
    parser.add_argument("--artifacts-dir", "--output-dir", dest="artifacts_dir", type=str, default="artifacts", help="Path to model artifacts")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuilding feature cache from CSVs")
    parser.add_argument("--home", "--predict-home-team", dest="home", type=str, default=None, help="Home team name for prediction")
    parser.add_argument("--away", "--predict-away-team", dest="away", type=str, default=None, help="Away team name for prediction")
    parser.add_argument("--neutral", "--neutral-site", dest="neutral", action="store_true", default=True, help="Neutral venue flag")
    parser.add_argument("--input", "--prediction-csv", dest="input", type=str, default="src/sample_predictions.csv", help="Input CSV for simulation")
    parser.add_argument("--output", type=str, default="artifacts/batch_predictions_results.csv", help="Output CSV for simulation")
    parser.add_argument("--runs", "--simulation-runs", dest="runs", type=int, default=500, help="Number of Monte Carlo simulation runs")

    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    artifacts_dir = Path(args.artifacts_dir).resolve()

    if args.mode == "train":
        run_training(data_dir, artifacts_dir, force_rebuild=args.force_rebuild)

    elif args.mode == "test_wc26":
        model_path = artifacts_dir / "wc26_ensemble_model.joblib"
        if not model_path.is_file():
            print(f"Trained model not found at {model_path}. Running training first...", flush=True)
            run_training(data_dir, artifacts_dir, force_rebuild=args.force_rebuild)
        else:
            profiles, meta = get_or_build_dataset(data_dir, artifacts_dir, force_rebuild=args.force_rebuild)
            pipeline, num_cols, cat_cols = load_artifacts(artifacts_dir)
            profiles["date"] = pd.to_datetime(profiles["date"])
            test_mask = (profiles["date"] >= "2026-06-11") & profiles["competition_id"].eq("FIWC") & profiles["label"].notna()
            test_df = profiles[test_mask].copy()

            X_test = test_df[num_cols + cat_cols]
            y_test = test_df["label"].astype(int).values
            y_prob = pipeline.predict_proba(X_test)
            metrics = evaluate_predictions(y_test, y_prob)

            print("\n=== World Cup 2026 Group Stage Evaluation (72 Matches) ===")
            print(f"Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
            print(f"Log Loss:           {metrics['log_loss']:.4f}")
            print(f"Brier Score:        {metrics['brier_score']:.4f}")
            print(f"ROC-AUC (Macro OVR): {metrics['roc_auc_ovr']:.4f}")
            print(f"Draw Precision:     {metrics['draw_precision']:.4f}")
            print(f"Draw Recall:        {metrics['draw_recall']:.4f}")
            print(f"Draw F1-score:      {metrics['draw_f1']:.4f}")
            print(f"Win F1-score:       {metrics['win_f1']:.4f}")
            print(f"Loss F1-score:      {metrics['loss_f1']:.4f}")
            print(f"Macro F1-score:     {metrics['macro_f1']:.4f}")

    elif args.mode == "predict":
        home = args.home or input("Enter Home team name: ").strip()
        away = args.away or input("Enter Away team name: ").strip()
        res = predict_match(home, away, artifacts_dir, is_neutral=args.neutral)
        print_prediction_card(res)

    elif args.mode == "simulate":
        input_path = Path(args.input).resolve()
        output_path = Path(args.output).resolve()
        run_batch_simulation(input_path, output_path, artifacts_dir, n_simulations=args.runs)


if __name__ == "__main__":
    main()
