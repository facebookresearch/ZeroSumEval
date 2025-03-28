import math
import json
import yaml
import os
import argparse
from glob import glob
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.linear_model import LogisticRegression


def summarize_results(matches_df: pd.DataFrame) -> None:
    """Prints various summary metrics of the matches with per-model and per-role results."""

    # Initialize summary table
    if 'tie' in set(matches_df['winner']):
        outcomes = ['wins', 'draws', 'loses', 'total']
    else:
        outcomes = ['wins', 'loses', 'total']

    roles = list(set(matches_df['role_a']) | set(matches_df['role_b']))
    models = list(set(matches_df['model_a']) | set(matches_df['model_b']))

    # '~' is used as a special substring for table parsing later
    columns = sum([[f'{role}~{outcome}' for outcome in outcomes] for role in roles], [])

    results_df = pd.DataFrame(0, index=models, columns=columns)

    # Insert raw values
    for row in matches_df.itertuples():
        if row.winner == 'model_a':
            results_df.loc[row.model_a, f"{row.role_a}~wins"]  += 1
            results_df.loc[row.model_b, f"{row.role_b}~loses"] += 1
        elif row.winner == 'model_b':
            results_df.loc[row.model_a, f"{row.role_a}~loses"] += 1
            results_df.loc[row.model_b, f"{row.role_b}~wins"]  += 1
        else:
            results_df.loc[row.model_a, f"{row.role_a}~draws"] += 1
            results_df.loc[row.model_b, f"{row.role_b}~draws"] += 1

    # Aggregate results
    for role in roles:
        role_cols = results_df.filter(like=f'{role}~').columns
        results_df[f'{role}~total'] = results_df[role_cols].sum(axis=1)
    for outcome in outcomes:
        outcome_cols = results_df.filter(like=f'~{outcome}').columns
        results_df[f'all~{outcome}'] = results_df[outcome_cols].sum(axis=1)

    return results_df


def compute_mle_elo(
    df, SCALE=100, BASE=10, INIT_RATING=1000, role_weights=None
):
    """
    Computes Bradley-Terry rating score. Heavily inspired by:
    https://lmsys.org/blog/2023-12-07-leaderboard/
    """
    
    models = set(df['model_a']) | set(df['model_b'])
    models = list(sorted(models))

    ptbl_win = pd.DataFrame(0, index=models, columns=models)

    for row in df.itertuples():
        if row.winner == 'model_a':
            ptbl_win.loc[row.model_a, row.model_b] += 2 * role_weights[row.role_a]
        elif row.winner == 'model_b':
            ptbl_win.loc[row.model_b, row.model_a] += 2 * role_weights[row.role_b]
        else:
            ptbl_win.loc[row.model_a, row.model_b] += 1
            ptbl_win.loc[row.model_b, row.model_a] += 1

    models = pd.Series(np.arange(len(ptbl_win.index)), index=ptbl_win.index)

    p = len(models)
    X = np.zeros([p * (p - 1) * 2, p])
    Y = np.zeros(p * (p - 1) * 2)

    cur_row = 0
    sample_weights = []
    for m_a in ptbl_win.index:
        for m_b in ptbl_win.columns:
            if m_a == m_b:
                continue
            # if nan skip
            if math.isnan(ptbl_win.loc[m_a, m_b]) or math.isnan(ptbl_win.loc[m_b, m_a]):
                continue
            X[cur_row, models[m_a]] = +math.log(BASE)
            X[cur_row, models[m_b]] = -math.log(BASE)
            Y[cur_row] = 1.0
            sample_weights.append(ptbl_win.loc[m_a, m_b])

            X[cur_row + 1, models[m_a]] = +math.log(BASE)
            X[cur_row + 1, models[m_b]] = -math.log(BASE)
            Y[cur_row + 1] = 0.0
            sample_weights.append(ptbl_win.loc[m_b, m_a])
            cur_row += 2
    X = X[:cur_row]
    Y = Y[:cur_row]

    lr = LogisticRegression(fit_intercept=False, penalty=None, tol=1e-6)
    lr.fit(X, Y, sample_weight=sample_weights)
    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    return pd.Series(elo_scores, index=models.index).sort_values(ascending=False)


def get_bootstrap_result(battles, func_compute_elo, num_round, role_weights=None):
    """From https://lmsys.org/blog/2023-12-07-leaderboard/"""
    rows = []
    for _ in tqdm(range(num_round), desc="bootstrap"):
        try:
            rows.append(func_compute_elo(battles.sample(frac=1.0, replace=True), role_weights=role_weights))
        except KeyError:
            pass
    
    if len(rows) != num_round:
        print(f"Only used {len(rows)}/{num_round} bootstrap rounds due to samples that have zero matches for some models")

    df = pd.DataFrame(rows)
    return df[df.median().sort_values(ascending=False).index]


def convert_matches_to_df(logs_path: str, max_player_attempts: int, max_time_per_player: Optional[float] = None) -> pd.DataFrame:

    matches = []
    for match_results_path in glob(f'{logs_path}/**/matches/*/scores.json', recursive=True):
        with open(match_results_path) as f:
            scores = json.load(f)
        
        # TODO: just uses first two models of each matchup for now, probably not the correct/best way to do this for multi-player games
        models = list(scores.keys())[:2]

        for model in models:
            if scores[model]['attempts'] >= max_player_attempts or (max_time_per_player is not None and scores[model]['total_time'] >= max_time_per_player):
                scores[model]['score'] = -math.inf
            
        def winner(scores: dict, models: List[str]) -> str:
            advantage_a = scores[models[0]]['score'] - scores[models[1]]['score']
            if advantage_a > 0:
                return 'model_a'
            elif advantage_a < 0:
                return 'model_b'
            return 'tie'
        
        matches.append([
            models[0],
            scores[models[0]]['role'],
            models[1],
            scores[models[1]]['role'],
            winner(scores, models),
        ])

    columns = ['model_a', 'role_a', 'model_b', 'role_b', 'winner']
    match_df = pd.DataFrame(matches, columns=columns)

    return match_df


def calculate_ratings(
    logs_path: str,
    bootstrap_rounds: int,
    max_time_per_player: Optional[float] = None,
    models: Optional[List[str]] = None,
    role_weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    

    with open(os.path.join(logs_path, 'pool_config.yaml')) as f:
        config = yaml.safe_load(f)
        max_player_attempts = config['manager']['max_player_attempts']

    match_df = convert_matches_to_df(logs_path=logs_path, max_player_attempts=max_player_attempts, max_time_per_player=max_time_per_player)

    if models is not None:
        match_df = match_df[match_df['model_a'].isin(models)]
        match_df = match_df[match_df['model_b'].isin(models)]
    roles = set(match_df['role_a']) | set(match_df['role_b'])
    
    if role_weights is None:
        role_weights = {role: 1.0 for role in roles}
    else:
        for role in roles:
            if role not in role_weights:
                role_weights[role] = 1.0

    results_df = summarize_results(match_df)

    np.random.seed(1)
    bootstrap_elo_lu = get_bootstrap_result(match_df, compute_mle_elo, bootstrap_rounds, role_weights)

    ratings_df = pd.DataFrame({
        'rating~lower': bootstrap_elo_lu.quantile(.025),
        'rating~predicted': compute_mle_elo(match_df, role_weights=role_weights),
        'rating~upper': bootstrap_elo_lu.quantile(.975),
    })

    results_df = results_df.join(ratings_df)

    results_df = results_df.sort_values("rating~predicted", ascending=False)
    results_df = results_df.round(decimals=1)

    results_df.columns = pd.MultiIndex.from_tuples(tuple(col.split('~')) for col in results_df.columns)
    
    results_df.to_json(os.path.join(logs_path, 'ratings.json'), orient='records')

    return results_df

    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--logs-path", "-p", help="Path to the match logs file", required=True)
    parser.add_argument("--bootstrap-rounds", "-b", help="Number of rounds to bootstrap for confidence intervals.", type=int, default=10_000)
    parser.add_argument("--max-time-per-player", "-t", help="Maximum number of player time.", type=float, default=None)
    parser.add_argument("--models", "-m", help="Filter analysis to matches with only these models. Uses all models by default", type=str, nargs='+', default=None)
    parser.add_argument("--role-weights", "-r", help="Weight for each role in format 'role=weight'. Uses equal weights by default", type=str, nargs='+', default=None)
    args = parser.parse_args()

    if args.role_weights is not None:
        args.role_weights = {role: float(weight) for role, weight in (arg.split('=') for arg in args.role_weights)}

    results_df = calculate_ratings(logs_path=args.logs_path, bootstrap_rounds=args.bootstrap_rounds, max_time_per_player=args.max_time_per_player, models=args.models, role_weights=args.role_weights)

    print(results_df)
