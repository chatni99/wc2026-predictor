"""
WC2026 V2 — Feature Engineering Functions
Single source of truth for all feature calculations.
All functions are point-in-time correct: only use data strictly
before the match date being predicted.
"""

import pandas as pd
import numpy as np
from scipy.stats import poisson


# ═══════════════════════════════════════════
# FEATURE COLUMNS — single source of truth
# ═══════════════════════════════════════════

# Every feature written to feature_matrix.csv by 02_feature_engineering (25)
ALL_FEATURE_COLS = [
    'home_form', 'home_attack', 'home_defense', 'home_elo', 'home_has_full_history',
    'away_form', 'away_attack', 'away_defense', 'away_elo', 'away_has_full_history',
    'h2h_matches', 'h2h_home_win_rate', 'elo_diff', 'elo_closeness', 'defensive_strength',
    'home_clean_sheet_rate', 'away_clean_sheet_rate',
    'home_days_rest', 'away_days_rest',
    'home_second_half_goals_pct', 'away_second_half_goals_pct',
    'home_penalty_goal_pct', 'away_penalty_goal_pct',
    'home_shootout_win_rate', 'away_shootout_win_rate',
]

# Features the trained models actually use (23): days_rest was dropped in 03.
# Order matters -- it must match the saved XGBoost model and scaler.
FEATURE_COLS = [c for c in ALL_FEATURE_COLS if c not in ('home_days_rest', 'away_days_rest')]


# ═══════════════════════════════════════════
# TEAM MATCH INDEX — pre-grouped for speed
# ═══════════════════════════════════════════

# def build_team_match_index(clean_df):
#     team_matches = {}
#     all_teams = pd.concat([clean_df['home_team'], clean_df['away_team']]).unique()
#     for team in all_teams:
#         mask = (clean_df['home_team'] == team) | (clean_df['away_team'] == team)
#         team_matches[team] = clean_df[mask].sort_values('date').reset_index(drop=True)
#     return team_matches
## new function for:: goals_for and goals_against
def build_team_match_index(clean_df):
    team_matches = {}
    all_teams = pd.concat([clean_df['home_team'], clean_df['away_team']]).unique()
    for team in all_teams:
        mask = (clean_df['home_team'] == team) | (clean_df['away_team'] == team)
        df = clean_df[mask].sort_values('date').reset_index(drop=True).copy()
        df['goals_for']     = np.where(df['home_team'] == team, df['home_score'], df['away_score'])
        df['goals_against'] = np.where(df['home_team'] == team, df['away_score'], df['home_score'])
        team_matches[team] = df
    return team_matches


# ═══════════════════════════════════════════
# GOALS INDEX — pre-grouped, for goal-level features
# ═══════════════════════════════════════════

def build_goals_index(goals_df):
    return {team: g[['date', 'minute', 'penalty']].sort_values('date').reset_index(drop=True)
            for team, g in goals_df.groupby('team')}


# ═══════════════════════════════════════════
# FORM — exponentially weighted, last 5 matches
# ═══════════════════════════════════════════

def get_team_form_fast(team, date, team_index, n=5, decay=0.8):
    if team not in team_index:
        return None
    team_matches = team_index[team]
    prior = team_matches[team_matches['date'] < date].tail(n)
    if len(prior) == 0:
        return None

    points = 0
    n_matches = len(prior)
    for i, (_, row) in enumerate(prior.iterrows()):
        scored   = row['home_score'] if row['home_team'] == team else row['away_score']
        conceded = row['away_score'] if row['home_team'] == team else row['home_score']
        weight = decay ** (n_matches - 1 - i)
        if scored > conceded:
            points += 3 * weight
        elif scored == conceded:
            points += 1 * weight
    return round(points, 4)


# ═══════════════════════════════════════════
# ATTACK / DEFENSE — last 10 matches
# ═══════════════════════════════════════════

def get_attack_defense_fast(team, date, team_index, n=10):
    if team not in team_index:
        return None, None
    team_matches = team_index[team]
    prior = team_matches[team_matches['date'] < date].tail(n)
    if len(prior) == 0:
        return None, None

    scored_list, conceded_list = [], []
    for _, row in prior.iterrows():
        if row['home_team'] == team:
            scored_list.append(row['home_score'])
            conceded_list.append(row['away_score'])
        else:
            scored_list.append(row['away_score'])
            conceded_list.append(row['home_score'])
    return round(sum(scored_list)/len(scored_list), 3), round(sum(conceded_list)/len(conceded_list), 3)


# ═══════════════════════════════════════════
# CLEAN SHEET RATE — last 10 matches
# ═══════════════════════════════════════════

def get_clean_sheet_rate(team, date, team_index, n=10):
    if team not in team_index:
        return 0.5
    prior = team_index[team][team_index[team]['date'] < date].tail(n)
    if len(prior) == 0:
        return 0.5
    conceded = prior.apply(
        lambda row: row['away_score'] if row['home_team'] == team else row['home_score'], axis=1
    )
    return round((conceded == 0).mean(), 4)


# ═══════════════════════════════════════════
# DAYS REST — since last match
# ═══════════════════════════════════════════

def get_days_rest(team, date, team_index):
    if team not in team_index:
        return 14
    prior = team_index[team][team_index[team]['date'] < date]
    if len(prior) == 0:
        return 14
    return (date - prior.iloc[-1]['date']).days


# ═══════════════════════════════════════════
# SECOND HALF GOALS % — career, from goals_index
# ═══════════════════════════════════════════

def get_second_half_goals_pct(team, date, goals_index):
    if team not in goals_index:
        return 0.5
    prior = goals_index[team][goals_index[team]['date'] < date].dropna(subset=['minute'])
    if len(prior) == 0:
        return 0.5
    return round((prior['minute'] > 45).mean(), 4)


# ═══════════════════════════════════════════
# PENALTY GOAL % — career, from goals_index
# ═══════════════════════════════════════════

def get_penalty_goal_pct(team, date, goals_index):
    if team not in goals_index:
        return 0.1
    prior = goals_index[team][goals_index[team]['date'] < date]
    if len(prior) == 0:
        return 0.1
    return round(prior['penalty'].mean(), 4)


# ═══════════════════════════════════════════
# HEAD TO HEAD
# ═══════════════════════════════════════════

def get_h2h_fast(team_a, team_b, date, team_index):
    if team_a not in team_index:
        return {'matches': 0, 'a_wins': 0, 'draws': 0, 'b_wins': 0, 'a_win_rate': None}

    team_matches = team_index[team_a]
    prior = team_matches[
        ((team_matches['away_team'] == team_b) | (team_matches['home_team'] == team_b)) &
        (team_matches['date'] < date)
    ]
    if len(prior) == 0:
        return {'matches': 0, 'a_wins': 0, 'draws': 0, 'b_wins': 0, 'a_win_rate': None}

    a_wins = draws = b_wins = 0
    for _, row in prior.iterrows():
        scored, conceded = (row['home_score'], row['away_score']) if row['home_team'] == team_a \
                            else (row['away_score'], row['home_score'])
        if scored > conceded: a_wins += 1
        elif scored == conceded: draws += 1
        else: b_wins += 1

    total = a_wins + draws + b_wins
    return {'matches': total, 'a_wins': a_wins, 'draws': draws, 'b_wins': b_wins,
            'a_win_rate': round(a_wins/total, 3)}


# ═══════════════════════════════════════════
# ELO — chronological, post-match ratings
# ═══════════════════════════════════════════

def calculate_elo_history(df, k=30, base_rating=1500):
    df_sorted = df.sort_values('date').reset_index(drop=True)
    current_elo = {}
    elo_records = []

    for _, row in df_sorted.iterrows():
        home, away = row['home_team'], row['away_team']
        home_elo = current_elo.get(home, base_rating)
        away_elo = current_elo.get(away, base_rating)

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
        expected_away = 1 - expected_home

        if row['home_score'] > row['away_score']:
            actual_home, actual_away = 1.0, 0.0
        elif row['home_score'] < row['away_score']:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home, actual_away = 0.5, 0.5

        new_home_elo = home_elo + k * (actual_home - expected_home)
        new_away_elo = away_elo + k * (actual_away - expected_away)

        elo_records.append({'date': row['date'], 'team': home, 'elo_after_match': new_home_elo})
        elo_records.append({'date': row['date'], 'team': away, 'elo_after_match': new_away_elo})

        current_elo[home] = new_home_elo
        current_elo[away] = new_away_elo

    return pd.DataFrame(elo_records), current_elo


def get_elo_at_date(team, date, elo_history):
    mask = (elo_history['team'] == team) & (elo_history['date'] < date)
    team_history = elo_history[mask].sort_values('date')
    if len(team_history) == 0:
        return 1500.0
    return team_history.iloc[-1]['elo_after_match']


# ═══════════════════════════════════════════
# SHOOTOUT WIN RATE — Bayesian shrinkage
# ═══════════════════════════════════════════

def build_shootout_index(shootouts_df):
    shootouts_df = shootouts_df.sort_values('date').reset_index(drop=True)
    index = {}
    all_teams = pd.concat([shootouts_df['home_team'], shootouts_df['away_team']]).unique()
    for team in all_teams:
        mask = (shootouts_df['home_team'] == team) | (shootouts_df['away_team'] == team)
        team_shootouts = shootouts_df[mask].copy()
        team_shootouts['won'] = team_shootouts['winner'] == team
        index[team] = team_shootouts[['date', 'won']].reset_index(drop=True)
    return index


def get_shootout_win_rate(team, date, shootout_index, k=3, prior=0.5):
    if team not in shootout_index:
        return prior
    team_shootouts = shootout_index[team]
    prior_shootouts = team_shootouts[team_shootouts['date'] < date]
    n = len(prior_shootouts)
    if n == 0:
        return prior
    wins = prior_shootouts['won'].sum()
    return round((wins + k * prior) / (n + k), 4)


# ═══════════════════════════════════════════
# POISSON — lambda calculation + scoreline matrix
# ═══════════════════════════════════════════

def calculate_lambda(home_attack, home_defense, away_attack, away_defense, avg_goals):
    lambda_home = (home_attack * away_defense) / (avg_goals + 1e-6)
    lambda_away = (away_attack * home_defense) / (avg_goals + 1e-6)
    lambda_home = max(0.2, min(5.0, lambda_home))
    lambda_away = max(0.2, min(5.0, lambda_away))
    return round(lambda_home, 4), round(lambda_away, 4)


def build_score_matrix(lambda_home, lambda_away, max_goals=6):
    matrix = np.zeros((max_goals+1, max_goals+1))
    for i in range(max_goals+1):
        for j in range(max_goals+1):
            matrix[i][j] = poisson.pmf(i, lambda_home) * poisson.pmf(j, lambda_away)
    return matrix / matrix.sum()


def build_match_features(home_team, away_team, date, team_index, goals_index, shootout_index, elo_history):
    """
    Builds the full 23-feature row for one match (real or hypothetical/future),
    point-in-time correct -- only uses data strictly before `date`.
    """
    h, a, d = home_team, away_team, date
    home_form = get_team_form_fast(h, d, team_index)
    away_form = get_team_form_fast(a, d, team_index)
    home_attack, home_defense = get_attack_defense_fast(h, d, team_index)
    away_attack, away_defense = get_attack_defense_fast(a, d, team_index)
    home_elo = get_elo_at_date(h, d, elo_history)
    away_elo = get_elo_at_date(a, d, elo_history)

    home_has_full_history = 1 if home_form is not None else 0
    away_has_full_history = 1 if away_form is not None else 0
    home_form = home_form if home_form is not None else 5.0
    away_form = away_form if away_form is not None else 5.0
    home_attack = home_attack if home_attack is not None else 1.38
    home_defense = home_defense if home_defense is not None else 1.38
    away_attack = away_attack if away_attack is not None else 1.38
    away_defense = away_defense if away_defense is not None else 1.38

    h2h = get_h2h_fast(h, a, d, team_index)
    h2h_home_win_rate = h2h['a_win_rate'] if h2h['a_win_rate'] is not None else 0.5

    elo_diff = home_elo - away_elo
    elo_closeness = 1 / (1 + abs(elo_diff))
    defensive_strength = min(1 / (home_defense + away_defense + 0.1), 3.0)

    return {
        'home_form': home_form, 'home_attack': home_attack, 'home_defense': home_defense,
        'home_elo': home_elo, 'home_has_full_history': home_has_full_history,
        'away_form': away_form, 'away_attack': away_attack, 'away_defense': away_defense,
        'away_elo': away_elo, 'away_has_full_history': away_has_full_history,
        'h2h_matches': h2h['matches'], 'h2h_home_win_rate': h2h_home_win_rate,
        'elo_diff': elo_diff, 'elo_closeness': elo_closeness, 'defensive_strength': defensive_strength,
        'home_clean_sheet_rate': get_clean_sheet_rate(h, d, team_index),
        'away_clean_sheet_rate': get_clean_sheet_rate(a, d, team_index),
        'home_second_half_goals_pct': get_second_half_goals_pct(h, d, goals_index),
        'away_second_half_goals_pct': get_second_half_goals_pct(a, d, goals_index),
        'home_penalty_goal_pct': get_penalty_goal_pct(h, d, goals_index),
        'away_penalty_goal_pct': get_penalty_goal_pct(a, d, goals_index),
        'home_shootout_win_rate': get_shootout_win_rate(h, d, shootout_index),
        'away_shootout_win_rate': get_shootout_win_rate(a, d, shootout_index),
    }