"""
Feature Engineering Module

Provides feature computation for goal prediction models.
While basic Poisson/Dixon-Coles models don't require explicit features
(they estimate team strength internally), this module demonstrates
how to compute derived features for model evaluation and analysis.

Features computed:
- Temporal decay weights (Dixon-Coles style)
- Rolling team statistics
- Home advantage metrics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class TeamStats:
    """Container for team statistics."""

    team: str
    matches_played: int
    goals_scored: float
    goals_conceded: float
    points: float
    home_advantage: float


class FeatureEngineer:
    """
    Compute features and statistics for goal prediction models.

    This class provides methods to:
    - Calculate temporal decay weights (for Dixon-Coles)
    - Compute rolling team statistics
    - Derive home advantage metrics
    - Build feature sets for enhanced models

    Example:
        >>> engineer = FeatureEngineer()
        >>> weights = engineer.compute_decay_weights(df, xi=0.005)
        >>> stats = engineer.compute_team_stats(df, window_days=365)
    """

    def __init__(self):
        """Initialize the feature engineer."""
        pass

    def compute_decay_weights(
        self,
        df: pd.DataFrame,
        xi: float = 0.005,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> np.ndarray:
        """
        Compute temporal decay weights for matches.

        Uses the Dixon-Coles exponential decay formula:
            w_i = exp(-xi * days_since_match)

        More recent matches get higher weights, allowing the model
        to adapt to changing team strengths over time.

        Args:
            df: DataFrame with 'date' column.
            xi: Decay rate parameter (higher = faster decay).
                Typical values: 0.001 to 0.01
            reference_date: Date to compute days from (default: max date in data).

        Returns:
            Array of weights for each match.
        """
        if reference_date is None:
            reference_date = df["date"].max()

        days_diff = (reference_date - df["date"]).dt.days.values
        weights = np.exp(-xi * days_diff)

        return weights

    def compute_team_stats(
        self,
        df: pd.DataFrame,
        window_days: Optional[int] = None,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> dict[str, TeamStats]:
        """
        Compute aggregate statistics for each team.

        Calculates:
        - Total matches played
        - Average goals scored/conceded
        - Total points (3 for win, 1 for draw)
        - Home advantage ratio

        Args:
            df: Prepared match DataFrame.
            window_days: Optional rolling window in days (None = all data).
            reference_date: Date to compute window from.

        Returns:
            Dictionary mapping team names to TeamStats.
        """
        if window_days and reference_date:
            cutoff = reference_date - pd.Timedelta(days=window_days)
            df = df[df["date"] >= cutoff]

        stats = {}

        # Get all teams
        teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

        for team in teams:
            # Home matches
            home_matches = df[df["home_team"] == team]
            home_goals_for = home_matches["home_goals"].sum()
            home_goals_against = home_matches["away_goals"].sum()
            home_wins = (home_matches["result"] == 0).sum()
            home_draws = (home_matches["result"] == 1).sum()
            n_home = len(home_matches)

            # Away matches
            away_matches = df[df["away_team"] == team]
            away_goals_for = away_matches["away_goals"].sum()
            away_goals_against = away_matches["home_goals"].sum()
            away_wins = (away_matches["result"] == 2).sum()
            away_draws = (away_matches["result"] == 1).sum()
            n_away = len(away_matches)

            total_matches = n_home + n_away
            if total_matches == 0:
                continue

            # Calculate stats
            total_scored = home_goals_for + away_goals_for
            total_conceded = home_goals_against + away_goals_against
            total_points = (home_wins + away_wins) * 3 + (home_draws + away_draws)

            # Home advantage: ratio of home win rate to away win rate
            home_win_rate = home_wins / n_home if n_home > 0 else 0
            away_win_rate = away_wins / n_away if n_away > 0 else 0
            home_advantage = (
                home_win_rate / away_win_rate if away_win_rate > 0 else home_win_rate
            )

            stats[team] = TeamStats(
                team=team,
                matches_played=total_matches,
                goals_scored=total_scored / total_matches,
                goals_conceded=total_conceded / total_matches,
                points=total_points / total_matches,
                home_advantage=home_advantage,
            )

        return stats

    def compute_league_averages(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Compute league-wide averages for normalization.

        Args:
            df: Prepared match DataFrame.

        Returns:
            Dictionary with league statistics.
        """
        total_matches = len(df)
        total_home_goals = df["home_goals"].sum()
        total_away_goals = df["away_goals"].sum()

        home_wins = (df["result"] == 0).sum()
        draws = (df["result"] == 1).sum()
        away_wins = (df["result"] == 2).sum()

        return {
            "avg_home_goals": total_home_goals / total_matches,
            "avg_away_goals": total_away_goals / total_matches,
            "avg_total_goals": (total_home_goals + total_away_goals) / total_matches,
            "home_win_rate": home_wins / total_matches,
            "draw_rate": draws / total_matches,
            "away_win_rate": away_wins / total_matches,
        }

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        windows: list[int] = [5, 10, 20],
    ) -> pd.DataFrame:
        """
        Add rolling average features to the dataset.

        Computes rolling averages for goals scored/conceded
        over multiple window sizes for each team.

        Note: This is for demonstration purposes. The Poisson and
        Dixon-Coles models estimate team strengths internally and
        don't require these explicit features.

        Args:
            df: Prepared match DataFrame.
            windows: List of rolling window sizes (in matches).

        Returns:
            DataFrame with additional rolling feature columns.
        """
        df = df.copy()

        # Create team-centric view for rolling calculations
        for window in windows:
            # Initialize columns
            df[f"home_scored_last_{window}"] = np.nan
            df[f"home_conceded_last_{window}"] = np.nan
            df[f"away_scored_last_{window}"] = np.nan
            df[f"away_conceded_last_{window}"] = np.nan

        # This is a simplified implementation for demonstration
        # A production system would use more efficient lookups
        teams = set(df["home_team"].unique()) | set(df["away_team"].unique())

        for team in teams:
            # Get all matches for this team in chronological order
            is_home = df["home_team"] == team
            is_away = df["away_team"] == team
            team_mask = is_home | is_away

            team_matches = df[team_mask].copy()
            team_matches["goals_for"] = np.where(
                team_matches["home_team"] == team,
                team_matches["home_goals"],
                team_matches["away_goals"],
            )
            team_matches["goals_against"] = np.where(
                team_matches["home_team"] == team,
                team_matches["away_goals"],
                team_matches["home_goals"],
            )

            for window in windows:
                # Rolling means (shift by 1 to avoid data leakage)
                scored_roll = (
                    team_matches["goals_for"].rolling(window, min_periods=1).mean().shift(1)
                )
                conceded_roll = (
                    team_matches["goals_against"].rolling(window, min_periods=1).mean().shift(1)
                )

                # Update home matches
                home_idx = df.index[is_home]
                team_home_idx = team_matches.index[team_matches["home_team"] == team]
                mapping = dict(zip(team_home_idx, scored_roll.loc[team_home_idx]))
                df.loc[home_idx, f"home_scored_last_{window}"] = df.loc[home_idx].index.map(
                    mapping
                )

                mapping = dict(zip(team_home_idx, conceded_roll.loc[team_home_idx]))
                df.loc[home_idx, f"home_conceded_last_{window}"] = df.loc[home_idx].index.map(
                    mapping
                )

                # Update away matches
                away_idx = df.index[is_away]
                team_away_idx = team_matches.index[team_matches["away_team"] == team]
                mapping = dict(zip(team_away_idx, scored_roll.loc[team_away_idx]))
                df.loc[away_idx, f"away_scored_last_{window}"] = df.loc[away_idx].index.map(
                    mapping
                )

                mapping = dict(zip(team_away_idx, conceded_roll.loc[team_away_idx]))
                df.loc[away_idx, f"away_conceded_last_{window}"] = df.loc[away_idx].index.map(
                    mapping
                )

        return df


def compute_xi_from_halflife(halflife_days: int) -> float:
    """
    Convert a halflife (in days) to the xi decay parameter.

    Halflife = the number of days until a match has half its original weight.

    Formula: xi = ln(2) / halflife

    Args:
        halflife_days: Desired halflife in days.

    Returns:
        Corresponding xi parameter.

    Example:
        >>> xi = compute_xi_from_halflife(365)  # 1 year halflife
        >>> xi
        0.0019...
    """
    return np.log(2) / halflife_days


def compute_halflife_from_xi(xi: float) -> float:
    """
    Convert xi decay parameter to halflife in days.

    Args:
        xi: Decay rate parameter.

    Returns:
        Halflife in days.

    Example:
        >>> halflife = compute_halflife_from_xi(0.005)
        >>> halflife
        138.6...
    """
    return np.log(2) / xi
