"""
Data Loader Module

Handles loading and validation of football match data from CSV files.
Inspired by production patterns for robust data ingestion.

Expected CSV columns:
    - date: Match date (YYYY-MM-DD)
    - season: Season identifier (e.g., "2023-2024")
    - home_team: Home team name
    - away_team: Away team name
    - home_goals: Goals scored by home team
    - away_goals: Goals scored by away team
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class ValidationResult:
    """Result of data validation."""

    is_valid: bool
    errors: list[str]
    warnings: list[str]


class MatchDataLoader:
    """
    Load and validate football match data from CSV files.

    This class provides methods to:
    - Load match data from CSV
    - Validate required columns and data types
    - Clean and normalize team names
    - Filter by date range or seasons

    Example:
        >>> loader = MatchDataLoader()
        >>> df = loader.load("matches.csv")
        >>> validation = loader.validate(df)
        >>> if validation.is_valid:
        ...     df_clean = loader.prepare(df)
    """

    REQUIRED_COLUMNS = [
        "date",
        "season",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
    ]

    def __init__(self, date_format: str = "%Y-%m-%d"):
        """
        Initialize the data loader.

        Args:
            date_format: Expected date format in CSV files.
        """
        self.date_format = date_format

    def load(self, filepath: str | Path) -> pd.DataFrame:
        """
        Load match data from a CSV file.

        Args:
            filepath: Path to the CSV file.

        Returns:
            DataFrame with raw match data.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If required columns are missing.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Check required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate match data for model training.

        Checks:
        - No empty dataset
        - At least 2 unique teams per side
        - Non-negative goal values
        - No duplicate matches
        - At least 2 seasons for proper evaluation

        Args:
            df: DataFrame to validate.

        Returns:
            ValidationResult with status and any errors/warnings.
        """
        errors = []
        warnings = []

        # Check for empty dataset
        if len(df) == 0:
            errors.append("Dataset is empty")
            return ValidationResult(False, errors, warnings)

        # Check minimum teams
        n_home_teams = df["home_team"].nunique()
        n_away_teams = df["away_team"].nunique()
        if n_home_teams < 2 or n_away_teams < 2:
            errors.append("Need at least 2 unique teams per side")

        # Check goals are non-negative integers
        if df["home_goals"].min() < 0 or df["away_goals"].min() < 0:
            errors.append("Goals cannot be negative")

        if df["home_goals"].isna().any() or df["away_goals"].isna().any():
            errors.append("Goals contain missing values")

        # Check for duplicate matches
        dup_cols = ["date", "home_team", "away_team"]
        duplicates = df.duplicated(subset=dup_cols, keep=False)
        if duplicates.any():
            n_dups = duplicates.sum()
            errors.append(f"Found {n_dups} duplicate matches")

        # Check seasons
        n_seasons = df["season"].nunique()
        if n_seasons < 2:
            warnings.append("Only 1 season found - evaluation may be limited")

        # Check date parsing
        try:
            pd.to_datetime(df["date"], format=self.date_format)
        except Exception:
            errors.append(f"Date column cannot be parsed with format {self.date_format}")

        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings)

    def prepare(
        self,
        df: pd.DataFrame,
        min_date: Optional[str] = None,
        max_date: Optional[str] = None,
        seasons: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """
        Prepare and clean match data for model training.

        Operations:
        - Parse dates and sort chronologically
        - Normalize team names (lowercase, strip whitespace)
        - Encode match results (0=Home, 1=Draw, 2=Away)
        - Filter by date range or seasons if specified

        Args:
            df: Raw DataFrame to prepare.
            min_date: Optional minimum date filter.
            max_date: Optional maximum date filter.
            seasons: Optional list of seasons to include.

        Returns:
            Cleaned and sorted DataFrame ready for training.
        """
        df = df.copy()

        # Parse dates
        df["date"] = pd.to_datetime(df["date"], format=self.date_format)

        # Normalize team names
        df["home_team"] = df["home_team"].str.lower().str.strip()
        df["away_team"] = df["away_team"].str.lower().str.strip()

        # Ensure goals are integers
        df["home_goals"] = df["home_goals"].astype(int)
        df["away_goals"] = df["away_goals"].astype(int)

        # Encode full-time result: 0=Home Win, 1=Draw, 2=Away Win
        df["result"] = 1  # Default to draw
        df.loc[df["home_goals"] > df["away_goals"], "result"] = 0
        df.loc[df["home_goals"] < df["away_goals"], "result"] = 2

        # Apply filters
        if min_date:
            df = df[df["date"] >= pd.to_datetime(min_date)]
        if max_date:
            df = df[df["date"] <= pd.to_datetime(max_date)]
        if seasons:
            df = df[df["season"].isin(seasons)]

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        return df

    def get_teams(self, df: pd.DataFrame) -> list[str]:
        """
        Extract unique team names from the dataset.

        Args:
            df: Prepared DataFrame.

        Returns:
            Sorted list of unique team names.
        """
        home_teams = set(df["home_team"].unique())
        away_teams = set(df["away_team"].unique())
        return sorted(home_teams | away_teams)

    def get_seasons(self, df: pd.DataFrame) -> list[str]:
        """
        Extract unique seasons from the dataset.

        Args:
            df: DataFrame with season column.

        Returns:
            List of seasons in chronological order.
        """
        return df["season"].unique().tolist()


# Utility function for quick loading
def load_matches(filepath: str | Path) -> pd.DataFrame:
    """
    Convenience function to load and prepare match data in one step.

    Args:
        filepath: Path to CSV file.

    Returns:
        Prepared DataFrame ready for model training.

    Raises:
        ValueError: If data validation fails.
    """
    loader = MatchDataLoader()
    df = loader.load(filepath)
    validation = loader.validate(df)

    if not validation.is_valid:
        raise ValueError(f"Data validation failed: {validation.errors}")

    if validation.warnings:
        for warning in validation.warnings:
            print(f"Warning: {warning}")

    return loader.prepare(df)
