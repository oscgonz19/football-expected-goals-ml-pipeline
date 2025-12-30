#!/usr/bin/env python3
"""
Run Demo Script

Complete pipeline demonstration for goal prediction models.
This script shows the full workflow from data loading to prediction.

Usage:
    python -m portfolio_demo.run_demo

Or with custom data:
    python -m portfolio_demo.run_demo --data path/to/matches.csv
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


def create_sample_data() -> pd.DataFrame:
    """
    Create sample match data for demonstration.

    This generates a small synthetic dataset that mimics real football
    league data structure. For production use, replace with actual
    historical match data.

    Returns:
        DataFrame with sample match records.
    """
    # Sample data simulating 2 seasons of a small league
    data = {
        "date": [
            # Season 2022-2023
            "2022-08-13", "2022-08-13", "2022-08-14", "2022-08-20", "2022-08-20",
            "2022-08-21", "2022-08-27", "2022-08-27", "2022-09-03", "2022-09-03",
            "2022-09-10", "2022-09-10", "2022-09-17", "2022-09-17", "2022-09-24",
            "2022-10-01", "2022-10-08", "2022-10-15", "2022-10-22", "2022-10-29",
            "2022-11-05", "2022-11-12", "2022-11-26", "2022-12-03", "2022-12-10",
            "2022-12-17", "2022-12-26", "2022-12-31", "2023-01-07", "2023-01-14",
            "2023-01-21", "2023-01-28", "2023-02-04", "2023-02-11", "2023-02-18",
            "2023-02-25", "2023-03-04", "2023-03-11", "2023-03-18", "2023-04-01",
            # Season 2023-2024
            "2023-08-12", "2023-08-12", "2023-08-13", "2023-08-19", "2023-08-19",
            "2023-08-20", "2023-08-26", "2023-08-26", "2023-09-02", "2023-09-02",
            "2023-09-16", "2023-09-16", "2023-09-23", "2023-09-23", "2023-09-30",
            "2023-10-07", "2023-10-21", "2023-10-28", "2023-11-04", "2023-11-11",
            "2023-11-25", "2023-12-02", "2023-12-09", "2023-12-16", "2023-12-23",
            "2023-12-26", "2023-12-30", "2024-01-06", "2024-01-13", "2024-01-20",
            "2024-01-27", "2024-02-03", "2024-02-10", "2024-02-17", "2024-02-24",
            "2024-03-02", "2024-03-09", "2024-03-16", "2024-03-30", "2024-04-06",
        ],
        "season": (
            ["2022-2023"] * 40 +
            ["2023-2024"] * 40
        ),
        "home_team": [
            # Season 2022-2023
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
            # Season 2023-2024
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
            "Team A", "Team B", "Team C", "Team D", "Team A", "Team B", "Team C", "Team D",
        ],
        "away_team": [
            # Season 2022-2023
            "Team B", "Team C", "Team D", "Team A", "Team C", "Team D", "Team A", "Team B",
            "Team D", "Team A", "Team B", "Team C", "Team B", "Team C", "Team D", "Team A",
            "Team C", "Team D", "Team A", "Team B", "Team D", "Team A", "Team B", "Team C",
            "Team B", "Team C", "Team D", "Team A", "Team C", "Team D", "Team A", "Team B",
            "Team D", "Team A", "Team B", "Team C", "Team B", "Team C", "Team D", "Team A",
            # Season 2023-2024
            "Team B", "Team C", "Team D", "Team A", "Team C", "Team D", "Team A", "Team B",
            "Team D", "Team A", "Team B", "Team C", "Team B", "Team C", "Team D", "Team A",
            "Team C", "Team D", "Team A", "Team B", "Team D", "Team A", "Team B", "Team C",
            "Team B", "Team C", "Team D", "Team A", "Team C", "Team D", "Team A", "Team B",
            "Team D", "Team A", "Team B", "Team C", "Team B", "Team C", "Team D", "Team A",
        ],
        "home_goals": [
            # Season 2022-2023 (Team A stronger at home, Team C weaker)
            3, 1, 0, 1, 2, 2, 1, 2, 2, 1, 0, 1, 3, 0, 1, 2,
            2, 1, 0, 1, 1, 2, 1, 0, 2, 1, 0, 2, 3, 1, 1, 1,
            2, 0, 0, 1, 2, 2, 1, 2,
            # Season 2023-2024
            2, 2, 1, 0, 3, 1, 0, 2, 2, 1, 1, 1, 2, 0, 0, 1,
            3, 1, 1, 2, 1, 1, 0, 1, 2, 2, 1, 2, 2, 0, 0, 1,
            3, 1, 1, 2, 2, 1, 0, 1,
        ],
        "away_goals": [
            # Season 2022-2023
            1, 1, 2, 1, 0, 1, 2, 1, 1, 2, 1, 0, 0, 1, 1, 0,
            1, 2, 1, 0, 2, 0, 0, 2, 0, 0, 1, 1, 1, 2, 0, 1,
            0, 3, 2, 0, 1, 1, 2, 1,
            # Season 2023-2024
            0, 1, 0, 1, 1, 2, 1, 1, 0, 2, 0, 0, 1, 3, 1, 0,
            0, 1, 2, 1, 2, 0, 1, 1, 1, 0, 0, 0, 1, 1, 2, 2,
            0, 2, 1, 0, 0, 0, 1, 1,
        ],
    }

    return pd.DataFrame(data)


def run_pipeline(data_path: str | Path | None = None, optimize: bool = True) -> None:
    """
    Run the complete prediction pipeline.

    Steps:
    1. Load and validate data
    2. Prepare features
    3. Optimize hyperparameters (optional)
    4. Train model
    5. Evaluate on test set
    6. Generate sample predictions

    Args:
        data_path: Path to CSV file (uses sample data if None).
        optimize: Whether to run hyperparameter optimization.
    """
    try:
        from .data_loader import MatchDataLoader
        from .feature_engineering import FeatureEngineer
        from .model_training import ModelTrainer, OptimizationConfig, TrainingConfig
        from .model_evaluation import ModelEvaluator
        from .predict import MatchPredictor, format_prediction
    except ImportError:
        from data_loader import MatchDataLoader
        from feature_engineering import FeatureEngineer
        from model_training import ModelTrainer, OptimizationConfig, TrainingConfig
        from model_evaluation import ModelEvaluator
        from predict import MatchPredictor, format_prediction

    print("=" * 60)
    print("GOAL PREDICTION PIPELINE DEMO")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load Data
    # =========================================================================
    print("\n[1/6] Loading Data...")

    loader = MatchDataLoader()

    if data_path:
        df = loader.load(data_path)
        print(f"  Loaded {len(df)} matches from {data_path}")
    else:
        df = create_sample_data()
        print(f"  Created {len(df)} sample matches (demo data)")

    # Validate
    validation = loader.validate(df)
    if not validation.is_valid:
        print(f"  ERROR: {validation.errors}")
        return

    if validation.warnings:
        for w in validation.warnings:
            print(f"  Warning: {w}")

    # Prepare
    df = loader.prepare(df)
    teams = loader.get_teams(df)
    seasons = loader.get_seasons(df)

    print(f"  Teams: {teams}")
    print(f"  Seasons: {seasons}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # =========================================================================
    # Step 2: Feature Engineering
    # =========================================================================
    print("\n[2/6] Computing Features...")

    engineer = FeatureEngineer()

    # Compute league averages
    league_stats = engineer.compute_league_averages(df)
    print(f"  Average home goals: {league_stats['avg_home_goals']:.2f}")
    print(f"  Average away goals: {league_stats['avg_away_goals']:.2f}")
    print(f"  Home win rate: {league_stats['home_win_rate']:.1%}")
    print(f"  Draw rate: {league_stats['draw_rate']:.1%}")
    print(f"  Away win rate: {league_stats['away_win_rate']:.1%}")

    # Compute team statistics
    team_stats = engineer.compute_team_stats(df)
    print("\n  Team Statistics:")
    for team, stats in sorted(team_stats.items()):
        print(
            f"    {team}: {stats.matches_played} matches, "
            f"{stats.goals_scored:.1f} GF/game, "
            f"{stats.goals_conceded:.1f} GA/game"
        )

    # =========================================================================
    # Step 3: Model Training
    # =========================================================================
    trainer = ModelTrainer(output_dir="portfolio_demo/models")

    if optimize:
        print("\n[3/6] Optimizing Hyperparameters...")
        print("  Running Optuna optimization (this may take a moment)...")

        opt_config = OptimizationConfig(
            n_trials=20,  # Reduced for demo speed
            metric="rps",
            xi_range=(0.001, 0.01),
            window_years_range=(2, 4),
        )

        model, result, opt_stats = trainer.optimize(
            df,
            config=opt_config,
            model_type="dixon_coles",
            verbose=False,
        )

        print(f"  Best xi: {opt_stats['best_params']['xi']:.6f}")
        print(f"  Best window: {opt_stats['best_params']['window_years']} years")
        print(f"  Best RPS: {opt_stats['best_score']:.4f}")

    else:
        print("\n[3/6] Training Model (fixed parameters)...")

        config = TrainingConfig(
            model_type="dixon_coles",
            xi=0.005,
            window_years=3,
        )

        model, result = trainer.train(df, config)
        opt_stats = None

    print(f"  Model type: {result.model_type}")
    print(f"  Training samples: {result.train_samples}")
    print(f"  Teams in model: {len(result.teams)}")

    # =========================================================================
    # Step 4: Save Model
    # =========================================================================
    print("\n[4/6] Saving Model...")

    model_path = trainer.save(model, result, name="demo_model")
    print(f"  Saved to: {model_path}")

    # =========================================================================
    # Step 5: Evaluate Model
    # =========================================================================
    print("\n[5/6] Evaluating Model...")

    evaluator = ModelEvaluator()

    # Use last season as test set
    test_season = seasons[-1]
    df_test = df[df["season"] == test_season]

    print(f"  Test season: {test_season} ({len(df_test)} matches)")

    eval_results = evaluator.evaluate(model, df_test)

    for metric, result_obj in eval_results.items():
        print(f"  {metric.upper()}: {result_obj.score:.4f} (+/- {result_obj.score_std:.4f})")

    # Calibration
    calibration = evaluator.evaluate_calibration(model, df_test, n_bins=5)
    if len(calibration) > 0:
        print("\n  Calibration (Home Win Probability):")
        for _, row in calibration.iterrows():
            print(
                f"    Predicted: {row['predicted_prob']:.0%} -> "
                f"Actual: {row['actual_freq']:.0%} "
                f"(n={int(row['count'])})"
            )

    # =========================================================================
    # Step 6: Generate Predictions
    # =========================================================================
    print("\n[6/6] Generating Sample Predictions...")

    predictor = MatchPredictor(model, model_type="dixon_coles")

    # Generate predictions using actual teams from the model
    available_teams = predictor.get_teams()
    if len(available_teams) >= 4:
        sample_matchups = [
            (available_teams[0], available_teams[1]),
            (available_teams[2], available_teams[3]),
            (available_teams[1], available_teams[3]),
        ]
    else:
        sample_matchups = [(available_teams[0], available_teams[1])]

    for home, away in sample_matchups:
        try:
            prediction = predictor.predict(home, away, date.today())
            print(format_prediction(prediction))
        except ValueError as e:
            print(f"\nCould not predict {home} vs {away}: {e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nModel saved to: {model_path}")
    print("You can load and use this model for predictions:")
    print(f"""
    from portfolio_demo import ModelTrainer, MatchPredictor

    trainer = ModelTrainer()
    model, metadata = trainer.load("{model_path}")
    predictor = MatchPredictor(model)
    prediction = predictor.predict("{available_teams[0]}", "{available_teams[1]}")
    """)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run the goal prediction pipeline demo"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV file with match data (uses sample data if not provided)",
    )
    parser.add_argument(
        "--no-optimize",
        action="store_true",
        help="Skip hyperparameter optimization (use fixed parameters)",
    )

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data,
        optimize=not args.no_optimize,
    )


if __name__ == "__main__":
    main()
