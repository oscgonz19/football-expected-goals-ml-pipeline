"""
Visualizations Module

Provides plotting functions for data exploration, model evaluation,
and prediction analysis. Designed for storytelling and documentation.

Requires: matplotlib, seaborn
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Lazy imports for optional dependencies
def _get_plt():
    import matplotlib.pyplot as plt
    return plt

def _get_sns():
    import seaborn as sns
    return sns


# =============================================================================
# Style Configuration
# =============================================================================

COLORS = {
    "home": "#2ecc71",      # Green
    "draw": "#f39c12",      # Orange
    "away": "#e74c3c",      # Red
    "primary": "#3498db",   # Blue
    "secondary": "#9b59b6", # Purple
    "neutral": "#95a5a6",   # Gray
}

def set_style():
    """Set consistent plot style."""
    plt = _get_plt()
    sns = _get_sns()

    plt.style.use("seaborn-v0_8-whitegrid")
    sns.set_palette("husl")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12


# =============================================================================
# Data Exploration Plots
# =============================================================================

def plot_goals_distribution(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot distribution of home and away goals.

    Shows histogram comparing home vs away goal frequencies,
    highlighting the home advantage effect.
    """
    plt = _get_plt()
    sns = _get_sns()
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Home goals distribution
    ax1 = axes[0]
    home_goals = df["home_goals"].value_counts().sort_index()
    ax1.bar(home_goals.index, home_goals.values, color=COLORS["home"], alpha=0.8, edgecolor="white")
    ax1.set_xlabel("Goals")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Home Goals Distribution\n(Mean: {df['home_goals'].mean():.2f})")
    ax1.set_xticks(range(0, min(8, home_goals.index.max() + 1)))

    # Away goals distribution
    ax2 = axes[1]
    away_goals = df["away_goals"].value_counts().sort_index()
    ax2.bar(away_goals.index, away_goals.values, color=COLORS["away"], alpha=0.8, edgecolor="white")
    ax2.set_xlabel("Goals")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Away Goals Distribution\n(Mean: {df['away_goals'].mean():.2f})")
    ax2.set_xticks(range(0, min(8, away_goals.index.max() + 1)))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_results_by_season(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot match results breakdown by season.

    Stacked bar chart showing Home Win / Draw / Away Win percentages.
    """
    plt = _get_plt()
    set_style()

    # Calculate result percentages by season
    results_by_season = df.groupby("season")["result"].value_counts(normalize=True).unstack(fill_value=0)
    results_by_season.columns = ["Home Win", "Draw", "Away Win"]
    results_by_season = results_by_season * 100

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(results_by_season))
    width = 0.6

    bottom = np.zeros(len(results_by_season))
    for col, color in zip(["Home Win", "Draw", "Away Win"],
                          [COLORS["home"], COLORS["draw"], COLORS["away"]]):
        ax.bar(x, results_by_season[col], width, label=col, bottom=bottom, color=color, edgecolor="white")
        bottom += results_by_season[col].values

    ax.set_xlabel("Season")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Match Results by Season")
    ax.set_xticks(x)
    ax.set_xticklabels(results_by_season.index, rotation=45, ha="right")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)

    # Add percentage labels
    for i, season in enumerate(results_by_season.index):
        home_pct = results_by_season.loc[season, "Home Win"]
        ax.text(i, home_pct / 2, f"{home_pct:.0f}%", ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_team_strength(
    df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot team attack vs defense strength.

    Scatter plot with goals scored (attack) vs goals conceded (defense).
    """
    plt = _get_plt()
    set_style()

    # Calculate team stats
    teams_home = df.groupby("home_team").agg({
        "home_goals": "sum",
        "away_goals": "sum"
    }).rename(columns={"home_goals": "scored", "away_goals": "conceded"})

    teams_away = df.groupby("away_team").agg({
        "away_goals": "sum",
        "home_goals": "sum"
    }).rename(columns={"away_goals": "scored", "home_goals": "conceded"})

    team_stats = teams_home.add(teams_away, fill_value=0)

    # Games played
    games_home = df["home_team"].value_counts()
    games_away = df["away_team"].value_counts()
    games = games_home.add(games_away, fill_value=0)

    team_stats["games"] = games
    team_stats["scored_per_game"] = team_stats["scored"] / team_stats["games"]
    team_stats["conceded_per_game"] = team_stats["conceded"] / team_stats["games"]
    team_stats["goal_diff"] = team_stats["scored_per_game"] - team_stats["conceded_per_game"]

    # Sort by goal difference and take top/bottom
    team_stats = team_stats.sort_values("goal_diff", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create scatter
    scatter = ax.scatter(
        team_stats["scored_per_game"],
        team_stats["conceded_per_game"],
        s=team_stats["games"] * 3,
        c=team_stats["goal_diff"],
        cmap="RdYlGn",
        alpha=0.7,
        edgecolors="white",
        linewidth=1.5
    )

    # Add team labels for top teams
    for team in team_stats.head(top_n).index:
        ax.annotate(
            team.title(),
            (team_stats.loc[team, "scored_per_game"],
             team_stats.loc[team, "conceded_per_game"]),
            fontsize=9,
            alpha=0.8
        )

    # Add diagonal line (equal attack/defense)
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, 'k--', alpha=0.3, zorder=0)

    ax.set_xlabel("Goals Scored per Game (Attack)")
    ax.set_ylabel("Goals Conceded per Game (Defense)")
    ax.set_title("Team Strength: Attack vs Defense\n(Size = games played, Color = goal difference)")

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Goal Difference per Game")

    # Add quadrant labels
    ax.axhline(team_stats["conceded_per_game"].mean(), color="gray", linestyle=":", alpha=0.5)
    ax.axvline(team_stats["scored_per_game"].mean(), color="gray", linestyle=":", alpha=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# Model Evaluation Plots
# =============================================================================

def plot_calibration(
    calibration_df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot calibration curve.

    Shows predicted probability vs actual frequency with perfect calibration line.
    """
    plt = _get_plt()
    set_style()

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.7)

    # Actual calibration
    ax.scatter(
        calibration_df["predicted_prob"],
        calibration_df["actual_freq"],
        s=calibration_df["count"] * 2,
        c=COLORS["primary"],
        alpha=0.7,
        edgecolors="white",
        linewidth=1.5,
        label="Model"
    )

    ax.plot(
        calibration_df["predicted_prob"],
        calibration_df["actual_freq"],
        color=COLORS["primary"],
        alpha=0.5
    )

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Actual Frequency")
    ax.set_title("Calibration Plot\n(Size = sample count)")
    ax.legend()
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_rps_distribution(
    predictions: list[list[float]],
    actuals: list[int],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot distribution of RPS scores.

    Histogram showing how prediction quality is distributed.
    """
    plt = _get_plt()
    sns = _get_sns()
    set_style()

    from model_evaluation import compute_rps

    scores = [compute_rps(p, a) for p, a in zip(predictions, actuals)]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(scores, bins=30, color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax.axvline(np.mean(scores), color=COLORS["away"], linestyle="--",
               linewidth=2, label=f"Mean RPS: {np.mean(scores):.4f}")
    ax.axvline(np.median(scores), color=COLORS["home"], linestyle="--",
               linewidth=2, label=f"Median RPS: {np.median(scores):.4f}")

    ax.set_xlabel("RPS Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of RPS Scores\n(Lower is better)")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_prediction_vs_actual(
    predictions: list[list[float]],
    actuals: list[int],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot confusion-style matrix of predictions vs actuals.

    Shows how often each predicted outcome matches actual outcome.
    """
    plt = _get_plt()
    sns = _get_sns()
    set_style()

    # Get predicted outcomes
    predicted_outcomes = [np.argmax(p) for p in predictions]

    # Create confusion matrix
    matrix = np.zeros((3, 3))
    for pred, actual in zip(predicted_outcomes, actuals):
        matrix[actual, pred] += 1

    # Normalize by row
    matrix_pct = matrix / matrix.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 7))

    labels = ["Home Win", "Draw", "Away Win"]

    sns.heatmap(
        matrix_pct,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Percentage (%)"}
    )

    ax.set_xlabel("Predicted Outcome")
    ax.set_ylabel("Actual Outcome")
    ax.set_title("Prediction Accuracy Matrix\n(Row-normalized percentages)")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# Prediction Visualization
# =============================================================================

def plot_probability_grid(
    grid: np.ndarray,
    home_team: str,
    away_team: str,
    max_goals: int = 6,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot probability heatmap for exact scores.

    Shows P(home_goals, away_goals) as a heatmap.
    """
    plt = _get_plt()
    sns = _get_sns()
    set_style()

    # Truncate grid
    grid_display = grid[:max_goals, :max_goals]

    fig, ax = plt.subplots(figsize=(9, 8))

    sns.heatmap(
        grid_display * 100,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Probability (%)"},
        xticklabels=range(max_goals),
        yticklabels=range(max_goals)
    )

    ax.set_xlabel(f"{away_team.title()} Goals")
    ax.set_ylabel(f"{home_team.title()} Goals")
    ax.set_title(f"Exact Score Probabilities\n{home_team.title()} vs {away_team.title()}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def plot_match_prediction(
    prediction,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Plot comprehensive match prediction visualization.

    Multi-panel plot showing 1X2, xG, BTTS, and Totals.
    """
    plt = _get_plt()
    set_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1X2 Probabilities
    ax1 = axes[0, 0]
    probs = [
        prediction.probabilities.home_win,
        prediction.probabilities.draw,
        prediction.probabilities.away_win
    ]
    labels = ["Home\nWin", "Draw", "Away\nWin"]
    colors = [COLORS["home"], COLORS["draw"], COLORS["away"]]
    bars = ax1.bar(labels, [p * 100 for p in probs], color=colors, edgecolor="white", linewidth=2)
    ax1.set_ylabel("Probability (%)")
    ax1.set_title("1X2 Probabilities")
    ax1.set_ylim(0, 100)
    for bar, pct in zip(bars, probs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{pct:.1%}", ha="center", fontsize=12, fontweight="bold")

    # Expected Goals
    ax2 = axes[0, 1]
    xg = [prediction.expected_goals.home, prediction.expected_goals.away]
    teams = [prediction.home_team.title(), prediction.away_team.title()]
    colors_xg = [COLORS["home"], COLORS["away"]]
    bars = ax2.barh(teams, xg, color=colors_xg, edgecolor="white", linewidth=2)
    ax2.set_xlabel("Expected Goals")
    ax2.set_title("Expected Goals (xG)")
    for bar, val in zip(bars, xg):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=12, fontweight="bold")

    # BTTS
    ax3 = axes[1, 0]
    if prediction.btts_yes is not None:
        btts = [prediction.btts_yes, prediction.btts_no]
        labels_btts = ["Yes", "No"]
        colors_btts = [COLORS["home"], COLORS["neutral"]]
        wedges, texts, autotexts = ax3.pie(
            btts, labels=labels_btts, colors=colors_btts,
            autopct="%1.1f%%", startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 2}
        )
        ax3.set_title("Both Teams To Score (BTTS)")
    else:
        ax3.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=20)
        ax3.set_title("BTTS")

    # Totals
    ax4 = axes[1, 1]
    if prediction.totals:
        lines = [t.line for t in prediction.totals]
        overs = [t.over * 100 for t in prediction.totals]
        unders = [t.under * 100 for t in prediction.totals]

        x = np.arange(len(lines))
        width = 0.35

        ax4.bar(x - width/2, overs, width, label="Over", color=COLORS["home"], edgecolor="white")
        ax4.bar(x + width/2, unders, width, label="Under", color=COLORS["away"], edgecolor="white")
        ax4.set_xlabel("Goal Line")
        ax4.set_ylabel("Probability (%)")
        ax4.set_title("Over/Under Totals")
        ax4.set_xticks(x)
        ax4.set_xticklabels([f"{l:.1f}" for l in lines])
        ax4.legend()
        ax4.set_ylim(0, 100)

    plt.suptitle(
        f"{prediction.home_team.title()} vs {prediction.away_team.title()}\n{prediction.match_date}",
        fontsize=14,
        fontweight="bold"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


# =============================================================================
# Summary Dashboard
# =============================================================================

def generate_all_plots(
    df: pd.DataFrame,
    model,
    evaluator,
    output_dir: str = "plots",
    show: bool = False,
) -> list[str]:
    """
    Generate all visualization plots and save to directory.

    Args:
        df: Match data DataFrame
        model: Trained model
        evaluator: ModelEvaluator instance
        output_dir: Directory to save plots
        show: Whether to display plots

    Returns:
        List of saved file paths
    """
    plt = _get_plt()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = []

    # 1. Goals distribution
    path = str(output_path / "01_goals_distribution.png")
    plot_goals_distribution(df, save_path=path, show=show)
    saved_files.append(path)
    print(f"  Saved: {path}")

    # 2. Results by season
    path = str(output_path / "02_results_by_season.png")
    plot_results_by_season(df, save_path=path, show=show)
    saved_files.append(path)
    print(f"  Saved: {path}")

    # 3. Team strength
    path = str(output_path / "03_team_strength.png")
    plot_team_strength(df, save_path=path, show=show)
    saved_files.append(path)
    print(f"  Saved: {path}")

    # 4. Calibration (if evaluator provided)
    if evaluator is not None:
        test_season = df["season"].iloc[-1]
        df_test = df[df["season"] == test_season]

        try:
            calibration = evaluator.evaluate_calibration(model, df_test)
            if len(calibration) > 0:
                path = str(output_path / "04_calibration.png")
                plot_calibration(calibration, save_path=path, show=show)
                saved_files.append(path)
                print(f"  Saved: {path}")
        except Exception as e:
            print(f"  Skipped calibration plot: {e}")

    # 5. Sample prediction visualization
    try:
        from predict import MatchPredictor

        predictor = MatchPredictor(model)
        teams = predictor.get_teams()

        if len(teams) >= 2:
            prediction = predictor.predict(teams[0], teams[1])

            path = str(output_path / "05_match_prediction.png")
            plot_match_prediction(prediction, save_path=path, show=show)
            saved_files.append(path)
            print(f"  Saved: {path}")

            # Probability grid
            path = str(output_path / "06_probability_grid.png")
            plot_probability_grid(
                predictor._last_grid,
                teams[0], teams[1],
                save_path=path,
                show=show
            )
            saved_files.append(path)
            print(f"  Saved: {path}")
    except Exception as e:
        print(f"  Skipped prediction plots: {e}")

    return saved_files


if __name__ == "__main__":
    # Quick test
    print("Visualizations module loaded successfully!")
    print("Available functions:")
    print("  - plot_goals_distribution(df)")
    print("  - plot_results_by_season(df)")
    print("  - plot_team_strength(df)")
    print("  - plot_calibration(calibration_df)")
    print("  - plot_probability_grid(grid, home, away)")
    print("  - plot_match_prediction(prediction)")
    print("  - generate_all_plots(df, model, evaluator)")
