# Football Match Prediction Pipeline

[![Python 3.13+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**An end-to-end machine learning pipeline for predicting football match outcomes using Poisson and Dixon-Coles statistical models.**

<p align="center">
  <img src="plots/05_match_prediction.png" width="70%" />
</p>

---

## The Problem

Predicting football match outcomes is a classic challenge in sports analytics. The goal is to estimate the probability of each possible result (Home Win, Draw, Away Win) along with derived markets like expected goals, both teams to score, and over/under totals.

This project demonstrates a complete ML pipeline that:
- Loads and validates historical match data
- Engineers temporal decay features
- Optimizes hyperparameters using Optuna
- Trains Dixon-Coles goal models
- Evaluates with proper scoring rules
- Generates probabilistic predictions

> **Case Study**: For the full methodology, statistical foundations, and real-world analysis, see the [complete case study](https://github.com/oscgonz19/football-probabilities-ml-poisson-case-study).

---

## Quick Start

```bash
# Clone and install
git clone git@github.com:oscgonz19/football-expected-goals-ml-pipeline.git
cd football-expected-goals-ml-pipeline
pip install -r requirements.txt

# Run the full pipeline with Eredivisie data
python run_demo.py --data data/eredivisie_sample.csv

# Quick run (skip hyperparameter optimization)
python run_demo.py --data data/eredivisie_sample.csv --no-optimize
```

---

## The Data

The pipeline includes real match data from the **Dutch Eredivisie** (2021-2025):

| Stat | Value |
|------|-------|
| Matches | 1,242 |
| Seasons | 4 (2021-2025) |
| Teams | 23 |
| Home Win Rate | 44.4% |
| Draw Rate | 23.4% |
| Away Win Rate | 32.1% |

<p align="center">
  <img src="plots/01_goals_distribution.png" width="80%" />
</p>

The data shows the typical **home advantage effect**: home teams score an average of **1.70 goals** compared to **1.34 goals** for away teams.

### Results by Season

<p align="center">
  <img src="plots/02_results_by_season.png" width="70%" />
</p>

---

## Team Analysis

Before training, we analyze team strengths by comparing attack (goals scored) vs defense (goals conceded):

<p align="center">
  <img src="plots/03_team_strength.png" width="80%" />
</p>

**Key Insights:**
- **PSV Eindhoven** leads with 2.9 goals/game and only 1.0 conceded
- **Ajax** and **Feyenoord** form the top tier with strong attack and defense
- Relegated teams (bottom-left quadrant) struggle on both ends

---

## The Model

### Dixon-Coles (1997)

We use the **Dixon-Coles model**, an extension of the basic Poisson model that adds:

1. **Team Attack/Defense Parameters**: Each team has offensive and defensive strength
2. **Home Advantage**: Systematic boost for home teams
3. **Low-Score Correlation**: Adjustment for 0-0, 1-0, 0-1, 1-1 results
4. **Temporal Decay**: Recent matches weighted more heavily

```
P(home=x, away=y) = τ(x,y,λ,μ,ρ) × Poisson(x|λ) × Poisson(y|μ)

where:
  λ = exp(attack_home - defense_away + home_advantage)
  μ = exp(attack_away - defense_home)
  τ = Dixon-Coles correlation adjustment
  ρ = correlation parameter for low scores
```

### Temporal Decay

Team strengths change over time. We weight recent matches more heavily:

```
weight = exp(-ξ × days_since_match)
```

With ξ ≈ 0.005, matches from 6 months ago have ~40% weight, while matches from 2 years ago have ~3% weight.

---

## Hyperparameter Optimization

We optimize two key parameters using **Optuna**:

| Parameter | Range | Best Value |
|-----------|-------|------------|
| `xi` (decay rate) | 0.001 - 0.01 | 0.0034 |
| `window_years` | 2 - 6 | 3 years |

The optimization uses **rolling window evaluation** to prevent data leakage:

1. For each match in the test season:
   - Train on all prior data (within window)
   - Predict the match
   - Score against actual outcome
2. Minimize average RPS across all test matches

---

## Model Evaluation

### Proper Scoring Rules

We evaluate using **Ranked Probability Score (RPS)**, which penalizes confident wrong predictions:

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **RPS** | 0.1837 | Lower is better (random ≈ 0.33) |
| **Brier** | 0.5507 | Mean squared error |
| **Log Loss** | 0.9325 | Negative log-likelihood |

### Calibration

A well-calibrated model's predicted probabilities should match actual frequencies:

<p align="center">
  <img src="plots/04_calibration.png" width="60%" />
</p>

**Results:**
- Predicted 30% → Actual 26% ✓
- Predicted 50% → Actual 53% ✓
- Predicted 69% → Actual 71% ✓
- Predicted 86% → Actual 87% ✓

The model is well-calibrated across all probability ranges.

---

## Predictions

### Match Probability Dashboard

For any matchup, the model generates:

<p align="center">
  <img src="plots/05_match_prediction.png" width="85%" />
</p>

**Markets Available:**
- **1X2**: Home Win / Draw / Away Win probabilities
- **Expected Goals (xG)**: Mean goals for each team
- **BTTS**: Both Teams To Score (Yes/No)
- **Totals**: Over/Under for 0.5, 1.5, 2.5, 3.5, 4.5 goals

### Exact Score Probabilities

The model produces a full probability grid for exact scores:

<p align="center">
  <img src="plots/06_probability_grid.png" width="60%" />
</p>

---

## Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│  CSV Data   │───▶│  Validation  │───▶│  Features   │
└─────────────┘    └──────────────┘    └─────────────┘
                                              │
                                              ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ Prediction  │◀───│   Training   │◀───│ Optimization│
└─────────────┘    └──────────────┘    └─────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │  Evaluation  │
                   └──────────────┘
```

### Module Overview

| Module | Purpose |
|--------|---------|
| `data_loader.py` | CSV parsing, validation, normalization |
| `feature_engineering.py` | Temporal decay, team stats, league averages |
| `model_training.py` | Optuna optimization, Dixon-Coles training |
| `model_evaluation.py` | RPS, Brier, Log Loss, calibration |
| `predict.py` | Probability grids, market extraction |
| `visualizations.py` | All plots for analysis and documentation |

---

## Usage Examples

### Basic Prediction

```python
from data_loader import load_matches
from model_training import ModelTrainer, TrainingConfig
from predict import MatchPredictor, format_prediction

# Load and prepare data
df = load_matches("data/eredivisie_sample.csv")

# Train model
trainer = ModelTrainer()
model, result = trainer.train(df, TrainingConfig(xi=0.005, window_years=3))

# Predict a match
predictor = MatchPredictor(model)
prediction = predictor.predict("afc ajax", "psv eindhoven")

print(format_prediction(prediction))
```

### With Hyperparameter Optimization

```python
from model_training import ModelTrainer, OptimizationConfig

trainer = ModelTrainer()
model, result, stats = trainer.optimize(
    df,
    config=OptimizationConfig(n_trials=50, metric="rps"),
    model_type="dixon_coles"
)

print(f"Best xi: {stats['best_params']['xi']:.6f}")
print(f"Best window: {stats['best_params']['window_years']} years")
print(f"Best RPS: {stats['best_score']:.4f}")
```

### Generate All Visualizations

```python
from visualizations import generate_all_plots
from model_evaluation import ModelEvaluator

evaluator = ModelEvaluator()
saved_files = generate_all_plots(df, model, evaluator, output_dir="plots")
```

---

## Project Structure

```
football-expected-goals-ml-pipeline/
├── README.md                  # This documentation
├── LICENSE                    # MIT License
├── requirements.txt           # Dependencies
├── run_demo.py                # Full pipeline script
│
├── data/
│   └── eredivisie_sample.csv  # Sample data (1,242 matches)
│
├── plots/                     # Generated visualizations
│   ├── 01_goals_distribution.png
│   ├── 02_results_by_season.png
│   ├── 03_team_strength.png
│   ├── 04_calibration.png
│   ├── 05_match_prediction.png
│   └── 06_probability_grid.png
│
├── models/                    # Saved model artifacts
│
├── data_loader.py             # Data ingestion
├── feature_engineering.py     # Feature computation
├── model_training.py          # Training & optimization
├── model_evaluation.py        # Scoring & evaluation
├── predict.py                 # Predictions & markets
└── visualizations.py          # Plotting functions
```

---

## Dependencies

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
optuna>=3.0
joblib>=1.3
penaltyblog>=1.0
matplotlib>=3.7
seaborn>=0.12
```

---

## References

- Dixon, M. J., & Coles, S. G. (1997). *Modelling Association Football Scores and Inefficiencies in the Football Betting Market*. Journal of the Royal Statistical Society.
- [penaltyblog](https://github.com/martineastwood/penaltyblog) - Python library for football analytics

---

## Author

**Oscar Gonzalez** — [@oscgonz19](https://github.com/oscgonz19)

*Part of my data science portfolio showcasing end-to-end ML pipelines for sports analytics.*

---

## License

MIT License - See [LICENSE](LICENSE) for details.
