# Portfolio Demo: Goal Prediction Pipeline

A simplified, educational implementation of a football match goal prediction system using statistical models (Poisson and Dixon-Coles). This demo showcases end-to-end machine learning pipeline development for sports analytics.

## Overview

This demo implements a complete pipeline for predicting football match outcomes:

```
CSV Data → Validation → Feature Engineering → Optimization → Training → Evaluation → Prediction
```

### Key Features

- **Data Loading**: Robust CSV parsing with validation and normalization
- **Feature Engineering**: Temporal decay weights, team statistics, league averages
- **Model Training**: Poisson and Dixon-Coles goal models with hyperparameter optimization
- **Evaluation**: Proper scoring rules (RPS, Brier Score, Log Loss) with calibration analysis
- **Prediction**: Match outcome probabilities with derived markets (BTTS, Totals)

## Case Study

For a complete analysis and detailed walkthrough of the methodology, statistical foundations, and real-world results, check out the full case study:

**[Football Match Probability Prediction - Case Study](https://github.com/oscgonz19/football-probabilities-ml-poisson-case-study)**

## Quick Start

```bash
# Clone and setup
git clone git@github.com:oscgonz19/football-expected-goals-ml-pipeline.git
cd football-expected-goals-ml-pipeline
pip install -r requirements.txt

# Run with sample Eredivisie data (1,242 matches, 4 seasons)
python run_demo.py --data data/eredivisie_sample.csv

# Or run with synthetic demo data
python run_demo.py

# Skip optimization for faster execution
python run_demo.py --no-optimize
```

### Sample Data

The repo includes real match data from the **Dutch Eredivisie** (2021-2025):
- `data/eredivisie_sample.csv` - 1,242 matches across 4 seasons
- 23 teams including Ajax, PSV, Feyenoord, etc.

## Directory Structure

```
football-expected-goals-ml-pipeline/
├── README.md                # This documentation
├── LICENSE                  # MIT License
├── requirements.txt         # Dependencies
├── run_demo.py              # Complete pipeline script
├── data/
│   └── eredivisie_sample.csv  # Sample match data (Eredivisie 2021-2025)
├── __init__.py              # Package exports
├── data_loader.py           # CSV loading and validation
├── feature_engineering.py   # Feature computation utilities
├── model_training.py        # Model training and optimization
├── model_evaluation.py      # Scoring and evaluation metrics
├── predict.py               # Prediction and market extraction
└── models/                  # Saved model artifacts (created on run)
```

## Pipeline Components

### 1. Data Loader (`data_loader.py`)

Handles loading and validation of match data from CSV files.

**Expected CSV Format:**
| Column | Type | Description |
|--------|------|-------------|
| date | YYYY-MM-DD | Match date |
| season | string | Season identifier (e.g., "2023-2024") |
| home_team | string | Home team name |
| away_team | string | Away team name |
| home_goals | int | Goals scored by home team |
| away_goals | int | Goals scored by away team |

**Key Validations:**
- Non-empty dataset
- At least 2 unique teams per side
- Non-negative goal values
- No duplicate matches
- At least 2 seasons for proper evaluation

```python
from portfolio_demo import MatchDataLoader

loader = MatchDataLoader()
df = loader.load("matches.csv")
validation = loader.validate(df)
df_clean = loader.prepare(df)
```

### 2. Feature Engineering (`feature_engineering.py`)

Computes features and statistics for model training and analysis.

**Features:**
- **Temporal Decay Weights**: Dixon-Coles exponential decay (`w = exp(-xi * days)`)
- **Team Statistics**: Goals scored/conceded, points, home advantage ratio
- **League Averages**: Overall scoring rates and outcome frequencies

```python
from portfolio_demo import FeatureEngineer

engineer = FeatureEngineer()
weights = engineer.compute_decay_weights(df, xi=0.005)
team_stats = engineer.compute_team_stats(df, window_days=365)
league_avg = engineer.compute_league_averages(df)
```

### 3. Model Training (`model_training.py`)

Trains goal prediction models with optional hyperparameter optimization.

**Supported Models:**
- **Poisson**: Basic independent Poisson model for home/away goals
- **Dixon-Coles**: Poisson with low-score correlation adjustment (recommended)

**Optimization:**
- Uses Optuna with TPE sampler for hyperparameter search
- Optimizes `xi` (decay rate) and `window_years` (training window)
- Rolling window evaluation on hold-out season

```python
from portfolio_demo import ModelTrainer
from portfolio_demo.model_training import OptimizationConfig

trainer = ModelTrainer(output_dir="models")

# Option 1: Optimize hyperparameters
model, result, stats = trainer.optimize(
    df,
    config=OptimizationConfig(n_trials=50),
    model_type="dixon_coles"
)

# Option 2: Train with fixed parameters
model, result = trainer.train(df, TrainingConfig(xi=0.005, window_years=4))

# Save model
trainer.save(model, result, name="my_model")
```

### 4. Model Evaluation (`model_evaluation.py`)

Evaluates predictions using proper scoring rules.

**Metrics:**
| Metric | Description | Range |
|--------|-------------|-------|
| RPS | Ranked Probability Score - penalizes distant misses more | [0, 1] |
| Brier | Mean squared error of probabilities | [0, 2] |
| Log Loss | Negative log likelihood of true outcome | [0, ∞) |

All metrics: **lower is better**.

```python
from portfolio_demo import ModelEvaluator

evaluator = ModelEvaluator()
results = evaluator.evaluate(model, df_test)
print(f"RPS: {results['rps'].score:.4f}")

# Calibration analysis
calibration = evaluator.evaluate_calibration(model, df_test)
```

### 5. Prediction (`predict.py`)

Generates match predictions with derived betting markets.

**Markets Supported:**
- **1X2**: Home win / Draw / Away win probabilities
- **Expected Goals**: Mean goals for each team
- **BTTS**: Both Teams To Score (yes/no)
- **Totals**: Over/Under for various goal lines (0.5, 1.5, 2.5, 3.5, 4.5)

```python
from portfolio_demo import MatchPredictor
from portfolio_demo.predict import format_prediction

predictor = MatchPredictor(model, model_type="dixon_coles")
prediction = predictor.predict("arsenal", "chelsea")

# Pretty print
print(format_prediction(prediction))

# Access specific markets
print(f"Home win: {prediction.probabilities.home_win:.1%}")
print(f"xG Home: {prediction.expected_goals.home:.2f}")
print(f"BTTS Yes: {prediction.btts_yes:.1%}")
```

## Statistical Models

### Poisson Model

The basic model assumes home and away goals follow independent Poisson distributions:

```
Home Goals ~ Poisson(λ_home)
Away Goals ~ Poisson(λ_away)

where:
λ_home = exp(attack_home - defence_away + home_advantage)
λ_away = exp(attack_away - defence_home)
```

### Dixon-Coles Model

Extension of Poisson that adds correlation adjustment for low-scoring matches (0-0, 1-0, 0-1, 1-1):

```
P(x,y) = τ(x,y,λ,μ,ρ) × Poisson(x|λ) × Poisson(y|μ)

where τ is the Dixon-Coles adjustment factor and ρ controls correlation.
```

The model also incorporates temporal decay weighting:

```
w_i = exp(-ξ × days_since_match)
```

This allows the model to adapt to changing team strengths over time.

## Evaluation Methodology

The optimization uses **rolling window evaluation**:

1. For each match in the test season:
   - Train on all data before that match (within window)
   - Generate prediction for the match
   - Score the prediction against actual outcome
2. Average scores across all test matches
3. Select hyperparameters that minimize average score

This prevents data leakage and simulates real-world prediction scenarios.

## Example Output

```
============================================================
GOAL PREDICTION PIPELINE DEMO
============================================================

[1/6] Loading Data...
  Created 80 sample matches (demo data)
  Teams: ['team a', 'team b', 'team c', 'team d']
  Seasons: ['2022-2023', '2023-2024']

[2/6] Computing Features...
  Average home goals: 1.35
  Average away goals: 0.95
  Home win rate: 45.0%

[3/6] Optimizing Hyperparameters...
  Best xi: 0.003421
  Best window: 3 years
  Best RPS: 0.1823

[5/6] Evaluating Model...
  RPS: 0.1845 (+/- 0.0912)
  BRIER: 0.5521 (+/- 0.2134)

[6/6] Generating Sample Predictions...
==================================================
Match: Team A vs Team B
==================================================

1X2 Probabilities:
  Home Win: 52.3%
  Draw:     24.1%
  Away Win: 23.6%

Expected Goals:
  Home: 1.58
  Away: 1.12
  Total: 2.70
```

## Dependencies

This demo uses standard data science libraries:

```
pandas>=2.0
numpy>=1.24
scipy>=1.10
optuna>=3.0
joblib>=1.3
penaltyblog>=0.5  # For statistical models
```

## Extending the Demo

### Adding Custom Metrics

```python
# In model_evaluation.py, add to METRICS:
def compute_custom_metric(probs, actual):
    # Your metric implementation
    return score

METRICS["custom"] = compute_custom_metric
```

### Using Real Data

Replace the sample data with actual historical match data:

```python
# Download from football-data.co.uk or similar sources
python -m portfolio_demo.run_demo --data EPL_2020_2024.csv
```

## License

MIT License - See [LICENSE](LICENSE) for details.

The underlying statistical models are from the [penaltyblog](https://github.com/martineastwood/penaltyblog) library.

## Author

**Oscar Gonzalez** - [@oscgonz19](https://github.com/oscgonz19)

Created as a portfolio demonstration of sports analytics and machine learning pipeline development.

---

*Part of my data science portfolio showcasing end-to-end ML pipelines for sports analytics.*
