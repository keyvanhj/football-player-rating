# ⚽ Football Player Rating: Cleaning, EDA, Modeling and Power BI Dashboard

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-HistGradientBoosting-F7931E?logo=scikitlearn&logoColor=white)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi&logoColor=black)
![Status](https://img.shields.io/badge/status-complete-2e7d32)

An end-to-end data project that answers one question: **what drives a football player's match rating?**
It covers data cleaning, exploratory analysis, a machine-learning model that predicts the rating, and an interactive Power BI dashboard.

> 🔒 **The dataset is confidential and is not included in this repository.** The code, methodology and aggregated results are shared; see [Dataset](#dataset) for the expected schema.

> **Headline result:** a HistGradientBoosting model predicts a player's match rating with **MAE 0.154** (on a 10-point scale) and **R² 0.89** on unseen matches.
> Using individual statistics alone the model reaches **R² 0.68**. Adding the match score lifts it to 0.89, and goalkeepers go from R² 0.17 to 0.75.

---

## Table of Contents

- [Project overview](#project-overview)
- [Key findings](#key-findings)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [Pipeline](#pipeline)
  - [1. Data cleaning](#1-data-cleaning)
  - [2. Exploratory data analysis](#2-exploratory-data-analysis)
  - [3. Modeling](#3-modeling)
- [Model results](#model-results)
- [Power BI dashboard](#power-bi-dashboard)
- [How to run](#how-to-run)
- [Design decisions](#design-decisions)
- [Limitations](#limitations)
- [Future work](#future-work)
- [Tech stack](#tech-stack)
- [License](#license)
- [Author](#author)

---

## Project overview

| | |
|---|---|
| **Goal** | Understand what shapes a player's rating in a match and predict it |
| **Unit of analysis** | One row = one player in one match |
| **Size** | 173,587 player-match records · 6,818 matches · 27,974 players · 188 league names |
| **Period** | 31 Jan 2025 to 8 Jun 2025 (97 match dates) |
| **Target** | `player_rating` (range 2.43 to 9.97, mean 6.78, std 0.76) |
| **Task type** | Regression |
| **Deliverables** | Analysis notebook (cleaning, EDA, modeling) and a 5-page Power BI report (screenshots only) |

## Key findings

1. **The team result shifts every rating.** Average rating is 6.38 after a loss, 6.80 after a draw and 7.18 after a win (about +0.8 between win and loss).
2. **Goalkeepers depend on the result far more than outfield players.** The win-loss gap is 1.33 for goalkeepers vs 0.76 for outfield players (about 1.75x). For goalkeepers, goals conceded is the main signal (correlation -0.72).
3. **Goals are the strongest individual action.** Each of the first two goals adds about 1.1 rating points; the effect then saturates close to 10.
4. **League, country, gender and home advantage matter little.** The average rating of the large countries only ranges from 6.67 to 6.85. Home teams win more often (43.9% vs 29.6%), but with the result held constant, home and away ratings are almost identical (difference 0.01 to 0.04).
5. **Team quality and rating move together.** Team average rating correlates 0.81 with team win rate (teams with at least 100 records).

## Dataset

> 🔒 **Confidential.** The data is proprietary/licensed and cannot be redistributed, so it is **not part of this repository**, and neither are files derived from it (`cleaned_data.csv`, `predictions.csv`).
> Only aggregated results are published here. No player or team level records are shown.

Each row is one player in one match. The raw file has **63 columns**:

| Group | Columns (summary) |
|---|---|
| Match and league | match id, match date, league id / name, parent league name and season, country code |
| Teams | home and away team id / name, the player's team id / name, `is_home_team`, final score (`home_score`, `away_score`), `team_goals`, `opponent_goals`, `goal_difference`, `team_result` |
| Player | player id / name, `is_goalkeeper`, `minutes_played` |
| Attacking | goals, assists, shots on target, blocked shots, expected assists, xG + xA, chances created, touches in the opposition box |
| Passing and possession | touches, accurate / total passes, pass accuracy, passes into the final third, crosses, long balls (counts, attempts, success rates) |
| Dribbling | successful dribbles, attempts, success rate |
| Defending | tackles won, interceptions, clearances, defensive actions, recoveries, dribbled past |
| Duels and fouls | duels won / lost, ground and aerial duels (won, attempts, rate), fouls committed, was fouled |
| **Target** | `player_rating` (continuous, about 2.4 to 10) |

After cleaning, the dataset has **68 columns** (63 original + `minutes_missing`, `has_xg_data`, `comp_format`, `gender`, `stage`).

**Using your own data:** the notebook runs on any CSV with the same column names and meaning (one row per player per match, with a numeric `player_rating`). The column names the code expects are the ones summarised in the table above.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Football_Player_Rating_Analysis.ipynb     # cleaning + EDA + modeling (outputs cleared)
└── docs/
    └── images/                                   # aggregated dashboard screenshots only
```

Files that are generated locally and **must not be committed** (they contain confidential data or are derived from it): the raw CSV, `cleaned_data.csv`, `predictions.csv`, `rating_model.joblib` and the Power BI `.pbix` file (a `.pbix` embeds a full copy of the imported data). See `.gitignore`.

## Pipeline

```text
raw CSV ──► Cleaning ──► cleaned_data.csv ──► EDA ──► Modeling ──► predictions.csv ──► Power BI
                                                          └──────► rating_model.joblib
```

### 1. Data cleaning

| Step | What was done | Why |
|---|---|---|
| Missing values | Analysed null patterns by column group, by goalkeeper/outfield and by league | Nulls were mostly structural, not random |
| Counting stats | Nulls in dribbles, crosses, long balls, duels, fouls suffered and passes into the final third filled with `0` | The minimum value is 1 and 0 never appears, so null means "did not happen" |
| Goalkeepers | Attacking stats null only for goalkeepers, filled with `0` | These stats are not recorded for goalkeepers |
| xG / xA | Filled with `0` only in leagues that record them; left as `NaN` in leagues that do not (70,031 rows, about 40%) | "Not recorded" is not the same as zero |
| Rates | Percentages recomputed from counts; `NaN` when attempts = 0 | 0% would wrongly mean "tried and failed every time" |
| Invalid `team_id` | 253 rows (23 matches) had a team id that belonged to neither team; rebuilt `is_home_team` and `team_id` by elimination and verified all 23 matches | Wrong id broke team goals and result |
| Team result | Rebuilt `team_goals`, `opponent_goals`, `goal_difference`, `team_result` from the score | Keeps all derived columns consistent |
| Minutes | 1,913 missing values (1.1%) imputed with the median of the same touches decile and goalkeeper group; flagged in `minutes_missing` | Deleting rows loses valid data, zero would mean "did not play" |
| Derived columns | `comp_format` (6v6 / 11v11), `gender`, `stage` created from keywords in the league name | Not present in the raw data |
| Outliers | Checked with IQR and logical rules; no impossible values, so none removed | Extremes are real (e.g. 8 goals in a match, goalkeepers rated 2.4) |
| Duplicates | No duplicate rows or duplicate (match, player) pairs | |
| Redundant columns | `parent_league_season` is constant, `parent_league_name` equals `league_name`; kept in the file, excluded from the model | |

Output: `cleaned_data.csv` (173,587 rows × 68 columns).

### 2. Exploratory data analysis

- Distribution of the target (approximately symmetric, skewness 0.34; goalkeepers far more dispersed than outfield players: std 1.02 vs 0.73)
- Correlation with the rating, computed separately for outfield players and goalkeepers
- Rating by team result, by home/away and by position
- Rating by goals scored and by minutes played
- Rating by league, country, competition format, gender and stage
- Weekly trend of the average rating (weeks with fewer than 500 rows are excluded)
- Top-rated players with at least 10 matches (only about 21% of players reach that many matches)

### 3. Modeling

**Two feature sets**

| Set | Features | Meaning |
|---|---|---|
| **A** | 240, individual statistics only | How well can an individual performance be predicted on its own? |
| **B** | 242, A + `team_goals` + `opponent_goals` | How much does the match context add? |

**Columns removed from the model and why**

| Columns | Reason |
|---|---|
| IDs (`match_id`, `player_id`, `team_id`, `home_team_id`, `away_team_id`, `league_id`) | Arbitrary numbers; trees could memorise them. `match_id` is used only as the grouping key for splits |
| Names (`player_name`, `team_name`, `home_team_name`, `away_team_name`, `parent_league_name`) | Duplicate the IDs; not unique (472 player names map to more than one id) |
| `match_date` | Only about 4 months of uneven coverage; the model could memorise week effects and cannot extrapolate to future dates |
| `country_code`, `parent_league_season` | Almost a function of the league / constant |
| `home_score`, `away_score`, `team_result`, `goal_difference` | Deterministic functions of `team_goals` and `opponent_goals`; would leak the result into A and be redundant in B |

**Features created:** `total_duels`, `shot_involvement`, `goal_contrib` (goals + assists), `pass_per_min`, `touches_per_min`. Categorical columns (`league_name`, `comp_format`, `gender`, `stage`, `is_goalkeeper`, `is_home_team`) are one-hot encoded.

**Split:** 80/20 grouped by `match_id` (`GroupShuffleSplit`), so no match appears in both sets: 138,933 training rows, 34,654 test rows, 0 shared matches.

**Models**

- **Linear Regression** (baseline), in a pipeline with a constant-0 imputer and a standard scaler
- **HistGradientBoostingRegressor** (main model): handles `NaN` natively, captures non-linear effects and interactions, fast on 170k rows

**Tuning:** 8 combinations of learning rate, number of leaves and L2 regularisation, evaluated with 3-fold cross-validation grouped by match on the training data only.

## Model results

Held-out test set (34,654 rows):

| Features | Model | Train R² | **Test R²** | Test MAE | Test RMSE |
|---|---|---|---|---|---|
| A (individual) | Linear | 0.660 | 0.658 | 0.293 | 0.443 |
| A (individual) | HistGradientBoosting | 0.720 | 0.679 | 0.278 | 0.429 |
| B (+ team goals) | Linear | 0.773 | 0.772 | 0.248 | 0.362 |
| **B (+ team goals)** | **HistGradientBoosting** | 0.914 | **0.890** | **0.153** | **0.251** |

A naive model that always predicts the average rating has an MAE of about 0.60.

**Goalkeepers vs outfield players (HistGradientBoosting)**

| | Set A: MAE / R² | Set B: MAE / R² | Test rows |
|---|---|---|---|
| Goalkeepers | 0.747 / 0.170 | 0.386 / 0.748 | 2,633 |
| Outfield | 0.240 / 0.759 | 0.134 / 0.913 | 32,021 |

**Final model** (set B, learning rate 0.05, 31 leaves, L2 = 1): **MAE 0.154, RMSE 0.251, R² 0.890**.
Grouped cross-validation R² was 0.8827 to 0.8844 for all eight settings, so the model is not sensitive to these hyperparameters.

**Error profile:** 82.8% of predictions are within 0.25 of the real rating and 94.5% within 0.5. The mean error (bias) is about 0.001. Very low ratings (below 5) are over-predicted; the model is weakest for goalkeepers and for 6v6 matches.

**Most important features** (permutation importance): `opponent_goals` (≈ 0.42), `goal_contrib` (≈ 0.33), `touches`, `duels_lost`, `ground_duel_success_rate`, `chances_created`, `team_goals`, `recoveries`.

> ⚠️ **How to read set B:** the model sees the goals scored and conceded in the match, which are known only after the game. It therefore *explains* a rating given the match outcome; it is **not** a pre-match forecast.

## Power BI dashboard

The Power BI report has five pages. The `.pbix` file is not published because it embeds the data; use screenshots that show only aggregated visuals.

| Page | Contents |
|---|---|
| **Home** | Navigation and the central question |
| **Overview** | KPI cards, weekly average rating, rating by result, venue × result, rating by position × result, rating by goals scored |
| **Players** | Player table, goals per 90 vs average rating, players by gender and stage, top 10 goal scorers |
| **Leagues** | League and team tables, top / bottom 10 leagues by average rating, average rating by country |
| **Model** | MAE / RMSE / R² / bias cards, actual vs predicted, error distribution, MAE and bias by rating, MAE by position and league, largest errors |

Slicers: league, stage, month, position, competition format and gender.

**Data model:** star schema with `cleaned_data` and `predictions` as fact tables and `Dim_League`, `Dim_Team`, `Dim_Player`, `Dim_Date`, `Dim_Position`, `Dim_Stage`, `Dim_Gender`, `Dim_Comp_format` as dimensions. Leagues, teams and players are joined by **id** (names are not unique: for example three different leagues are called "Premier League").

<!-- Add screenshots here, for example: -->
<!-- ![Overview](docs/images/overview.png) -->
<!-- ![Model](docs/images/model.png) -->

## How to run

The pipeline needs data with the schema described in [Dataset](#dataset).

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
jupyter
```

### 2. Provide your data

Put your CSV (same schema) next to the notebook, or edit the path in the first cell, and keep it out of version control.

### 3. Run the notebook

```bash
jupyter notebook notebooks/Football_Player_Rating_Analysis.ipynb
```

Run all cells in order (**Kernel → Restart & Run All**). The notebook writes `cleaned_data.csv`, `predictions.csv` and `rating_model.joblib` locally. Fitting the boosting model and the cross-validation take a few minutes.

### 4. Dashboard (optional)

Build the Power BI report from `cleaned_data.csv` and `predictions.csv` as a star schema (see [Power BI dashboard](#power-bi-dashboard)). Join leagues, teams and players on **ids**, not names.

## Design decisions

- **Two feature sets, not one.** Keeping the result-related columns out of set A separates "individual performance" from "performance given the match outcome", so the contribution of each is measurable.
- **Grouped split.** Rows from one match are very similar; splitting them across train and test would inflate the results.
- **Missing values are treated individually.** A null can mean zero, "not recorded" or "undefined"; each case got its own rule instead of one global fill.
- **Join on ids, not names.** 12 league names, 10 team names and 472 player names map to more than one id.
- **Boosting over linear.** The effect of goals saturates near 10, and the effect of goals conceded differs between goalkeepers and outfield players. Trees capture both; a single linear coefficient cannot.

## Limitations

- The data covers about four months with uneven weekly coverage, and about 40% of rows come from leagues that do not record xG/xA.
- The split is random by match, not chronological, so no claim is made about future matches.
- Set B uses information available only after the match.
- The choice between sets A and B was made after looking at test-set scores; cross-validation was run on set B only.
- The final model used all 1,000 allowed iterations (early stopping did not trigger), so more iterations might help slightly.
- Only linear regression and HistGradientBoosting were compared; other models (Random Forest, LightGBM, neural networks) were not tested.
- Goalkeepers are only 7.6% of the data and goalkeeper-specific statistics (saves, etc.) are missing; very low ratings are over-predicted.
- The dataset contains far more men (93.8% of players) than women, so conclusions about women's football are less certain.

## Future work

- Add goalkeeper-specific statistics (saves, post-shot xG) if available
- Increase `max_iter` and tune with grouped cross-validation
- Compare with LightGBM / Random Forest
- Use native categorical support instead of one-hot league columns
- Add a chronological train/test split to test generalisation over time
- Report separate metrics for 6v6 matches and women's football

## Tech stack

- **Python:** pandas, NumPy, matplotlib, seaborn, scikit-learn, joblib
- **Modeling:** HistGradientBoostingRegressor, LinearRegression, GroupShuffleSplit, GroupKFold, permutation importance
- **BI:** Power BI Desktop (Power Query, DAX)

## License

All rights reserved. This repository is shared for portfolio and demonstration purposes. No license is granted to copy, modify or redistribute the code, and the underlying dataset is confidential and not covered by or included in this repository.

## Author

**`[Keyvan Hajizadeh]`**

If you find this project useful, feel free to ⭐ the repository.
