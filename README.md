# 🏀 NCAA Matchup Outcome Prediction - 2025

This project uses machine learning models to predict the outcome of NCAA basketball matchups using team performance difference metrics. It evaluates multiple algorithms, compares them to simple baselines, and provides detailed visual and statistical insights.

---

## 📁 Contents

- `matchup_modeling_analysis.py` – Full executable analysis script
- `matchups_full_2025.csv` – Dataset (features start with `diff_`)
- `figures/` – All saved visualizations
- `cross_validation_results.csv` – Cross-validation summary
- `summary_stats_by_label.csv` – Summary stats grouped by outcome

---

## 🧠 Models Evaluated

| Model                  | Description                                 |
|------------------------|---------------------------------------------|
| Logistic Regression    | Linear baseline                             |
| Random Forest          | Tree-based ensemble                         |
| Gradient Boosting      | Best performer overall                      |
| Support Vector Machine | Kernel method                               |
| Higher Seed Guess      | Simple rule-based baseline (seed diff)      |
| Random Guess           | Random 50/50 prediction                     |

---

## ⚙️ Metrics Used

- **Accuracy**: Overall correct predictions  
- **F1 Score**: Harmonic mean of precision and recall  
- **Log Loss**: Penalizes incorrect confidence

---

## 🔍 Key Feature Engineering

Only columns starting with `diff_` were used. These represent team-vs-team differences in:

- Efficiency
- Rebounding
- Turnover rates
- Power rankings
- NCAA seed and round progress

---

---

## 📊 Visualizations (Improved Descriptions)

All charts and plots are saved in the `figures/` folder for easy viewing.

---

### 📈 Model Performance

**1. `accuracy_comparison.png`**  
*What it shows:*  
Bar chart comparing how often each model correctly predicted the winner of a game.

**2. `f1_score_comparison.png`**  
*What it shows:*  
Compares models by F1 Score, which balances precision (how often wins were correctly identified) and recall (how many actual wins were found).  
*Why it matters:* Especially important if the data has class imbalance.

**3. `log_loss_comparison.png`**  
*What it shows:*  
Evaluates how good the models were at assigning probabilities to outcomes.  
*Why it matters:* A model that confidently makes the wrong call is penalized more.

---

### 🔎 Feature Insights

**4. `feature_importance.png`**  
*What it shows:*  
Top 15 features (stat differences between teams) that the Gradient Boosting model relied on most.  
*Why it matters:* Helps identify which stats are most predictive (e.g., seed difference, margin of victory).

**5. `permutation_importance.png`**  
*What it shows:*  
Alternative method showing how much model accuracy drops when each feature is shuffled.  
*Why it matters:* Confirms importance using a model-agnostic method.

**6. `correlation_heatmap.png`**  
*What it shows:*  
Grid showing how related each stat is to the others (blue = negative, red = positive).  
*Why it matters:* Useful to spot redundant features or strong relationships.

---

### 🧪 Feature Distributions and Differences

**7. `class_distribution.png`**  
*What it shows:*  
Bar chart of how many matchups were won by Team A vs. Team B.  
*Why it matters:* Reveals if the dataset is balanced or skewed toward one outcome.

**8. `dist_<feature>.png`**  
*What it shows:*  
For each top feature, this shows how its values differ between Team A wins and Team B wins.  
*Why it matters:* Helpful to visually see separation between classes.

**9. `boxplot_<feature>.png`**  
*What it shows:*  
Side-by-side comparison of the distribution of feature values for wins vs. losses.  
*Why it matters:* Makes it easier to compare feature medians, spreads, and outliers.

---

### 🔀 Relationships Between Features

**10. `pairplot_top5_features.png`**  
*What it shows:*  
Scatterplot matrix showing pairwise relationships between the top 5 most important features, colored by match outcome.  
*Why it matters:* Helps see how combinations of features relate to win/loss.
