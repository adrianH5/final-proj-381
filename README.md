# 🏀 NCAA Matchup Outcome Prediction (2025)

This project uses machine learning to predict the outcome of NCAA basketball matchups using engineered statistical differences between teams. Models are trained on features such as point differential, seed difference, strength of schedule, and others — all calculated as team A minus team B.

The goal is to evaluate how well different machine learning models can predict the winner of a given matchup.

Datasets were acquired from Kaggle:
- [March Madness 2025](https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data?select=Conferences.csv})
- [March Madness Data](https://www.kaggle.com/datasets/nishaanamin/march-madness-data)
---

## 🚀 Features

- Model training using:
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
  - Support Vector Machine
- 5-fold Stratified Cross-Validation with mean and std reporting
- Baseline comparisons: random guessing and higher-seed wins
- Visual feature analysis: importance, distribution, correlation
- Error bar plots for accuracy with standard deviation
- All outputs saved for reproducibility

---

## 🗂️ Project Structure

```
.
├── matchup_modeling_analysis.py       # Main analysis script
├── matchups_full_2025.csv             # Input dataset
├── cross_validation_results.csv       # Mean/std scores from CV
├── summary_stats_by_label.csv         # Descriptive stats by label
└── figures/                           # Saved charts and figures
    ├── cv_accuracy_with_std.png
    ├── cv_f1_score.png
    ├── log_loss_test_set.png
    ├── class_distribution.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── permutation_importance.png
    ├── dist_<feature>.png
    ├── boxplot_<feature>.png
    └── pairplot_top5_features.png
```

---

## ⚙️ Setup & Installation

1. Clone the repo or download the files.
2. Place your input dataset as `matchups_full_2025.csv` in the root folder.
3. Install required dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

4. Run the script in google colab/jupyter notebook. You can run it in drive and colab easily.


All figures and CSV outputs will be saved automatically.

---

## 📚 Dataset Details

- Input: `matchups_full_2025.csv`
- Rows represent matchups between two teams.
- Label: `1` if Team A wins, `0` if Team B wins.
- Features: any column starting with `diff_` is used, representing statistical differences between Team A and Team B (e.g., `diff_efg_pct`, `diff_seed`, etc.)

---

## 🧠 Models Evaluated

| Model               | Description                                   |
| ------------------- | --------------------------------------------- |
| Logistic Regression | Baseline linear model                         |
| Random Forest       | Tree-based ensemble, non-linear relationships |
| Gradient Boosting   | Boosted tree ensemble, strongest performer    |
| SVM                 | Support Vector Machine with scaling           |
| Higher Seed Guess   | Simple baseline using `diff_seed`             |
| Random Guess        | Random binary prediction (coin flip)          |

---

## 🧪 Evaluation Methods

- **Train/Test Split:** 80/20 holdout to test generalization
- **Cross-Validation:** 5-fold stratified CV to compute mean ± std for:

  - Accuracy
  - F1 Score

- **Log Loss (Binary Cross-Entropy):** Computed on test set for probability calibration

---

## 📊 Visualizations

All figures are saved in the `figures/` folder. Key plots include:

### Model Performance

- `cv_accuracy_with_std.png`: Accuracy with ± std error bars
- `cv_f1_score.png`: F1 Score comparison
- `log_loss_test_set.png`: Log loss on test data

### Feature Insights

- `feature_importance.png`: Top 15 from Gradient Boosting
- `permutation_importance.png`: Model-agnostic feature importance
- `correlation_heatmap.png`: Feature correlation map

### Distribution Visuals

- `class_distribution.png`: 0 vs 1 matchup label counts
- `dist_<feature>.png`: Density plot per class
- `boxplot_<feature>.png`: Distribution range comparison
- `pairplot_top5_features.png`: Visual interaction between top features

---

## 🔍 Summary Statistics

Stored in:

- `summary_stats_by_label.csv`  
  Includes descriptive statistics (mean, std, min, max) for the top 5 most predictive features, grouped by match outcome.

---

## ✅ Key Insights

- Gradient Boosting performed best in both F1 Score and Log Loss.
- `diff_seed`, `diff_margin_of_victory`, and `diff_strength_of_schedule` were among the most predictive features.
- Machine learning outperforms naive baselines like picking the higher seed or guessing randomly.

---

## 💡 Future Directions

- Integrate XGBoost or LightGBM for optimized boosting
- Incorporate contextual data: recent form, injuries, home-court
- Add time-based splits to reflect seasonal dynamics
- Use SHAP or LIME for explainable AI predictions

---

## 🧑‍💻 Author & Tools

Built with:

- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn

---
