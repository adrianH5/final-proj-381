import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone

DATA_PATH = Path("matchups_full_2025.csv")

# 0.  load & split data
df = pd.read_csv(DATA_PATH)
feature_cols = [c for c in df.columns if c.startswith("diff_")]
X = df[feature_cols].values
y = df["label"].values
seed_diff = df["diff_seed"].values

X_train, X_test, y_train, y_test, sd_train, sd_test = train_test_split(
    X, y, seed_diff, test_size=0.20, random_state=42, stratify=y
)


# 1.  Baseline #1 – always pick better seed
class NaiveSeedBaseline(BaseEstimator, ClassifierMixin):
    """
    Predict 1 if Team A's seed < Team B's seed  (diff_seed < 0),
    else predict 0.  Tie: predict 0.5 randomly.
    """

    def fit(self, X, y=None):
        return self

    def predict(self, X, seed_diff):
        rng = np.random.RandomState(42)
        preds = np.where(
            seed_diff < 0,
            1,
            np.where(seed_diff > 0, 0, rng.randint(0, 2, size=len(seed_diff))),
        )
        return preds

    def score(self, X, y, seed_diff):
        return accuracy_score(y, self.predict(X, seed_diff))


# 2.  Baseline #2 – random guessing
class RandomBaseline(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return self.rng.randint(0, 2, size=len(X))

    def predict_proba(self, X):
        p = np.full((len(X), 2), 0.5)
        return p

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


# 3.  Logistic‑regression model
class LogisticRegressionModel(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.clf = LogisticRegression(max_iter=2000, C=C, solver="lbfgs")

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def score(self, X, y):
        return self.clf.score(X, y)


# 4.  Neural‑network model (one hidden layer)
class NeuralNetworkModel(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes=(1024, 512, 256, 128, 64),
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=1000,  # give it more room
        early_stopping=True,  # stop when no progress
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.random_state = random_state

        self.clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            early_stopping=early_stopping,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
        )

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)

    def score(self, X, y):
        return self.clf.score(X, y)


# 5.  Demo / sanity check
if __name__ == "__main__":
    models = {
        "Naive‑seed": NaiveSeedBaseline(),
        "Random": RandomBaseline(),
        "Logistic": LogisticRegressionModel(C=1.0),
        "Neural net": NeuralNetworkModel(),  # (512,256,128,64,32)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, base_model in models.items():
        fold_acc = []

        # ---- special case: NaiveSeedBaseline needs seed_diff ----
        if name == "Naive‑seed":
            for _, test_idx in skf.split(X, y):
                acc = base_model.score(
                    X[test_idx],  # ignored
                    y[test_idx],
                    seed_diff[test_idx],  # required
                )
                fold_acc.append(acc)

        # ---- all other models follow normal sklearn pattern ----
        else:
            for train_idx, test_idx in skf.split(X, y):
                model = clone(base_model)  # fresh model each fold
                model.fit(X[train_idx], y[train_idx])
                acc = model.score(X[test_idx], y[test_idx])
                fold_acc.append(acc)

        print(
            f"{name:<12}:  "
            f"mean acc = {np.mean(fold_acc):.3f}  ± {np.std(fold_acc):.3f}"
        )
