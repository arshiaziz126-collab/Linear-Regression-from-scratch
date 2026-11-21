"""
GlycoCast - Linear Regression written from scratch with only NumPy.

Goal: predict how much a patient's diabetes will progress in one year
from 10 baseline health measurements (age, BMI, blood pressure, blood tests).

The model is a straight-line (linear) formula:

    prediction = w1*x1 + w2*x2 + ... + w10*x10 + b

    w = weights (how much each feature pushes the prediction up or down)
    b = bias (the starting value when every feature is at its average)

Two ways to find the best w and b, both written by hand:
    1. Gradient descent  -> start at 0 and walk downhill on the error, step by step
    2. Normal equation   -> solve the maths exactly in one shot
If both give the same numbers, the implementation is correct.
"""

import numpy as np


class LinearRegression:
    def __init__(self, lr=0.05, epochs=2000, l2=0.0, tol=1e-9, track_path=False):
        self.lr = lr                  # learning rate (size of each step)
        self.epochs = epochs          # maximum number of steps
        self.l2 = l2                  # Ridge penalty (0 = plain linear regression)
        self.tol = tol                # stop when the loss stops improving
        self.track_path = track_path  # remember (w, b) at each step for plots
        self.w, self.b = None, 0.0
        self.loss_history, self.path = [], []

    # ---------------------------------------------------------------- core
    def predict(self, X):
        return X @ self.w + self.b

    def loss(self, X, y):
        """Mean Squared Error (+ Ridge penalty if l2 > 0)."""
        err = self.predict(X) - y
        return np.mean(err ** 2) + self.l2 * np.sum(self.w ** 2)

    def gradients(self, X, y):
        """
        Derivatives of the loss (calculus, done by hand):
            dL/dw = (2/n) * X^T (y_pred - y) + 2*l2*w
            dL/db = (2/n) * sum(y_pred - y)
        """
        n = len(y)
        err = self.predict(X) - y
        dw = (2 / n) * (X.T @ err) + 2 * self.l2 * self.w
        db = (2 / n) * np.sum(err)
        return dw, db

    # ------------------------------------------------- 1. gradient descent
    def fit(self, X, y, w0=None, b0=0.0):
        self.w = np.zeros(X.shape[1]) if w0 is None else np.array(w0, dtype=float)
        self.b = float(b0)
        self.loss_history, self.path = [], []
        prev = np.inf
        for _ in range(self.epochs):
            if self.track_path:
                self.path.append((*self.w.copy(), self.b))
            dw, db = self.gradients(X, y)
            self.w -= self.lr * dw        # walk downhill
            self.b -= self.lr * db
            cur = self.loss(X, y)
            self.loss_history.append(cur)
            if abs(prev - cur) < self.tol:
                break
            prev = cur
        self.n_steps = len(self.loss_history)
        return self

    # ------------------------------------------------- 2. normal equation
    def fit_closed_form(self, X, y):
        """
        Exact answer:  theta = (X^T X + l2*I)^-1 X^T y
        A column of 1s is added so the bias is learned too (bias is not penalised).
        """
        Xb = np.c_[np.ones(len(X)), X]
        reg = self.l2 * len(X) * np.eye(Xb.shape[1])
        reg[0, 0] = 0.0
        theta = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
        self.b, self.w = float(theta[0]), theta[1:]
        return self


# --------------------------------------------------------------- helpers
class StandardScaler:
    """z = (x - mean) / std, learned on training data only."""

    def fit(self, X):
        self.mean, self.std = X.mean(axis=0), X.std(axis=0) + 1e-12
        return self

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_weights(self, w, b):
        """Turn weights learned on scaled data back into real-unit weights."""
        w_real = w / self.std
        b_real = b - np.sum(w_real * self.mean)
        return w_real, b_real


def train_test_split(X, y, test_frac=0.2, seed=42):
    idx = np.random.default_rng(seed).permutation(len(X))
    n_test = int(round(len(X) * test_frac))
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def kfold_indices(n, k=5, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        val = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, val


# --------------------------------------------------------------- metrics
def mse(y, p):  return float(np.mean((y - p) ** 2))
def rmse(y, p): return float(np.sqrt(mse(y, p)))
def mae(y, p):  return float(np.mean(np.abs(y - p)))


def r2(y, p):
    """R^2 = 1 - (model error / error of always guessing the average)."""
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2))
