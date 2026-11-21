"""
Train GlycoCast end to end:
    load -> split -> scale -> gradient descent -> check vs exact solution
    -> 5-fold cross-validation -> evaluate -> visuals -> save

Run:  python train.py
"""

import json
import numpy as np
from sklearn.datasets import load_diabetes          # only used to get the raw numbers

from glycocast.linreg import (LinearRegression, StandardScaler, train_test_split,
                              kfold_indices, mse, rmse, mae, r2)
from glycocast import viz

LABELS = {"age": "Age", "sex": "Sex", "bmi": "BMI", "bp": "Blood pressure",
          "s1": "Total cholesterol", "s2": "LDL (bad cholesterol)", "s3": "HDL (good cholesterol)",
          "s4": "Cholesterol / HDL ratio", "s5": "Triglycerides (log)", "s6": "Blood sugar"}


def main():
    d = load_diabetes(scaled=False)
    X, y, names = d.data, d.target, list(d.feature_names)
    labels = [LABELS[n] for n in names]
    print(f"Data: {X.shape[0]} patients, {X.shape[1]} features, target range {y.min():.0f}-{y.max():.0f}")

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_frac=0.2, seed=42)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)
    print(f"Split: train {len(ytr)} | test {len(yte)}")

    # 1. Gradient descent on all 10 features
    gd = LinearRegression(lr=0.05, epochs=20000, tol=1e-10).fit(Xtr_s, ytr)
    print(f"Gradient descent converged in {gd.n_steps} steps, train MSE {gd.loss_history[-1]:.2f}")

    # 2. Exact normal-equation solution -> must match gradient descent
    exact = LinearRegression().fit_closed_form(Xtr_s, ytr)
    max_diff = float(np.max(np.abs(np.r_[gd.w, gd.b] - np.r_[exact.w, exact.b])))
    print(f"Max difference GD vs normal equation: {max_diff:.2e}")

    # 3. 5-fold cross-validation on the training set (more reliable score)
    cv = []
    for tr, va in kfold_indices(len(Xtr), k=5):
        sc = StandardScaler().fit(Xtr[tr])
        m = LinearRegression().fit_closed_form(sc.transform(Xtr[tr]), ytr[tr])
        cv.append(r2(ytr[va], m.predict(sc.transform(Xtr[va]))))
    print(f"5-fold CV R^2: {np.mean(cv):.3f} ± {np.std(cv):.3f}")

    # 4. Test set
    p = gd.predict(Xte_s)
    res = {"r2": r2(yte, p), "rmse": rmse(yte, p), "mae": mae(yte, p), "mse": mse(yte, p)}
    baseline_rmse = rmse(yte, np.full_like(yte, ytr.mean()))
    print("\n=== TEST SET ===")
    for k, v in res.items():
        print(f"{k:5s} {v:.3f}")
    print(f"baseline RMSE (always guess the average): {baseline_rmse:.1f}")

    # 5. Sanity benchmark against scikit-learn
    from sklearn.linear_model import LinearRegression as SkLR
    sk = SkLR().fit(Xtr_s, ytr)
    sk_r2 = r2(yte, sk.predict(Xte_s))
    coef_diff = float(np.max(np.abs(sk.coef_ - gd.w)))
    print(f"sklearn R^2 {sk_r2:.3f} | max coefficient difference {coef_diff:.2e}")

    # 6. One-feature model (BMI) for the hero picture and the loss landscape
    j = names.index("bmi")
    xb, xb_te = Xtr[:, j], Xte[:, j]
    xb_mean, xb_std = xb.mean(), xb.std()
    STEPS = 300
    xs = (xb - xb_mean) / xb_std
    one = LinearRegression(lr=0.05, epochs=STEPS, tol=0, track_path=True).fit(xs[:, None], ytr)
    w_real = one.w[0] / xb_std
    b_real = one.b - w_real * xb_mean
    r2_bmi = r2(yte, w_real * xb_te + b_real)
    raw = LinearRegression(lr=0.0013, epochs=STEPS, tol=0, track_path=True).fit(xb[:, None], ytr)
    raw_opt = LinearRegression().fit_closed_form(xb[:, None], ytr)
    scaled_opt = LinearRegression().fit_closed_form(xs[:, None], ytr)
    print(f"Unscaled GD after {STEPS} steps: w={raw.w[0]:.2f}, b={raw.b:.2f} (best w={raw_opt.w[0]:.2f}, b={raw_opt.b:.2f})")

    viz.plot_hero(np.r_[xb, xb_te], np.r_[ytr, yte], w_real, b_real, r2_bmi, "assets/hero.png")
    viz.plot_loss_landscape((xb, ytr, raw.path, raw_opt.w[0], raw_opt.b),
                            (xs, ytr, one.path, scaled_opt.w[0], scaled_opt.b), STEPS, "assets/loss_landscape.png")
    viz.plot_loss_curve(gd.loss_history, "assets/loss_curve.png")
    viz.plot_pred_vs_actual(yte, p, res["r2"], res["rmse"], "assets/pred_vs_actual.png")
    viz.plot_residuals(yte, p, "assets/residuals.png")
    corr = np.corrcoef(Xtr[:, 4], Xtr[:, 5])[0, 1]
    viz.plot_coefficients(gd.w, labels, "assets/coefficients.png",
                          note=f"Note: total cholesterol and LDL are strongly correlated (r = {corr:.2f}), so their weights partly cancel each other.\n"
                               "This is multicollinearity: a property of the model, not a medical finding.")
    viz.plot_results_card([("R² (test)", f"{res['r2']:.3f}"), ("RMSE", f"{res['rmse']:.1f}"),
                           ("MAE", f"{res['mae']:.1f}"), ("5-fold CV R²", f"{np.mean(cv):.3f}"),
                           ("GD steps", f"{gd.n_steps:,}"), ("GD vs exact gap", f"{max_diff:.3f}")],
                          "assets/results_card.png")
    print("\nVisuals saved in assets/")

    # 7. Save everything the app needs
    np.savez("models/glycocast.npz", w=gd.w, b=gd.b, mean=scaler.mean, std=scaler.std,
             names=np.array(names), labels=np.array(labels),
             fmin=X.min(axis=0), fmax=X.max(axis=0), fmed=np.median(X, axis=0),
             Xte=Xte, yte=yte, bmi_w=w_real, bmi_b=b_real,
             bmi_train=xb, y_train=ytr)                     # used by the gradient descent playground
    out = dict(res, baseline_rmse=baseline_rmse, cv_r2_mean=float(np.mean(cv)), cv_r2_std=float(np.std(cv)),
               cv_folds=[float(c) for c in cv], gd_steps=gd.n_steps, gd_vs_exact=max_diff,
               sklearn_r2=sk_r2, sklearn_coef_diff=coef_diff, r2_bmi_only=r2_bmi,
               bmi_slope=float(w_real), bmi_intercept=float(b_real),
               coefficients={l: float(c) for l, c in zip(labels, gd.w)})
    with open("models/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Model saved to models/glycocast.npz")


if __name__ == "__main__":
    main()
