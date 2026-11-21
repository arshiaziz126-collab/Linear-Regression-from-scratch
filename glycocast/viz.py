"""Wine red + white visuals."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

# ---- theme ---------------------------------------------------------------
BG = "#FFFFFF"
WINE = "#7B1E3A"        # main wine red
WINE_DARK = "#4A0E1F"
WINE_MID = "#A3354F"
ROSE = "#D99AA8"
BLUSH = "#F6E6EA"
TEXT = "#2B0A13"
MUTED = "#8A5A66"
GRID = "#F1E3E6"

WINE_CMAP = LinearSegmentedColormap.from_list("wine", ["#FFFFFF", BLUSH, ROSE, WINE_MID, WINE, WINE_DARK])


def _fonts():
    names = {f.name for f in font_manager.fontManager.ttflist}
    body = "Inter" if "Inter" in names else "DejaVu Sans"
    head = "Poppins" if "Poppins" in names else body
    return head, body


HEAD, BODY = _fonts()
plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": ROSE, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": TEXT, "font.family": BODY, "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9, "legend.frameon": False,
})


def _title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontfamily=HEAD, fontsize=15, fontweight="bold",
                 color=WINE, pad=24 if sub else 12)
    if sub:
        ax.text(0, 1.025, sub, transform=ax.transAxes, color=MUTED, fontsize=10)


# ---- 1. Hero: best-fit line with residuals -------------------------------
def plot_hero(x, y, w, b, r2_value, path, xlabel="Body mass index (BMI)"):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_axes([0.07, 0.11, 0.62, 0.68])
    order = np.argsort(x)
    pred = w * x + b
    for xi, yi, pi in zip(x, y, pred):                       # residual lines
        ax.plot([xi, xi], [yi, pi], color=ROSE, lw=0.8, alpha=0.7, zorder=1)
    ax.scatter(x, y, s=38, color=WINE_MID, alpha=0.75, edgecolor="white", lw=0.6, zorder=3)
    ax.plot(x[order], pred[order], color=ROSE, lw=9, alpha=0.35, zorder=4)
    ax.plot(x[order], pred[order], color=WINE, lw=3, zorder=5)
    ax.set_xlabel(xlabel); ax.set_ylabel("Diabetes progression after 1 year")

    fig.text(0.07, 0.92, "GlycoCast", fontfamily=HEAD, fontsize=38, fontweight="bold", color=WINE, va="center")
    fig.text(0.07, 0.855, "Linear regression built from scratch in NumPy  ·  predicting diabetes progression",
             fontsize=14, color=MUTED, va="center")

    # side panel
    side = fig.add_axes([0.73, 0.11, 0.24, 0.68]); side.axis("off")
    side.set_xlim(0, 1); side.set_ylim(0, 1)
    side.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.04",
                                  fc=BLUSH, ec="none"))
    side.text(0.08, 0.9, "The learned line", fontsize=12, color=MUTED)
    side.text(0.08, 0.80, f"y = {w:.2f}·BMI {'+' if b >= 0 else '−'} {abs(b):.1f}",
              fontsize=19, fontweight="bold", color=WINE, fontfamily=HEAD)
    side.text(0.08, 0.64, "Meaning", fontsize=12, color=MUTED)
    side.text(0.08, 0.475, f"Every +1 BMI point adds\nabout {w:.1f} points of\ndisease progression.",
              fontsize=13.5, color=TEXT, linespacing=1.45)
    side.text(0.08, 0.30, "Fit (one feature only)", fontsize=12, color=MUTED)
    side.text(0.08, 0.17, f"R² = {r2_value:.2f}", fontsize=26, fontweight="bold", color=WINE, fontfamily=HEAD)
    side.text(0.08, 0.07, "Thin lines = errors (residuals)\nthe model makes as small as possible",
              fontsize=10, color=MUTED, linespacing=1.4)
    fig.savefig(path, dpi=150); plt.close(fig)


# ---- 2. Loss landscape + gradient descent path ---------------------------
def _landscape(ax, x, y, path_wb, w_opt, b_opt, title, sub, pad=(0.35, 0.35)):
    path_wb = np.array(path_wb)
    allw = np.r_[path_wb[:, 0], w_opt]; allb = np.r_[path_wb[:, 1], b_opt]
    dw = (allw.max() - allw.min()) or 1; db = (allb.max() - allb.min()) or 1
    ws = np.linspace(allw.min() - pad[0] * dw, allw.max() + pad[0] * dw, 260)
    bs = np.linspace(allb.min() - pad[1] * db, allb.max() + pad[1] * db, 260)
    W, B = np.meshgrid(ws, bs)
    L = np.mean((W[..., None] * x + B[..., None] - y) ** 2, axis=-1)
    ax.grid(False)
    ax.contourf(W, B, np.log(L), levels=30, cmap=WINE_CMAP)
    ax.contour(W, B, np.log(L), levels=30, colors="white", linewidths=0.45, alpha=0.55)
    ax.plot(path_wb[:, 0], path_wb[:, 1], color="white", lw=4.5, alpha=0.95)
    ax.plot(path_wb[:, 0], path_wb[:, 1], color=WINE_DARK, lw=1.8)
    ax.scatter(*path_wb[0], s=110, color="white", edgecolor=WINE_DARK, lw=2, zorder=5)
    ax.scatter(*path_wb[-1], s=70, color=WINE_DARK, edgecolor="white", lw=1.5, zorder=6)
    ax.scatter([w_opt], [b_opt], s=300, marker="*", color="white", edgecolor=WINE_DARK, lw=1.4, zorder=7)
    ax.set_xlabel("weight w (slope)"); ax.set_ylabel("bias b (intercept)")
    _title(ax, title, sub)


def plot_loss_landscape(raw, scaled, steps, out):
    """raw / scaled = (x, y, path, w_opt, b_opt)"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    _landscape(axes[0], *raw, "Without feature scaling",
               f"narrow valley · bounces sideways, then crawls · nowhere near the star after {steps} steps")
    _landscape(axes[1], *scaled, "With feature scaling",
               f"even bowl · walks straight to the star in {steps} steps")
    for ax in axes:
        ax.legend(handles=[
            plt.Line2D([], [], marker="o", ls="", color="white", markeredgecolor=WINE_DARK, ms=9, label="start (0, 0)"),
            plt.Line2D([], [], marker="o", ls="", color=WINE_DARK, ms=7, label=f"after {steps} steps"),
            plt.Line2D([], [], marker="*", ls="", color="white", markeredgecolor=WINE_DARK, ms=15, label="lowest error")],
            loc="upper right", labelcolor=TEXT, frameon=True, facecolor="white", edgecolor=BLUSH)
    fig.text(0.5, 0.005, "Background = error (MSE) of every possible line, lighter = lower error.  "
             "Same data (BMI → progression), same number of gradient-descent steps.",
             ha="center", color=MUTED, fontsize=10.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(out, dpi=150); plt.close(fig)


# ---- 3. Loss curve -------------------------------------------------------
def plot_loss_curve(history, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    ep = np.arange(1, len(history) + 1)
    ax.fill_between(ep, history, min(history), color=BLUSH)
    ax.plot(ep, history, color=WINE, lw=2.6)
    ax.set_yscale("log"); ax.set_xscale("log")
    ax.set_xlabel("step (log scale)"); ax.set_ylabel("training MSE (log scale)")
    ax.text(ep[-1], history[-1], f"  converged\n  MSE {history[-1]:,.0f}", color=WINE, fontsize=10, va="bottom", ha="right")
    _title(ax, "Error going down", "gradient descent on all 10 features")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---- 4. Predicted vs actual ---------------------------------------------
def plot_pred_vs_actual(y, p, r2_value, rmse_value, out):
    fig, ax = plt.subplots(figsize=(7, 6.5))
    lo, hi = min(y.min(), p.min()) - 10, max(y.max(), p.max()) + 10
    ax.plot([lo, hi], [lo, hi], color=ROSE, ls="--", lw=1.5, label="perfect prediction")
    ax.scatter(y, p, s=46, color=WINE, alpha=0.75, edgecolor="white", lw=0.7)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Actual progression"); ax.set_ylabel("Predicted progression")
    ax.text(0.04, 0.93, f"R² = {r2_value:.3f}\nRMSE = {rmse_value:.1f}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color=WINE, va="top", fontfamily=HEAD)
    ax.legend(loc="lower right", labelcolor=MUTED)
    _title(ax, "Predicted vs actual", "held-out test patients")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---- 5. Residuals --------------------------------------------------------
def plot_residuals(y, p, out):
    res = y - p
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].axhline(0, color=ROSE, lw=1.5, ls="--")
    axes[0].scatter(p, res, s=40, color=WINE, alpha=0.7, edgecolor="white", lw=0.6)
    axes[0].set_xlabel("Predicted progression"); axes[0].set_ylabel("Error (actual − predicted)")
    _title(axes[0], "Residuals vs prediction", "no clear pattern = a linear model is reasonable")
    axes[1].hist(res, bins=18, color=WINE_MID, edgecolor="white")
    axes[1].axvline(0, color=WINE_DARK, lw=1.5)
    axes[1].set_xlabel("Error (actual − predicted)"); axes[1].set_ylabel("patients")
    _title(axes[1], "Error distribution", "centred near zero = no systematic bias")
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---- 6. Coefficients -----------------------------------------------------
def plot_coefficients(w, labels, out, note=None):
    order = np.argsort(np.abs(w))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_axisbelow(True); ax.grid(axis="y", visible=False)
    colors = [WINE if v > 0 else ROSE for v in w[order]]
    ax.barh(np.arange(len(w)), w[order], color=colors, height=0.62)
    ax.axvline(0, color=TEXT, lw=1)
    ax.set_yticks(np.arange(len(w)), [labels[i] for i in order], color=TEXT)
    span = np.abs(w).max()
    for i, v in enumerate(w[order]):
        ax.text(v + (0.02 * span if v >= 0 else -0.02 * span), i, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5, color=MUTED)
    ax.set_xlim(-span * 1.25, span * 1.25)
    ax.set_xlabel("effect on progression for +1 standard deviation of the feature")
    _title(ax, "What drives diabetes progression",
           "wine = pushes progression up · rose = pushes it down")
    if note:
        fig.text(0.02, 0.012, note, color=MUTED, fontsize=9.5, ha="left", va="bottom")
        fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(out, dpi=150); plt.close(fig); return
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


# ---- 7. Results card -----------------------------------------------------
def plot_results_card(tiles, out):
    fig = plt.figure(figsize=(14, 4.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.03, 0.86, "GlycoCast · test-set results", fontfamily=HEAD, fontsize=22, fontweight="bold", color=WINE)
    ax.text(0.03, 0.74, "Unseen patients · NumPy only · gradient descent matches the exact solution",
            fontsize=11.5, color=MUTED)
    wdt = 0.94 / len(tiles)
    for k, (label, value) in enumerate(tiles):
        x0 = 0.03 + k * wdt
        ax.add_patch(FancyBboxPatch((x0, 0.1), wdt - 0.015, 0.5, boxstyle="round,pad=0.005,rounding_size=0.02",
                                    fc=BLUSH, ec="none"))
        ax.text(x0 + 0.015, 0.48, label, fontsize=11, color=MUTED)
        ax.text(x0 + 0.015, 0.24, value, fontsize=23, fontweight="bold", color=WINE, fontfamily=HEAD)
    fig.savefig(out, dpi=150); plt.close(fig)


# ---- 8. Per-patient contributions (app) ----------------------------------
def plot_contributions(contrib, labels, base, pred, out):
    order = np.argsort(np.abs(contrib))
    fig, ax = plt.subplots(figsize=(9, 5.6))
    ax.set_axisbelow(True); ax.grid(axis="y", visible=False)
    ax.barh(np.arange(len(contrib)), contrib[order], height=0.62,
            color=[WINE if v > 0 else ROSE for v in contrib[order]])
    ax.axvline(0, color=TEXT, lw=1)
    ax.set_yticks(np.arange(len(contrib)), [labels[i] for i in order], color=TEXT)
    span = max(np.abs(contrib).max(), 1)
    for i, v in enumerate(contrib[order]):
        ax.text(v + (0.03 * span if v >= 0 else -0.03 * span), i, f"{v:+.0f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9.5, color=MUTED)
    ax.set_xlim(-span * 1.3, span * 1.3)
    ax.set_xlabel("points added to / removed from the average patient")
    _title(ax, "Why this prediction", f"average patient {base:.0f}  →  this patient {pred:.0f}")
    fig.tight_layout(); fig.savefig(out, dpi=140); plt.close(fig)


# ---- 9. Gradient descent playground (app) --------------------------------
def plot_playground(x, y, path_wb, w_opt, b_opt, losses, out, scaled):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [1.25, 1]})
    sub = "scaled feature" if scaled else "raw feature (no scaling)"
    _landscape(axes[0], x, y, path_wb, w_opt, b_opt, "Path on the error surface", sub)
    ax = axes[1]
    ok = np.isfinite(losses)
    ax.plot(np.arange(1, len(losses) + 1)[ok], np.array(losses)[ok], color=WINE, lw=2.4)
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("MSE (log scale)")
    best = np.mean((w_opt * x + b_opt - y) ** 2)
    ax.axhline(best, color=ROSE, ls="--", lw=1.5)
    ax.text(len(losses), best, "best possible ", color=MUTED, ha="right", va="bottom", fontsize=9.5)
    _title(ax, "Error per step", f"final MSE {losses[-1]:,.0f}  ·  best {best:,.0f}")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)
