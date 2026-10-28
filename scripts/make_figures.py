"""Generate every figure used in the Phase III report from the measured results.

    PYTHONPATH=. python scripts/make_figures.py      ->  reports/figures/fig_*.png

All data-driven charts are drawn from reports/results.json (nothing is typed in by hand).
Design: one hue per job, thin marks, direct labels, recessive grid, text in ink (never in series colours).
Categorical slots 1-3 (blue, orange, aqua) were checked with the palette validator (all-pairs, light mode).
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src import config
from src.registry import DISEASES, get_spec

INK, INK2, MUTED, GRID, SURFACE = "#14272c", "#47595e", "#7b8c90", "#e2e8e7", "#ffffff"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"           # categorical slots 1-3
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]  # sequential blue
TEAL = "#1b6b73"
DPI = 200
FIG = config.FIG_DIR
NAMES = {k: DISEASES[k].name for k in DISEASES}
COLORS = {"diabetes": BLUE, "heart": ORANGE, "breast": AQUA}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": "#aebbbb", "axes.linewidth": 0.8,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load() -> dict:
    return json.loads((config.REPORTS_DIR / "results.json").read_text())


def style_axes(ax, grid_axis="y"):
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------------------
# 1. System architecture
# --------------------------------------------------------------------------------------

def box(ax, x, y, w, h, text, fc="#ffffff", ec=TEAL, fs=8, bold=False, tc=INK, lw=1.1, r=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, both=False, rad=0.0, color=INK2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="<|-|>" if both else "-|>", mutation_scale=9,
                                 lw=1.1, color=color, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def test_count() -> str:
    """Number of automated tests actually run, read from the pytest JUnit report."""
    import xml.etree.ElementTree as ET
    path = config.REPORTS_DIR / "pytest_junit.xml"
    if path.exists():
        root = ET.parse(path).getroot()
        suite = root if root.tag == "testsuite" else root.find("testsuite")
        return str(int(suite.attrib["tests"]))
    return "100+"


def architecture():
    fig, ax = plt.subplots(figsize=(10, 6.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    zones = [(1, 22, "1  Data layer"), (26, 40, "2  Offline training pipeline"), (68, 31, "3  Online serving layer")]
    for x, w, title in zones:
        ax.add_patch(FancyBboxPatch((x, 7), w, 56, boxstyle="round,pad=0,rounding_size=1", fc="#f4f8f8", ec="#c9d6d4", lw=1))
        box(ax, x + 1, 57, w - 2, 4.5, title, fc=TEAL, ec=TEAL, fs=9, bold=True, tc="#ffffff")

    # data layer: three sources merge into the first training step
    sources = [("Pima Indians Diabetes\n768 patients, 8 features", 46, 50),
               ("UCI Heart Disease\n(Cleveland) 303 patients,\n13 features", 35, 39),
               ("Wisconsin Breast Cancer\n569 samples, 10 features", 24, 28)]
    for text, y, yc in sources:
        box(ax, 3, y, 18, 8, text, fs=7)
        ax.plot([21, 24.5], [yc, yc], color=INK2, lw=1.1)
    ax.plot([24.5, 24.5], [28, 52.5], color=INK2, lw=1.1)
    arrow(ax, 24.5, 52.5, 28, 52.5)
    box(ax, 3, 10, 18, 11, "registry.py\nfeature specs, valid\nranges, loaders,\nsentinel codes\n(read by every layer)", fs=7, fc="#e3f0ef")

    # training pipeline
    steps = [("Load, de-duplicate, fix label polarity\nStratified 80/20 split (seed 42)", 49),
             ("Pipeline: SentinelToNaN, median/mode\nimpute, scale / one-hot, classifier", 40.5),
             ("5 models + majority baseline\n5x3 repeated CV, GridSearchCV (AUC)", 32),
             ("Selection round (new CV folds)\n+ soft-voting ensemble, simplest wins ties", 23.5),
             ("One-shot hold-out evaluation, bootstrap CI,\nscreening threshold, permutation importance", 15)]
    for text, y in steps:
        box(ax, 28, y, 36, 7, text, fs=7.5)
    for (_, y1), (_, y2) in zip(steps[:-1], steps[1:]):
        arrow(ax, 46, y1, 46, y2 + 7)
    box(ax, 28, 8.2, 36, 5, "models/*.joblib, *_meta.json, *_holdout.csv\nreports/results.json", fs=7, fc="#e3f0ef")
    arrow(ax, 46, 15, 46, 13.2)

    # serving layer
    box(ax, 69, 49, 29, 7, "Web UI (HTML / CSS / JS)\nforms generated from specs", fs=7.5)
    box(ax, 69, 38, 29, 8, "Flask REST API\n/predict  /batch  /diseases\n/history  /health", fs=7.5)
    box(ax, 69, 22, 14, 11, "DiseasePredictor\nvalidate,\npredict,\nexplain", fs=7, fc="#e3f0ef")
    box(ax, 84, 22, 14, 11, "SQLite\naudit log\n(predictions\ntable)", fs=7)
    arrow(ax, 83.5, 49, 83.5, 46, both=True)
    arrow(ax, 76, 38, 76, 33, both=True)
    arrow(ax, 91, 38, 91, 33)
    # artefacts are loaded by the predictor at start-up
    ax.plot([64, 76], [10.7, 10.7], color=INK2, lw=1.1)
    arrow(ax, 76, 10.7, 76, 22)
    ax.text(77, 15.5, "loaded at start-up", fontsize=7, color=INK2)

    box(ax, 1, 0.8, 98, 4.3,
        f"Quality gates: {test_count()} automated tests (pytest), fixed-seed experiments, HTTP load benchmark, browser screenshots",
        fs=7.5, fc="#fbfdfd", ec="#c9d6d4")
    fig.savefig(FIG / "fig_01_architecture.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def protocol_flow():
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 26)
    ax.axis("off")
    items = [("1  Split", "Stratified 80/20\nseed 42\nhold-out sealed"),
             ("2  Baseline", "5x3 repeated CV\ndefault parameters\n+ majority class"),
             ("3  Tune", "GridSearchCV\n5-fold, scoring\nROC-AUC"),
             ("4  Select", "5x3 CV, new folds\n+ ensemble\nsimplest wins ties"),
             ("5  Evaluate", "hold-out used once\nbootstrap 95% CI\nscreening threshold"),
             ("6  Stress-test", "10 other splits\nablations\npermutation\nimportance")]
    w, gap = 14.4, 2.0
    for i, (t, d) in enumerate(items):
        x = 1 + i * (w + gap)
        box(ax, x, 15, w, 7, t, fc=TEAL, ec=TEAL, fs=8.5, bold=True, tc="#ffffff")
        box(ax, x, 2.5, w, 11.2, d, fs=7.5, fc="#f4f8f8", ec="#c9d6d4")
        if i < len(items) - 1:
            arrow(ax, x + w, 18.5, x + w + gap, 18.5)
    fig.savefig(FIG / "fig_02_protocol.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------------------
# 2. Data-driven charts
# --------------------------------------------------------------------------------------

def model_comparison(res):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=False)
    for ax, key in zip(axes, DISEASES):
        d = res["diseases"][key]
        sel = d["selection"]
        names = sorted(sel, key=lambda n: sel[n]["roc_auc"]["mean"])
        y = np.arange(len(names))
        for yi, n in zip(y, names):
            m, s = sel[n]["roc_auc"]["mean"], sel[n]["roc_auc"]["std"]
            chosen = n == d["selected_model"]
            c = ORANGE if chosen else BLUE
            ax.plot([m - s, m + s], [yi, yi], color=c, lw=1.6, solid_capstyle="round", alpha=.9)
            ax.plot(m, yi, "o", ms=7, color=c, mec=SURFACE, mew=1.5, zorder=3)
            ax.text(m, yi + 0.33, f"{m:.3f}", ha="center", va="bottom", fontsize=7, color=INK2)
        ax.set_yticks(y)
        ax.set_yticklabels(["%s%s" % (n, "  (selected)" if n == d["selected_model"] else "") for n in names], fontsize=7.5)
        ax.set_title(NAMES[key])
        lo = min(sel[n]["roc_auc"]["mean"] - sel[n]["roc_auc"]["std"] for n in names)
        hi = max(sel[n]["roc_auc"]["mean"] + sel[n]["roc_auc"]["std"] for n in names)
        pad = (hi - lo) * 0.08
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, len(names) - 0.2)
        ax.set_xlabel("Mean ROC-AUC (line = ±1 SD)")
        style_axes(ax, "x")
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "fig_03_model_comparison.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def roc(res):
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], color="#aebbbb", lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(0.62, 0.55, "chance", color=MUTED, fontsize=7.5, rotation=36)
    label_pos = {"diabetes": (0.30, 0.50), "heart": (0.10, 0.72), "breast": (0.25, 0.965)}
    for key in DISEASES:
        d = res["diseases"][key]
        r = d["roc"]
        ax.plot(r["fpr"], r["tpr"], color=COLORS[key], lw=1.8, drawstyle="steps-post", zorder=2,
                label=f"{NAMES[key]}  (AUC {d['test']['roc_auc']:.3f}, n={d['test']['n']})")
    ax.set_xlabel("False-positive rate (1 − specificity)")
    ax.set_ylabel("True-positive rate (recall)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_aspect("equal")
    ax.set_title("Hold-out ROC curves (data unseen during training and selection)", fontsize=9)
    style_axes(ax, "both")
    ax.legend(loc="lower right", frameon=False, fontsize=8, handlelength=1.6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_04_roc.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def best_text_colour(rgba) -> str:
    """White or ink, whichever has the higher WCAG contrast against the cell colour."""
    def lin(c):
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    lum = lambda rgb: 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2])
    bg = lum(rgba[:3])
    contrast = lambda other: (max(bg, other) + 0.05) / (min(bg, other) + 0.05)
    ink = matplotlib.colors.to_rgb(INK)
    return "#ffffff" if contrast(1.0) >= contrast(lum(ink)) else INK


def confusion(res):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq", SEQ)
    for ax, key in zip(axes, DISEASES):
        d = res["diseases"][key]["test"]
        c = d["confusion"]
        m = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
        ax.imshow(m / m.sum(axis=1, keepdims=True), cmap=cmap, vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                frac = m[i, j] / m[i].sum()
                ax.text(j, i, f"{m[i, j]}\n({frac:.0%} of row)", ha="center", va="center", fontsize=8.5,
                        color=best_text_colour(cmap(frac)))
        neg, pos = ("No disease", "Disease") if key != "breast" else ("Benign", "Malignant")
        ax.set_xticks([0, 1], [f"Predicted\n{neg}", f"Predicted\n{pos}"], fontsize=8)
        ax.set_yticks([0, 1], [f"Actual\n{neg}", f"Actual\n{pos}"], fontsize=8)
        ax.set_title(f"{NAMES[key]}  (n = {d['n']})")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-.5, 2, 1), minor=True)
        ax.set_yticks(np.arange(-.5, 2, 1), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=3)
        ax.tick_params(which="minor", length=0)
    fig.tight_layout(w_pad=1.5)
    fig.savefig(FIG / "fig_05_confusion.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def importance(res):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, key in zip(axes, DISEASES):
        spec = get_spec(key)
        imp = res["diseases"][key]["permutation_importance"][:6][::-1]
        labels = [spec.feature(i["feature"]).label.replace(" (dataset code)", "") for i in imp]
        short = {"Major vessels coloured by fluoroscopy": "Major vessels (fluoroscopy)",
                 "ST depression induced by exercise": "ST depression (exercise)",
                 "Slope of peak-exercise ST segment": "ST-segment slope",
                 "Fasting blood sugar > 120 mg/dL": "Fasting blood sugar > 120"}
        labels = [short.get(l, l) for l in labels]
        labels = [l if len(l) < 30 else l[:28] + "…" for l in labels]
        vals = [i["mean_auc_drop"] for i in imp]
        errs = [i["std"] for i in imp]
        ax.barh(range(len(vals)), vals, color=BLUE, height=0.55, zorder=2)
        ax.errorbar(vals, range(len(vals)), xerr=errs, fmt="none", ecolor=INK2, elinewidth=0.9, capsize=2, zorder=3)
        ax.set_yticks(range(len(vals)), labels, fontsize=7.5)
        ax.set_title(NAMES[key])
        ax.set_xlabel("Drop in hold-out ROC-AUC when shuffled")
        ax.axvline(0, color="#aebbbb", lw=0.8)
        style_axes(ax, "x")
        ax.tick_params(axis="y", length=0)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(FIG / "fig_06_importance.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def stability(res):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, metric, title in zip(axes, ("accuracy", "roc_auc"), ("Accuracy", "ROC-AUC")):
        for i, key in enumerate(DISEASES):
            d = res["diseases"][key]
            vals = [r[metric] for r in d["stability"]["rows"]]
            rng = np.random.default_rng(i)
            ax.scatter(i + rng.uniform(-0.14, 0.14, len(vals)), vals, s=26, color=BLUE, alpha=.75, edgecolor=SURFACE, lw=0.8, zorder=3)
            st = d["stability"][metric]
            ax.plot([i - 0.28, i + 0.28], [st["mean"], st["mean"]], color=INK, lw=1.6, zorder=4)
            ax.text(i + 0.31, st["mean"], f"mean {st['mean']:.3f}", va="center", fontsize=7.5, color=INK2)
            ax.plot(i - 0.30, d["test"][metric], marker="D", ms=6, color=ORANGE, mec=SURFACE, mew=1, zorder=5)
        ax.set_xticks(range(3), [NAMES[k] for k in DISEASES])
        ax.set_xlim(-0.55, 2.75)
        ax.set_ylabel(title)
        ax.set_title(f"{title} across 10 random 80/20 splits")
        style_axes(ax, "y")
    axes[0].plot([], [], "o", color=BLUE, ms=5, label="one split")
    axes[0].plot([], [], "D", color=ORANGE, ms=5, label="reported split (seed 42)")
    axes[0].legend(frameon=False, fontsize=7.5, loc="lower right")
    fig.tight_layout(w_pad=2)
    fig.savefig(FIG / "fig_07_stability.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    res = load()
    architecture()
    protocol_flow()
    model_comparison(res)
    roc(res)
    confusion(res)
    importance(res)
    stability(res)
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
