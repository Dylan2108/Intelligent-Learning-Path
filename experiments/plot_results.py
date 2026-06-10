"""
Generates publication-ready figures from `experiments/results.csv`.

Outputs (in `experiments/figures/`):
  - path_length_bar.png     : mean path length per (instance, algo).
  - summary_table.png       : aggregated metrics as a figure.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CSV = Path("experiments/results.csv")
FIG_DIR = Path("experiments/figures")

_LABEL_MAP: dict[str, str] = {}


def _label(instance: str) -> str:
    """
    Converts 'Backend Developer | init=Python' -> 'Backend (Py)',
    'Backend Developer | init=∅' -> 'Backend (∅)', etc.
    """
    if instance in _LABEL_MAP:
        return _LABEL_MAP[instance]
    parts = instance.split(" | ")
    career = parts[0].strip()
    abbrevs = {
        "ML Engineer": "ML Eng",
        "Data Scientist": "Data Sci",
        "Data Engineer": "Data Eng",
        "MLOps Engineer": "MLOps",
        "NLP Engineer": "NLP Eng",
        "Computer Vision Engineer": "CV Eng",
        "Backend Developer": "Backend",
        "Cloud Architect": "Cloud",
        "DevOps Engineer": "DevOps",
        "Data Analyst": "Data Ana",
    }
    short = abbrevs.get(career, career)
    if len(parts) > 1:
        init = parts[1].replace("init=", "").strip()
        if init == "∅":
            short += " (∅)"
        else:
            skills = [s.strip() for s in init.split(",")]
            abbrev_skills = [s[:2] for s in skills]
            short += " (" + "+".join(abbrev_skills) + ")"
    _LABEL_MAP[instance] = short
    return short


def bar_path_length(df: pd.DataFrame, out: Path) -> None:
    instances = sorted(df["instance"].unique(), key=lambda x: (
        0 if "ML Eng" in x else 1 if "Data Sci" in x else 2 if "Data Eng" in x
        else 3 if "NLP" in x else 4 if "CV" in x else 5 if "MLOps" in x
        else 6 if "DevOps" in x else 7 if "Backend" in x else 8,
    ))
    labels = [_label(i) for i in instances]
    means_a = [
        df[(df["instance"] == i) & (df["algorithm"] == "A*")]["path_length"].mean()
        for i in instances
    ]
    means_g = [
        df[(df["instance"] == i) & (df["algorithm"] == "GA")]["path_length"].mean()
        for i in instances
    ]
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.8), 5))
    x = np.arange(n)
    width = 0.35
    ax.bar(x - width / 2, means_a, width, color="#60a5fa", edgecolor="#2563eb", linewidth=0.8, label="A*")
    ax.bar(x + width / 2, means_g, width, color="#f87171", edgecolor="#dc2626", linewidth=0.8, label="GA")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean path length (courses)", fontsize=11)
    ax.set_title("Path length per instance", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def summary_table(df: pd.DataFrame, out: Path) -> None:
    rows = []
    for (inst, algo), g in df.groupby(["instance", "algorithm"]):
        succ = g["success"].sum()
        rows.append(
            {
                "Instance": _label(inst),
                "Algo": algo,
                "Trials": len(g),
                "Success": int(succ),
                "Mean ms": g.loc[g["success"], "exec_time_ms"].mean(),
                "Mean path": g.loc[g["success"], "path_length"].mean(),
                "Mean time": g.loc[g["success"], "total_time"].mean(),
                "Mean cost": g.loc[g["success"], "total_cost"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    fig_h = max(3, 0.45 * len(summary) + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary.round(1).values,
        colLabels=list(summary.columns),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.25)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e2e8f0")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f8fafc")
    ax.set_title("Aggregated metrics", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=FIG_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    df = df[df["success"]].copy()
    logger.info("Loaded %d successful trials", len(df))

    bar_path_length(df, args.out / "path_length_bar.png")
    logger.info("Wrote path_length_bar.png")
    summary_table(df, args.out / "summary_table.png")
    logger.info("Wrote summary_table.png")


if __name__ == "__main__":
    main()
