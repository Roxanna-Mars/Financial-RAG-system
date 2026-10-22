"""
Regression tracker — persists RAGAS evaluation runs to CSV and plots
metric trends across configs so degradations are caught immediately.

CSV schema (one row per run):
  timestamp, dataset, config_name, n_samples,
  faithfulness, answer_relevancy, context_utilization

Usage
-----
from finrag.regression import RegressionTracker

tracker = RegressionTracker()          # default: results/ragas_regression.csv
tracker.record(ragas_result)           # append one run
tracker.compare_configs(              # print table + save chart
    dataset="financebench",
    output_path="results/ragas_comparison.png",
)
"""

import logging
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for CLI / no display
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from .ragas_eval import RAGASResult

logger = logging.getLogger(__name__)

_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "results" / "ragas_regression.csv"

_METRICS = ["faithfulness", "answer_relevancy", "context_utilization"]
_COLORS  = {"faithfulness": "#4C72B0", "answer_relevancy": "#DD8452", "context_utilization": "#55A868"}

_CSV_COLS = [
    "timestamp", "dataset", "config_name", "model", "n_samples",
    "faithfulness", "answer_relevancy", "context_utilization",
]


class RegressionTracker:
    """
    Append-only log of RAGAS evaluation runs with comparison utilities.

    Parameters
    ----------
    csv_path : path to the regression CSV (created if absent)
    """

    def __init__(self, csv_path: str | Path = _DEFAULT_CSV) -> None:
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.csv_path.exists():
            pd.DataFrame(columns=_CSV_COLS).to_csv(self.csv_path, index=False)
            logger.info("Regression CSV initialised at %s", self.csv_path)

    # ── Record ──────────────────────────────────────────────────────────────

    def record(self, result: RAGASResult) -> None:
        """Append one RAGASResult to the regression CSV."""
        row = {
            "timestamp":            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset":              result.dataset,
            "config_name":          result.config_name,
            "model":                result.model,
            "n_samples":            result.n_samples,
            "faithfulness":         round(result.faithfulness, 6),
            "answer_relevancy":     round(result.answer_relevancy, 6),
            "context_utilization":  round(result.context_utilization, 6),
        }
        pd.DataFrame([row]).to_csv(self.csv_path, mode="a", header=False, index=False)
        logger.info(
            "Regression row appended  |  dataset=%s  config=%s  model=%s  "
            "faith=%.4f  rel=%.4f  util=%.4f",
            result.dataset, result.config_name, result.model,
            result.faithfulness, result.answer_relevancy, result.context_utilization,
        )

    # ── Load ────────────────────────────────────────────────────────────────

    def load(self, dataset: str | None = None) -> pd.DataFrame:
        """
        Load regression history.

        Parameters
        ----------
        dataset : filter to this dataset; None returns all rows.
        """
        df = pd.read_csv(self.csv_path, parse_dates=["timestamp"])
        if dataset:
            df = df[df["dataset"] == dataset]
        return df.reset_index(drop=True)

    # ── Compare configs ──────────────────────────────────────────────────────

    def compare_configs(
        self,
        dataset: str,
        *,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> pd.DataFrame:
        """
        Print a comparison table and optionally save a bar chart.

        Each config is represented by its **most recent** run (deduplication
        by config_name keeps the latest timestamp).

        Parameters
        ----------
        dataset     : dataset label to filter on (e.g. "financebench")
        output_path : if given, save the bar-chart PNG to this path
        show        : display the chart interactively (requires a display)

        Returns
        -------
        pd.DataFrame  with one row per config and metric columns
        """
        df = self.load(dataset=dataset)
        if df.empty:
            logger.warning("No regression data found for dataset='%s'.", dataset)
            return df

        # Keep the latest run per config
        df = (
            df.sort_values("timestamp")
              .drop_duplicates(subset=["config_name"], keep="last")
              .set_index("config_name")
        )

        # ── Console table ────────────────────────────────────────────────────
        header = f"{'Config':<22}  {'Model':<34}  {'Faithful':>10}  {'Relevancy':>10}  {'Ctx Util':>10}  {'N':>5}"
        sep    = "─" * len(header)
        print(f"\n{sep}")
        print(f"  RAGAS Comparison  |  dataset={dataset}")
        print(sep)
        print(f"  {header}")
        print(f"  {sep}")
        for cfg, row in df.iterrows():
            model_str = str(row.get("model", "unknown"))
            print(
                f"  {cfg:<22}  {model_str:<34}  {row['faithfulness']:>10.4f}  "
                f"{row['answer_relevancy']:>10.4f}  "
                f"{row['context_utilization']:>10.4f}  "
                f"{int(row['n_samples']):>5}"
            )
        print(f"  {sep}\n")

        # ── Bar chart ────────────────────────────────────────────────────────
        if output_path or show:
            _plot_comparison(df, dataset=dataset, output_path=output_path, show=show)

        return df.reset_index()

    # ── Trend (all runs for one config) ──────────────────────────────────────

    def trend(
        self,
        dataset: str,
        config_name: str,
        *,
        output_path: str | Path | None = None,
        show: bool = False,
    ) -> pd.DataFrame:
        """
        Plot metric trend over successive runs of one config.

        Useful for tracking improvement as the pipeline is tuned.
        """
        df = self.load(dataset=dataset)
        df = df[df["config_name"] == config_name].sort_values("timestamp")

        if df.empty:
            logger.warning(
                "No data for dataset=%s config=%s", dataset, config_name
            )
            return df

        if output_path or show:
            _plot_trend(df, dataset=dataset, config_name=config_name,
                        output_path=output_path, show=show)

        return df.reset_index(drop=True)


# ── Private plotting helpers ──────────────────────────────────────────────────

def _plot_comparison(
    df: pd.DataFrame,
    *,
    dataset: str,
    output_path: str | Path | None,
    show: bool,
) -> None:
    """Grouped bar chart: one group per config, three bars per group (metrics)."""
    configs = df.index.tolist()
    n       = len(configs)
    x       = range(n)
    width   = 0.25

    fig, ax = plt.subplots(figsize=(max(6, n * 1.8 + 2), 5))

    for i, metric in enumerate(_METRICS):
        vals = df[metric].tolist()
        bars = ax.bar(
            [xi + i * width for xi in x],
            vals,
            width,
            label=metric.replace("_", " ").title(),
            color=_COLORS[metric],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        # Value labels on bars
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=7.5, color="0.25",
            )

    ax.set_xticks([xi + width for xi in x])
    ax.set_xticklabels(configs, rotation=15, ha="right", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"RAGAS Metrics by Config  |  {dataset}", fontsize=11, pad=10)
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    _save_or_show(fig, output_path, show)


def _plot_trend(
    df: pd.DataFrame,
    *,
    dataset: str,
    config_name: str,
    output_path: str | Path | None,
    show: bool,
) -> None:
    """Line chart of metric values over successive runs for one config."""
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.8 + 3), 4))

    for metric in _METRICS:
        ax.plot(
            df["timestamp"].dt.strftime("%m-%d %H:%M"),
            df[metric],
            marker="o",
            label=metric.replace("_", " ").title(),
            color=_COLORS[metric],
            linewidth=1.6,
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_xlabel("Run timestamp (UTC)", fontsize=9)
    ax.set_title(
        f"RAGAS Trend  |  {dataset}  |  config={config_name}",
        fontsize=11, pad=10,
    )
    ax.legend(fontsize=9, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(rotation=25, ha="right", fontsize=8)

    fig.tight_layout()
    _save_or_show(fig, output_path, show)


def _save_or_show(fig: plt.Figure, output_path: str | Path | None, show: bool) -> None:
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        logger.info("Chart saved → %s", p)
    if show:
        plt.show()
    plt.close(fig)
