import os
import math
import shutil
import pandas as pd
import matplotlib.pyplot as plt

Z_95 = 1.96  # 95% normal-approx critical value

def _slugify(text: str) -> str:
    return (
        str(text).lower()
        .replace(" ", "_")
        .replace("/", "-")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace(":", "")
        .replace("|", "-")
    )

def _mean_half_ci(series) -> tuple[float, float, int]:
    """
    Return (mean, half_width_95CI, n) ignoring NaNs.
    Ensures numeric dtype (coerce errors to NaN).
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = int(s.shape[0])
    if n == 0:
        return float("nan"), float("nan"), 0
    m = float(s.mean())
    if n == 1:
        return m, 0.0, 1
    sd = float(s.std(ddof=1))
    half = Z_95 * sd / math.sqrt(n)
    return m, half, n

def _two_bar_with_error(ax, labels, means, halves, title, ylabel):
    """
    Two bars with 95% CI error bars and numeric mean labels on top.
    Uses thicker error lines/caps and lifts text above the caps.
    """
    error_kw = dict(elinewidth=2.0, ecolor="black", capthick=2.0)
    bars = ax.bar(labels, means, yerr=halves, capsize=8, alpha=0.8, edgecolor="black", error_kw=error_kw)

    # Compute a y ceiling that includes caps
    tops = []
    for m, h in zip(means, halves):
        h_eff = 0.0 if (h is None or (isinstance(h, float) and math.isnan(h))) else h
        tops.append(m + max(h_eff, 0.0))
    y_max = max(tops) if tops else 0.0
    pad = 0.05 * y_max if y_max > 0 else 0.05
    ax.set_ylim(0, (y_max + pad) if y_max > 0 else 1)

    # Place labels above cap (not on it), so they don't hide the line
    for rect, m, h in zip(bars, means, halves):
        h_eff = 0.0 if (h is None or (isinstance(h, float) and math.isnan(h))) else h
        label_y = m + max(h_eff, 0.0) + pad * 0.3
        ax.text(rect.get_x() + rect.get_width() / 2, label_y, f"{m:.3g}",
                ha="center", va="bottom", fontsize=10)

    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)

def generate_policy_comparison_plots(
    xlsx_path: str = "replication_results.xlsx",
    sheet: str = "results",
    output_dir: str = "plots",
    overwrite_dir: bool = True,
) -> list[str]:
    """
    Reads per-replication results, groups by 'comparison', and writes 4 PNGs per comparison:
      1) Average Mean RT
      2) Average 90th Percentile RT
      3) Average Queue Size
      4) Average Max Queue Size

    Each plot shows a bar per policy with its 95% CI as an error line and the mean value on top.
    Returns list of saved file paths.
    """
    df = pd.read_excel(xlsx_path, sheet_name=sheet)

    if overwrite_dir and os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
    os.makedirs(output_dir, exist_ok=True)

    saved = []

    # Define metric pairs: (title, p1_col, p2_col, ylabel)
    metrics = [
        ("Average Mean RT",            "p1_mean_RT",        "p2_mean_RT",        "Minutes"),
        ("Average 90th Percentile RT", "p1_percentile_90",  "p2_percentile_90",  "Minutes"),
        ("Average Queue Size",         "p1_avg_queue",      "p2_avg_queue",      "Queue length"),
        ("Average Max Queue Size",     "p1_max_queue",      "p2_max_queue",      "Queue length"),
    ]

    for comp, g in df.groupby("comparison"):
        # Parse policy names from "PolicyA vs PolicyB"
        if " vs " in comp:
            policy1, policy2 = [s.strip() for s in comp.split(" vs ", 1)]
        elif "vs" in comp:
            policy1, policy2 = [s.strip() for s in comp.split("vs", 1)]
        else:
            policy1, policy2 = "Policy1", "Policy2"

        for title, p1_col, p2_col, ylabel in metrics:
            # Skip gracefully if a metric column is missing (e.g., avg_queue not recorded)
            if p1_col not in g.columns or p2_col not in g.columns:
                continue

            m1, h1, n1 = _mean_half_ci(g[p1_col])
            m2, h2, n2 = _mean_half_ci(g[p2_col])

            fig, ax = plt.subplots()
            _two_bar_with_error(
                ax,
                [policy1, policy2],
                [float(m1), float(m2)],
                [float(h1), float(h2)],
                f"{comp} — {title}",
                ylabel,
            )

            fname = os.path.join(output_dir, f"{_slugify(comp)}_{_slugify(title)}.png")
            fig.savefig(fname, bbox_inches="tight", dpi=150)
            plt.close(fig)
            saved.append(fname)

    return saved