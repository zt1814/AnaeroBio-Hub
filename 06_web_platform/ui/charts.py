#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Publication-style Matplotlib figures for the portal (white background so they read well in light and dark mode)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TEAL, AMBER, RED, BLUE, GREY = "#0f766e", "#b45309", "#b42318", "#2563eb", "#6b7f78"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, colors="#26332f")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)


def attribution_chart(contribs: dict, baseline: float, final_pct: float):
    """Horizontal bars of per-modality shifts (% ThGP) relative to the training-set baseline."""
    if not contribs:
        return None
    items = sorted(contribs.items(), key=lambda kv: abs(kv[1]))  # largest effect ends up on top
    labels = [k for k, _ in items]
    vals = [float(v) for _, v in items]
    fig, ax = plt.subplots(figsize=(5.8, 3.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.barh(labels, vals, color=[TEAL if v >= 0 else RED for v in vals], height=0.62, edgecolor="none")
    ax.axvline(0, color="#26332f", linewidth=0.9)
    for y, v in enumerate(vals):
        ax.text(v + (0.6 if v >= 0 else -0.6), y, f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=8, color="#26332f")
    span = max(abs(v) for v in vals) or 1.0
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_xlabel("Shift in % ThGP vs. baseline", fontsize=9, fontweight="bold")
    ax.set_title(f"Modality attribution (baseline {baseline:.1f}% → {final_pct:.1f}%)", fontsize=9.5, fontweight="bold", loc="left")
    _style(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig


def kinetics_chart(days, preds, lag_d: float, cur_day: float, cur_pct: float):
    """Predicted mineralisation trajectory with lag marker, readily-biodegradable threshold and the selected duration."""
    fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=150)
    fig.patch.set_facecolor("white")
    ax.plot(days, preds, color=TEAL if cur_pct >= 0 else RED, marker="o", markersize=4, linewidth=2.1, label="Predicted trajectory")
    ax.axhline(60.0, color=BLUE, linestyle="--", linewidth=1.0, label="Readily biodegradable (60%)")
    ax.axhline(0.0, color=GREY, linestyle=":", linewidth=0.9)
    ax.axvline(lag_d, color=AMBER, linestyle="-.", linewidth=1.1, label=f"Acclimation lag = {lag_d:.1f} d")
    ax.scatter([cur_day], [cur_pct], color=RED, s=60, zorder=5, label=f"Selected ({cur_day:.0f} d)")
    lo = min(-35.0, min(preds) - 5.0)
    ax.set_xlim(0, 185)
    ax.set_ylim(lo, 108)
    ax.set_xlabel("Incubation duration (days)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Mineralisation (% ThGP)", fontsize=9, fontweight="bold")
    ax.set_title("Simulated mineralisation kinetics", fontsize=9.5, fontweight="bold", loc="left")
    ax.legend(loc="lower right", fontsize=7.8, frameon=True, framealpha=0.95)
    _style(ax)
    fig.tight_layout()
    return fig
