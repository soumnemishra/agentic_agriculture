"""
Agricultural Cognitive Architecture (ACA) v1.0 — Empirical Visualization Suite
==============================================================================

Generates publication-quality figures and plots (300 DPI) from the 100-step
closed-loop evaluation artifacts (experiment_results.csv and experiment_traces.jsonl).

Outputs generated in `./figures/`:
  1. fig1_environmental_dynamics.png          - 4-channel microclimate time series
  2. fig2_bayesian_confidence_dynamics.png    - Vision Prior vs Bayesian Posterior tracking
  3. fig3_latency_breakdown_by_layer.png      - Latency distribution across cognitive layers
  4. fig4_diagnosis_and_action_dist.png       - Diagnostic & physical intervention distributions
  5. fig5_prior_vs_posterior_scatter.png      - Likelihood modulation & belief shift scatter
  6. aca_evaluation_dashboard.png             - 6-panel comprehensive research dashboard
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

# Set overall publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#333333"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.color"] = "#E0E0E0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.alpha"] = 0.7


def ensure_output_dir(output_dir: str = "figures") -> Path:
    """Ensure destination directory exists."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(csv_path: str = "datasets/experiment_results.csv") -> pd.DataFrame:
    """Load and preprocess the experiment CSV output."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Experiment results CSV not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def plot_environmental_dynamics(df: pd.DataFrame, out_dir: Path) -> str:
    """Figure 1: Synchronized 4-channel physical microclimate trajectory."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    steps = df["step_index"]

    # 1. Temperature
    axes[0].plot(steps, df["env_temp_c"], color="#D9534F", lw=2, label="Ambient Temp (°C)")
    axes[0].plot(steps, df["soil_temp_c"], color="#F0AD4E", lw=1.5, linestyle="--", label="Soil Temp (°C)")
    axes[0].set_ylabel("Temp (°C)", fontsize=11, fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].set_title("Physical Microclimate & Telemetry Trajectory (100 Experimental Steps)", fontsize=14, fontweight="bold", pad=12)

    # 2. Relative Humidity & Soil Moisture
    axes[1].plot(steps, df["env_humidity_pct"], color="#0275D8", lw=2, label="Relative Humidity (%)")
    axes[1].plot(steps, df["soil_moisture_pct"], color="#5BC0DE", lw=1.5, linestyle="-.", label="Soil Moisture (%)")
    axes[1].set_ylabel("Moisture (%)", fontsize=11, fontweight="bold")
    axes[1].legend(loc="upper right", frameon=True)

    # 3. Solar / Battery & Water TDS
    axes[2].plot(steps, df["solar_battery_v"], color="#5CB85C", lw=2, label="Solar Battery (V)")
    axes[2].set_ylabel("Battery (V)", fontsize=11, fontweight="bold")
    axes[2].legend(loc="upper right", frameon=True)

    # 4. Light Intensity & Water TDS
    ax4_twin = axes[3].twinx()
    axes[3].plot(steps, df["env_light_lux"], color="#F7B731", lw=2, label="Light Intensity (Lux)")
    ax4_twin.plot(steps, df["water_tds_mg_l"], color="#8854D0", lw=1.5, linestyle=":", label="Water TDS (mg/L)")
    axes[3].set_ylabel("Light (Lux)", fontsize=11, fontweight="bold")
    ax4_twin.set_ylabel("TDS (mg/L)", fontsize=11, fontweight="bold", color="#8854D0")
    axes[3].set_xlabel("Experimental Step Index (Synchronized IoT Cycles)", fontsize=12, fontweight="bold")

    # Add phase boundary annotations
    for ax in axes:
        ax.axvspan(1, 20, color="#E8F8F5", alpha=0.5, label="Ph1: Late Blight Surge" if ax == axes[0] else "")
        ax.axvspan(21, 40, color="#FDEDEC", alpha=0.5, label="Ph2: High Humidity Stress" if ax == axes[0] else "")
        ax.axvspan(41, 60, color="#FEF9E7", alpha=0.5, label="Ph3: Viral Pressure" if ax == axes[0] else "")
        ax.axvspan(61, 100, color="#EAFAF1", alpha=0.5, label="Ph4-5: Recovery & Healthy" if ax == axes[0] else "")

    out_file = out_dir / "fig1_environmental_dynamics.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return str(out_file)


def plot_bayesian_confidence(df: pd.DataFrame, out_dir: Path) -> str:
    """Figure 2: Vision Prior vs Bayesian Posterior Confidence Dynamics."""
    fig, ax = plt.subplots(figsize=(12, 6))

    steps = df["step_index"]

    ax.plot(steps, df["reasoning_prior"], color="#7F8C8D", lw=1.8, linestyle="--", label="Computer Vision Prior $P(H)$ (CondConViT_V2)", alpha=0.85)
    ax.plot(steps, df["reasoning_posterior"], color="#27AE60", lw=2.5, label="Bayesian Posterior $P(H|E)$ (Microclimate Fused)")

    # Highlight Likelihood Boost vs Suppression
    boost_mask = df["reasoning_posterior"] > df["reasoning_prior"]
    suppress_mask = df["reasoning_posterior"] < df["reasoning_prior"]

    ax.fill_between(steps, df["reasoning_prior"], df["reasoning_posterior"], where=boost_mask, color="#2ECC71", alpha=0.3, label="Agronomic Etiology Confirmation ($L > 1.0$)")
    ax.fill_between(steps, df["reasoning_prior"], df["reasoning_posterior"], where=suppress_mask, color="#E74C3C", alpha=0.3, label="Microclimate Contradiction Suppression ($L < 1.0$)")

    ax.set_title("Epistemic Belief Revision: Vision Prior vs Multi-Modal Posterior", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Experimental Step Index", fontsize=12, fontweight="bold")
    ax.set_ylabel("Confidence / Probability [0.0 – 1.0]", fontsize=12, fontweight="bold")
    ax.set_ylim(0.3, 1.05)
    ax.legend(loc="lower right", frameon=True, fontsize=10)

    out_file = out_dir / "fig2_bayesian_confidence_dynamics.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return str(out_file)


def plot_latency_breakdown(df: pd.DataFrame, out_dir: Path) -> str:
    """Figure 3: Latency Decomposition across Cognitive Pipeline Layers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    latency_cols = {
        "Perception\n(CondConViT_V2)": df["vision_latency_ms"],
        "Reasoning\n(Sensor Fusion)": df["reasoning_latency_ms"],
        "Planning\n(Action Matrix)": df["planning_latency_ms"],
        "Execution\n(Actuators)": df["execution_latency_ms"],
    }
    lat_df = pd.DataFrame(latency_cols)

    # 1. Bar plot of Mean Latencies
    means = lat_df.mean()
    errors = lat_df.std()
    colors = ["#3498DB", "#9B59B6", "#E67E22", "#1ABC9C"]

    bars = ax1.bar(means.index, means.values, yerr=errors.values, capsize=5, color=colors, edgecolor="#2C3E50", alpha=0.85)
    ax1.set_ylabel("Latency (Milliseconds / Cycle)", fontsize=11, fontweight="bold")
    ax1.set_title("Mean Execution Latency per Cognitive Layer", fontsize=13, fontweight="bold", pad=10)
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f"{yval:.2f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 2. Cumulative End-to-End Latency Histogram & KDE
    sns.histplot(df["total_cycle_latency_ms"], kde=True, ax=ax2, color="#2C3E50", bins=20, stat="density", edgecolor="white")
    median_lat = df["total_cycle_latency_ms"].median()
    ax2.axvline(median_lat, color="#E74C3C", lw=2, linestyle="--", label=f"Median: {median_lat:.2f} ms")
    ax2.set_xlabel("Total End-to-End Cycle Latency (ms)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
    ax2.set_title("End-to-End Closed-Loop Cycle Latency Distribution", fontsize=13, fontweight="bold", pad=10)
    ax2.legend(loc="upper right", frameon=True)

    out_file = out_dir / "fig3_latency_breakdown_by_layer.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return str(out_file)


def plot_distributions(df: pd.DataFrame, out_dir: Path) -> str:
    """Figure 4: Diagnostic Class and Physical Intervention Distributions."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Pathogen Distribution
    diag_counts = df["vision_predicted_class"].value_counts()
    colors1 = ["#2ECC71", "#E74C3C", "#9B59B6", "#F39C12"][: len(diag_counts)]
    wedges, texts, autotexts = ax1.pie(
        diag_counts.values,
        labels=[l.replace("_", " ") for l in diag_counts.index],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors1,
        textprops=dict(fontweight="bold"),
        wedgeprops=dict(width=0.6, edgecolor="white", linewidth=2),
    )
    ax1.set_title("Pathogen Diagnosis Distribution (N=100)", fontsize=13, fontweight="bold", pad=10)

    # Physical Intervention Actions
    act_counts = df["planning_action"].value_counts()
    colors2 = ["#3498DB", "#E74C3C", "#8E44AD", "#F1C40F"][: len(act_counts)]
    bars = ax2.barh([a.replace("_", " ") for a in act_counts.index], act_counts.values, color=colors2, edgecolor="#2C3E50", alpha=0.85)
    ax2.set_xlabel("Dispatched Intervention Count", fontsize=11, fontweight="bold")
    ax2.set_title("Physical Actuator Interventions Dispatched", fontsize=13, fontweight="bold", pad=10)
    for bar in bars:
        w = bar.get_width()
        ax2.text(w + 1, bar.get_y() + bar.get_height() / 2.0, f"{int(w)}", ha="left", va="center", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, max(act_counts.values) + 10)

    out_file = out_dir / "fig4_diagnosis_and_action_dist.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return str(out_file)


def plot_prior_posterior_scatter(df: pd.DataFrame, out_dir: Path) -> str:
    """Figure 5: Scatter of Prior vs Posterior with Likelihood Contours."""
    fig, ax = plt.subplots(figsize=(8, 6.5))

    scatter = ax.scatter(
        df["reasoning_prior"],
        df["reasoning_posterior"],
        c=df["reasoning_likelihood"],
        cmap="coolwarm",
        s=70,
        edgecolors="#2C3E50",
        alpha=0.9,
    )

    # Diagonal unity line
    ax.plot([0.3, 1.0], [0.3, 1.0], color="#7F8C8D", linestyle=":", lw=2, label="Unity Line ($P(H|E) = P(H)$)")

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Microclimate Likelihood Multiplier ($L$)", fontsize=11, fontweight="bold")

    ax.set_title("Bayesian Belief Shift Modulated by Microclimate Evidence", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Vision Prior Confidence $P(H)$", fontsize=11, fontweight="bold")
    ax.set_ylabel("Fused Posterior Confidence $P(H|E)$", fontsize=11, fontweight="bold")
    ax.set_xlim(0.35, 1.02)
    ax.set_ylim(0.35, 1.02)
    ax.legend(loc="upper left", frameon=True)

    out_file = out_dir / "fig5_prior_vs_posterior_scatter.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()
    return str(out_file)


def plot_comprehensive_dashboard(df: pd.DataFrame, out_dir: Path) -> str:
    """Figure 6: Full 6-Panel Academic Dashboard Figure for Paper."""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    steps = df["step_index"]

    # 1. Top-Left: Environmental Telemetry
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, df["env_temp_c"], color="#E74C3C", label="Temp (°C)", lw=1.8)
    ax1.plot(steps, df["env_humidity_pct"], color="#3498DB", label="Humidity (%)", lw=1.8)
    ax1.plot(steps, df["soil_moisture_pct"], color="#1ABC9C", label="Soil Moisture (%)", lw=1.8)
    ax1.set_title("(a) Multi-Modal IoT Telemetry Stream", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Sensor Reading", fontsize=10)
    ax1.legend(loc="lower left", fontsize=8, frameon=True)

    # 2. Top-Right: Bayesian Belief Revision
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, df["reasoning_prior"], color="#95A5A6", linestyle="--", label="Vision Prior $P(H)$", lw=1.5)
    ax2.plot(steps, df["reasoning_posterior"], color="#27AE60", label="Fused Posterior $P(H|E)$", lw=2.0)
    ax2.set_title("(b) Epistemic Bayesian Belief Updating", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Confidence [0..1]", fontsize=10)
    ax2.legend(loc="lower right", fontsize=8, frameon=True)

    # 3. Mid-Left: Latency per Cognitive Layer
    ax3 = fig.add_subplot(gs[1, 0])
    lat_data = [df["vision_latency_ms"], df["reasoning_latency_ms"], df["planning_latency_ms"], df["execution_latency_ms"]]
    bp = ax3.boxplot(lat_data, tick_labels=["Perception", "Reasoning", "Planning", "Execution"], patch_artist=True)
    colors = ["#3498DB", "#9B59B6", "#E67E22", "#1ABC9C"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_title("(c) Latency Profile by Cognitive Layer", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Latency (ms)", fontsize=10)

    # 4. Mid-Right: End-to-End Closed-Loop Latency Distribution
    ax4 = fig.add_subplot(gs[1, 1])
    sns.histplot(df["total_cycle_latency_ms"], kde=True, ax=ax4, color="#34495E", bins=20)
    ax4.axvline(df["total_cycle_latency_ms"].median(), color="#E74C3C", linestyle="--", label=f"Median: {df['total_cycle_latency_ms'].median():.1f} ms")
    ax4.set_title("(d) Total Cognitive Cycle Latency Distribution", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Cycle Latency (ms)", fontsize=10)
    ax4.legend(fontsize=8, frameon=True)

    # 5. Bottom-Left: Pathogen Class Distribution
    ax5 = fig.add_subplot(gs[2, 0])
    diag_counts = df["vision_predicted_class"].value_counts()
    ax5.pie(
        diag_counts.values,
        labels=[l.replace("_", " ") for l in diag_counts.index],
        autopct="%1.0f%%",
        colors=["#2ECC71", "#E74C3C", "#9B59B6"],
        wedgeprops=dict(width=0.5, edgecolor="white"),
        textprops=dict(fontweight="bold", fontsize=9),
    )
    ax5.set_title("(e) Classified Pathogen Proportions", fontsize=12, fontweight="bold")

    # 6. Bottom-Right: Actuator Interventions
    ax6 = fig.add_subplot(gs[2, 1])
    act_counts = df["planning_action"].value_counts()
    ax6.barh([a.replace("_", " ") for a in act_counts.index], act_counts.values, color=["#2980B9", "#C0392B", "#8E44AD"], alpha=0.85)
    ax6.set_title("(f) Physical Actuator Directives Dispatched", fontsize=12, fontweight="bold")
    ax6.set_xlabel("Dispatches", fontsize=10)

    fig.suptitle("Agricultural Cognitive Architecture (ACA) v1.0 — 100-Cycle Evaluation Dashboard", fontsize=15, fontweight="bold", y=0.98)

    out_file = out_dir / "aca_evaluation_dashboard.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close()
    return str(out_file)


def generate_all_plots(csv_path: str = "datasets/experiment_results.csv", out_dir: str = "figures") -> List[str]:
    """Execute full visualization pipeline and generate all artifacts."""
    out_path = ensure_output_dir(out_dir)
    print(f"[*] Loading dataset: {csv_path}...")
    df = load_dataset(csv_path)
    print(f"[*] Successfully loaded {len(df)} cycles.")

    generated = []
    print("[1/6] Generating Figure 1: Environmental Dynamics...")
    generated.append(plot_environmental_dynamics(df, out_path))

    print("[2/6] Generating Figure 2: Bayesian Confidence Dynamics...")
    generated.append(plot_bayesian_confidence(df, out_path))

    print("[3/6] Generating Figure 3: Layer Latency Breakdown...")
    generated.append(plot_latency_breakdown(df, out_path))

    print("[4/6] Generating Figure 4: Diagnostic & Action Distributions...")
    generated.append(plot_distributions(df, out_path))

    print("[5/6] Generating Figure 5: Prior vs Posterior Scatter...")
    generated.append(plot_prior_posterior_scatter(df, out_path))

    print("[6/6] Generating Figure 6: Comprehensive Academic Dashboard...")
    generated.append(plot_comprehensive_dashboard(df, out_path))

    print(f"\n[+] All 6 publication figures generated successfully in: {os.path.abspath(out_dir)}")
    for f in generated:
        print(f"  - {f}")
    return generated


if __name__ == "__main__":
    generate_all_plots()
