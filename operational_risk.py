import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import product


# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("Model_Performance_Full.csv")

model_labels = {
    "deepseek/deepseek-r1": "DeepSeek",
    "google/gemini-2.5-pro": "Gemini",
    "openai/gpt-4o-mini": "GPT-4o-mini",
    "openai/gpt-4o": "GPT-4o"
}
df["Model_Short"] = df["Model"].map(model_labels)

colors = {
    "DeepSeek": "#2196F3",
    "Gemini": "#4CAF50",
    "GPT-4o-mini": "#FF9800",
    "GPT-4o": "#9C27B0"
}

short_models = ["DeepSeek", "Gemini", "GPT-4o-mini", "GPT-4o"]


# ── GAP 1: Baseline ORS ───────────────────────────────────────────────────────
max_fn = df["Avg_Misses_FN"].max()
max_fp = df["Avg_Escapes_FP"].max()

df["FN_Risk"] = df["Avg_Misses_FN"] / max_fn
df["FP_Risk"] = (df["Avg_Escapes_FP"] / max_fp) * (1 + df["Operational_Noise_Pct"] / 100)
df["Reliability_Penalty"] = df["API_Failure_Pct"] / 100

df["ORS_Baseline"] = (
    0.60 * df["FN_Risk"] +
    0.25 * df["FP_Risk"] +
    0.15 * df["Reliability_Penalty"]
).clip(upper=1.0)

print("=== Baseline ORS per Model ===")
print(df.groupby("Model_Short")["ORS_Baseline"].mean().round(4))


# ── GAP 1: Sensitivity Analysis ───────────────────────────────────────────────
weight_configs = {
    "Baseline":          (0.60, 0.25, 0.15),
    "FP_Heavy":          (0.40, 0.45, 0.15),
    "Balanced":          (0.45, 0.40, 0.15),
    "Reliability_Heavy": (0.50, 0.25, 0.25),
}

sensitivity_results = {}
for config_name, (w_fn, w_fp, w_rel) in weight_configs.items():
    col = f"ORS_{config_name}"
    df[col] = (
        w_fn * df["FN_Risk"] +
        w_fp * df["FP_Risk"] +
        w_rel * df["Reliability_Penalty"]
    ).clip(upper=1.0)
    sensitivity_results[config_name] = df.groupby("Model_Short")[col].mean().round(4)

sensitivity_df = pd.DataFrame(sensitivity_results)
print("\n=== Sensitivity Analysis — Model Average ORS Across Weight Configs ===")
print(sensitivity_df)

# Figure 5.1
fig, ax = plt.subplots(figsize=(10, 5))
im = ax.imshow(sensitivity_df.values, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(sensitivity_df.columns)))
ax.set_xticklabels(sensitivity_df.columns, rotation=15)
ax.set_yticks(range(len(sensitivity_df.index)))
ax.set_yticklabels(sensitivity_df.index)
for i in range(len(sensitivity_df.index)):
    for j in range(len(sensitivity_df.columns)):
        ax.text(j, i, f"{sensitivity_df.values[i, j]:.3f}",
                ha="center", va="center", fontsize=10, color="black")
plt.colorbar(im, ax=ax, label="Mean ORS")
ax.set_title("ORS Sensitivity Analysis Across Weight Configurations")
plt.tight_layout()
plt.savefig("fig5_1_sensitivity_heatmap.png", dpi=300)
plt.close()
print("\nSaved fig5_1_sensitivity_heatmap.png")


# ── GAP 2: IOC Type Risk Weighting ───────────────────────────────────────────
ioc_type_cols = ["Avg_FN_IP", "Avg_FN_Domain", "Avg_FN_Hash"]
if all(col in df.columns for col in ioc_type_cols):
    max_fn_ip     = df["Avg_FN_IP"].max()
    max_fn_domain = df["Avg_FN_Domain"].max()
    max_fn_hash   = df["Avg_FN_Hash"].max()

    df["FN_Risk_Weighted"] = (
        0.35 * (df["Avg_FN_IP"] / max_fn_ip if max_fn_ip > 0 else 0) +
        0.40 * (df["Avg_FN_Domain"] / max_fn_domain if max_fn_domain > 0 else 0) +
        0.25 * (df["Avg_FN_Hash"] / max_fn_hash if max_fn_hash > 0 else 0)
    )

    df["ORS_TypeWeighted"] = (
        0.60 * df["FN_Risk_Weighted"] +
        0.25 * df["FP_Risk"] +
        0.15 * df["Reliability_Penalty"]
    ).clip(upper=1.0)

    print("\n=== IOC Type-Weighted ORS per Model ===")
    print(df.groupby("Model_Short")["ORS_TypeWeighted"].mean().round(4))

    print("\n=== Baseline vs Type-Weighted ORS Comparison ===")
    compare = df.groupby("Model_Short")[["ORS_Baseline", "ORS_TypeWeighted"]].mean().round(4)
    print(compare)
else:
    print("\n[WARNING] Per-type FN columns not found in CSV.")
    print("IOC Type Weighting block skipped.")
    print("Required columns: Avg_FN_IP, Avg_FN_Domain, Avg_FN_Hash")


# ── GAP 3: Failure Combination Scenarios ─────────────────────────────────────
scenario_definitions = {
    "Full Miss (All IOC Types)":           {"risk_level": "Critical",     "score": 1.00,
        "rationale": "No indicators surfaced. Complete detection blind spot."},
    "Domain Miss / IP Extracted":          {"risk_level": "High",         "score": 0.75,
        "rationale": "C2 domain unblocked. IP rotation defeats IP only block. "
                     "Grounded in Bianco (2013) pyramid, domains sit above IPs."},
    "IP Miss / Domain Extracted":          {"risk_level": "Medium",       "score": 0.50,
        "rationale": "Domain sinkholed. Direct IP traffic unblocked but "
                     "C2 channel partially disrupted."},
    "Hash Miss / Network IOCs Extracted":  {"risk_level": "Medium-High",  "score": 0.65,
        "rationale": "Network indicators blocked but malware execution undetected. "
                     "EDR/AV signature absent. Payload can still run."},
    "Full Extraction":                     {"risk_level": "Acceptable",   "score": 0.00,
        "rationale": "All IOC types correctly extracted. No residual risk."},
}

print("\n=== Failure Combination Scenarios ===")
for scenario, details in scenario_definitions.items():
    print(f"\nScenario: {scenario}")
    print(f"  Risk Level : {details['risk_level']}")
    print(f"  Score      : {details['score']}")
    print(f"  Rationale  : {details['rationale']}")

# Figure 5.2
scenario_names  = list(scenario_definitions.keys())
scenario_scores = [v["score"] for v in scenario_definitions.values()]
scenario_colors = ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c", "#1565c0"]

fig, ax = plt.subplots(figsize=(11, 6))
bars = ax.barh(scenario_names, scenario_scores,
               color=scenario_colors[::-1], alpha=0.85)
ax.set_xlabel("Residual Risk Score")
ax.set_title("Failure Combination Scenario Risk Levels")
ax.set_xlim(0, 1.1)
for bar, score in zip(bars, scenario_scores):
    ax.text(score + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}", va="center", fontsize=10)
ax.grid(axis="x", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig5_2_failure_scenarios.png", dpi=300)
plt.close()
print("\nSaved fig5_2_failure_scenarios.png")


# ── GAP 4: Risk Bands ─────────────────────────────────────────────────────────
def assign_risk_band(ors):
    if ors <= 0.25:
        return "Acceptable"
    elif ors <= 0.55:
        return "Elevated"
    else:
        return "Critical"

df["Risk_Band"] = df["ORS_Baseline"].apply(assign_risk_band)

print("\n=== Risk Band Distribution ===")
print(df["Risk_Band"].value_counts())

print("\n=== Risk Band per Model ===")
print(df.groupby(["Model_Short", "Risk_Band"]).size().unstack(fill_value=0))

# Figure 5.3
fig, ax = plt.subplots(figsize=(11, 6))
for model in short_models:
    subset = df[df["Model_Short"] == model]["ORS_Baseline"].sort_values().values
    ax.plot(subset, label=model, color=colors[model], linewidth=2)

ax.axhspan(0.00, 0.25, alpha=0.08, color="green",  label="Acceptable (0.00 to 0.25)")
ax.axhspan(0.25, 0.55, alpha=0.08, color="orange", label="Elevated (0.26 to 0.55)")
ax.axhspan(0.55, 1.00, alpha=0.08, color="red",    label="Critical (0.56 to 1.00)")
ax.axhline(0.25, color="green",  linestyle="--", linewidth=0.8)
ax.axhline(0.55, color="orange", linestyle="--", linewidth=0.8)

ax.set_xlabel("Report (sorted by ORS)")
ax.set_ylabel("Operational Risk Score (ORS)")
ax.set_title("ORS Distribution with NIST SP 800-30 Risk Bands")
ax.set_ylim(0, 1.05)
ax.legend(loc="upper left", fontsize=8)
ax.grid(linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("fig5_3_ors_risk_bands.png", dpi=300)
plt.close()
print("Saved fig5_3_ors_risk_bands.png")


# ── GAP 4: Downstream Artifact Risk Mapping ───────────────────────────────────
downstream_artifacts = {
    "Firewall Rules": {
        "FP Impact": "Spurious block rules, legitimate traffic disrupted",
        "FN Impact": "Malicious IP/domain unblocked, threat actor retains access"
    },
    "IDS/IPS Signatures": {
        "FP Impact": "Alert fatigue, analyst queue inflated with false triggers",
        "FN Impact": "Detection blind spot, malicious traffic passes uninspected"
    },
    "SIEM Correlation Rules": {
        "FP Impact": "Incident queue inflation, investigation resource waste",
        "FN Impact": "Genuine incident suppressed, no alert raised"
    },
    "SOAR Playbook Automation": {
        "FP Impact": "Automated response on non-threat, service disruption risk",
        "FN Impact": "Playbook never triggered, threat propagates at machine speed"
    },
    "Threat Intel Publications": {
        "FP Impact": "Credibility damage, partner orgs misled by fabricated IOCs",
        "FN Impact": "Incomplete report, attack vector undocumented"
    },
    "Root Cause Analysis": {
        "FP Impact": "Incorrect attribution, remediation targets wrong vector",
        "FN Impact": "Actual attack path not documented, vulnerability stays open"
    },
}

print("\n=== Downstream Artifact Risk Mapping ===")
for artifact, impacts in downstream_artifacts.items():
    print(f"\n{artifact}")
    print(f"  FP: {impacts['FP Impact']}")
    print(f"  FN: {impacts['FN Impact']}")


# ── GAP 4: Operational Deployment Risk Summary ────────────────────────────────
print("\n=== Operational Deployment Risk Summary ===")
deployment_summary = df.groupby("Model_Short").agg(
    Mean_ORS=("ORS_Baseline", "mean"),
    Max_ORS=("ORS_Baseline", "max"),
    Mean_Reliability_Penalty=("Reliability_Penalty", "mean"),
    Mean_Entropy=("Semantic_Entropy_Hsem", "mean"),
    Critical_Reports=("Risk_Band", lambda x: (x == "Critical").sum()),
    Acceptable_Reports=("Risk_Band", lambda x: (x == "Acceptable").sum()),
).round(3)
print(deployment_summary)


# ── Export ────────────────────────────────────────────────────────────────────
ors_cols = ["Report_ID", "Model_Short", "Stratum",
            "FN_Risk", "FP_Risk", "Reliability_Penalty",
            "ORS_Baseline", "Risk_Band"] + \
           [f"ORS_{k}" for k in weight_configs.keys()]

export_cols = [c for c in ors_cols if c in df.columns]
df[export_cols].to_csv("Section5_ORS_Results.csv", index=False)
print("\nSaved Section5_ORS_Results.csv")

print("\nAll Section 5 computations complete.")