import os
import json
import pandas as pd
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────
RESULTS_DIR = "results_ablation"
GT_CSV      = "Combined_Ground_Truth_Master_Cleaned.csv"
OUTPUT_CSV  = "Ablation_Results.csv"

# ── Load ground truth ─────────────────────────────────────────────────────────
gt_df = pd.read_csv(GT_CSV)

def parse_gt_field(value):
    if pd.isna(value) or str(value).strip() == "":
        return set()
    return set(v.strip().lower() for v in str(value).split(";") if v.strip())

gt_lookup = {}
for _, row in gt_df.iterrows():
    rid = str(row["Report_ID"]).strip()
    gt_lookup[rid] = {
        "ips":     parse_gt_field(row.get("Verified_IPs")),
        "domains": parse_gt_field(row.get("Verified_Domains")),
        "hashes":  parse_gt_field(row.get("Verified_Hashes")),
    }

# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize(items):
    return set(str(i).strip().lower() for i in items if i)

def score(extracted, ground_truth):
    # Zero-IOC hallucination guard — if GT is empty, anything extracted is wrong
    if not ground_truth and not extracted:
        return 1.0, 1.0, 1.0, 0, 0
    if not ground_truth:
        fp = len(extracted)
        return 0.0, 0.0, 0.0, fp, 0
    if not extracted:
        fn = len(ground_truth)
        return 1.0, 0.0, 0.0, 0, fn
    tp = len(extracted & ground_truth)
    fp = len(extracted - ground_truth)
    fn = len(ground_truth - extracted)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1, fp, fn

# ── Grade all result files ────────────────────────────────────────────────────
records = []

for fname in sorted(os.listdir(RESULTS_DIR)):
    if not fname.endswith(".json"):
        continue

    fpath = os.path.join(RESULTS_DIR, fname)
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)

    report_id      = data.get("report_id", "")
    model          = data.get("model", "")
    prompt_variant = data.get("prompt_variant", "")
    stratum        = data.get("stratum", "")

    if report_id not in gt_lookup:
        print(f"[!] No ground truth for {report_id}, skipping.")
        continue

    gt = gt_lookup[report_id]

    for p in data.get("passes", []):
        ext = p.get("extraction") or {}
        ioc = ext.get("Indicators_of_Compromise") or {}

        extracted_ips     = normalize(ioc.get("IPv4_Addresses") or [])
        extracted_domains = normalize(ioc.get("Domains") or [])
        extracted_hashes  = normalize(ioc.get("Hashes") or [])

        prec_ip,  rec_ip,  f1_ip,  fp_ip,  fn_ip  = score(extracted_ips,     gt["ips"])
        prec_dom, rec_dom, f1_dom, fp_dom, fn_dom  = score(extracted_domains, gt["domains"])
        prec_has, rec_has, f1_has, fp_has, fn_has  = score(extracted_hashes,  gt["hashes"])

        pass_f1        = (f1_ip + f1_dom + f1_has) / 3
        total_fp       = fp_ip + fp_dom + fp_has
        total_fn       = fn_ip + fn_dom + fn_has

        records.append({
            "Report_ID":      report_id,
            "Stratum":        stratum,
            "Model":          model,
            "Prompt_Variant": prompt_variant,
            "Pass":           p.get("pass_number"),
            "F1_IP":          round(f1_ip,  4),
            "F1_Domain":      round(f1_dom, 4),
            "F1_Hash":        round(f1_has, 4),
            "F1_Macro":       round(pass_f1, 4),
            "Precision_Macro":round((prec_ip + prec_dom + prec_has) / 3, 4),
            "Recall_Macro":   round((rec_ip  + rec_dom  + rec_has)  / 3, 4),
            "FP_Count":       total_fp,
            "FN_Count":       total_fn,
            "Latency_s":      p.get("latency_seconds"),
        })

# ── Save ──────────────────────────────────────────────────────────────────────
if records:
    out_df = pd.DataFrame(records)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[+] Ablation grading complete. {len(records)} pass records saved to {OUTPUT_CSV}")
    print("\n── Mean F1 by Prompt Variant and Model ──")
    print(out_df.groupby(["Prompt_Variant", "Model"])["F1_Macro"].mean().round(4).to_string())
    print("\n── Mean FP by Prompt Variant and Model ──")
    print(out_df.groupby(["Prompt_Variant", "Model"])["FP_Count"].mean().round(2).to_string())
else:
    print("[!] No records graded. Check ground truth keys match report IDs in results_ablation/.")