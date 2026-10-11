import os
import json
import math
import numpy as np
import pandas as pd
from collections import Counter

"""
=========================================================================================
THESIS METRICS: EMPIRICAL EVALUATION ENGINE (v4 - Section 5 Extended)
=========================================================================================
Changes from v3:
- Per-type TP, FP, FN counts added (IP, Domain, Hash)
- Per-type Precision and Recall computed and exported
- Ground truth size per IOC type exported (GT_IP_Count, GT_Domain_Count, GT_Hash_Count)
- Per-type average miss and escape counts exported
- Per-type operational noise detection added
- All new columns flow directly into Section 5 ORS script
- Backward compatible: all v3 columns retained unchanged
=========================================================================================
"""

# --- Configuration ---
RESULTS_DIR     = "results/"
GROUND_TRUTH_CSV = "Combined_Ground_Truth_Master_Cleaned.csv"
OUTPUT_FULL_CSV  = "Model_Performance_Full.csv"
OUTPUT_CSV       = "Model_Performance_Summary.csv"
EXPECTED_PASSES  = 1


def safe_split(value, delimiter=';'):
    """
    Safely splits a CSV cell into a clean lowercase list.
    Handles NaN, None, 'N/A', 'nan', and empty strings gracefully.
    """
    if value is None:
        return []
    str_val = str(value).strip()
    if str_val.lower() in ('nan', 'n/a', 'none', ''):
        return []
    return [item.strip().lower() for item in str_val.split(delimiter) if item.strip()]


def calculate_semantic_entropy(successful_passes):
    """Calculates Discrete Semantic Entropy (H_sem) on successful generations only."""
    if not successful_passes:
        return float('nan')
    extracted_strings = [
        json.dumps(p.get("extraction", {}), sort_keys=True)
        for p in successful_passes
    ]
    counts = Counter(extracted_strings)
    total  = len(successful_passes)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def count_true_positives(extracted_list, gt_list):
    """True Positives: extracted items that match ground truth."""
    if not extracted_list or not gt_list:
        return 0
    extracted_clean = [str(e).strip().lower() for e in extracted_list]
    return sum(1 for gt in gt_list if str(gt).lower() in extracted_clean)


def count_escapes(extracted_list, gt_list):
    """False Positives: extracted items not in ground truth."""
    if not extracted_list:
        return 0
    gt_clean = [str(gt).lower() for gt in gt_list]
    return sum(
        1 for item in extracted_list
        if str(item).strip().lower() not in gt_clean
    )


def count_misses(extracted_list, gt_list):
    """False Negatives: ground truth items not extracted."""
    if not gt_list:
        return 0
    if not extracted_list:
        return len(gt_list)
    extracted_clean = [str(e).strip().lower() for e in extracted_list]
    return sum(1 for gt in gt_list if str(gt).lower() not in extracted_clean)


def check_operational_noise(extraction, trap_keywords):
    """Detects benign trap leakage into model output."""
    if not trap_keywords:
        return False
    extraction_str = json.dumps(extraction).lower()
    return any(len(trap) > 2 and trap in extraction_str for trap in trap_keywords)


def calculate_pass_f1(pass_data, gt_ips, gt_domains, gt_hashes):
    """
    Calculates Precision, Recall, F1, and per-type TP/FP/FN for a single pass.

    Zero-IOC handling:
    - If all ground truth lists are empty AND model returns all empty, F1 = 1.0
    - If ground truth is empty but model returns IOCs, all are false positives
    """
    extraction     = pass_data.get("extraction") or {}
    iocs           = extraction.get("Indicators_of_Compromise") or {}
    extracted_ips     = iocs.get("IPv4_Addresses") or []
    extracted_domains = iocs.get("Domains")        or []
    extracted_hashes  = iocs.get("Hashes")         or []

    all_gt_empty       = (not gt_ips and not gt_domains and not gt_hashes)
    all_extracted_empty = (not extracted_ips and not extracted_domains
                           and not extracted_hashes)

    # Correct null extraction on a zero-IOC report
    if all_gt_empty and all_extracted_empty:
        return {
            "tp": 0, "fp": 0, "fn": 0,
            "precision": 1.0, "recall": 1.0, "f1": 1.0,
            # Per-type counts — all zero on null extraction
            "tp_ip": 0, "fp_ip": 0, "fn_ip": 0,
            "tp_domain": 0, "fp_domain": 0, "fn_domain": 0,
            "tp_hash": 0, "fp_hash": 0, "fn_hash": 0,
        }

    # --- Per-type counts ---
    tp_ip     = count_true_positives(extracted_ips,     gt_ips)
    fp_ip     = count_escapes(extracted_ips,            gt_ips)
    fn_ip     = count_misses(extracted_ips,             gt_ips)

    tp_domain = count_true_positives(extracted_domains, gt_domains)
    fp_domain = count_escapes(extracted_domains,        gt_domains)
    fn_domain = count_misses(extracted_domains,         gt_domains)

    tp_hash   = count_true_positives(extracted_hashes,  gt_hashes)
    fp_hash   = count_escapes(extracted_hashes,         gt_hashes)
    fn_hash   = count_misses(extracted_hashes,          gt_hashes)

    # --- Aggregate totals ---
    tp = tp_ip + tp_domain + tp_hash
    fp = fp_ip + fp_domain + fp_hash
    fn = fn_ip + fn_domain + fn_hash

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall) / (precision + recall) \
                if (precision + recall) > 0 else 0.0

    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        # Per-type breakdowns
        "tp_ip":     tp_ip,     "fp_ip":     fp_ip,     "fn_ip":     fn_ip,
        "tp_domain": tp_domain, "fp_domain": fp_domain, "fn_domain": fn_domain,
        "tp_hash":   tp_hash,   "fp_hash":   fp_hash,   "fn_hash":   fn_hash,
    }


def compute_per_type_metrics(per_pass_scores, success_count, gt_ips,
                              gt_domains, gt_hashes):
    """
    Computes averaged per-type TP, FP, FN, Precision, and Recall
    across all successful passes for a single report-model pair.

    Per-type Precision and Recall are computed from summed TP/FP/FN
    across passes rather than averaged ratios — this is the micro-average
    approach and is more stable when individual passes have zero denominators.
    """
    if not per_pass_scores or success_count == 0:
        return {
            "GT_IP_Count":    len(gt_ips),
            "GT_Domain_Count": len(gt_domains),
            "GT_Hash_Count":  len(gt_hashes),
            "Avg_FN_IP":    0.0, "Avg_FP_IP":    0.0, "Avg_TP_IP":    0.0,
            "Avg_FN_Domain": 0.0, "Avg_FP_Domain": 0.0, "Avg_TP_Domain": 0.0,
            "Avg_FN_Hash":  0.0, "Avg_FP_Hash":  0.0, "Avg_TP_Hash":  0.0,
            "Precision_IP":    0.0, "Recall_IP":    0.0,
            "Precision_Domain": 0.0, "Recall_Domain": 0.0,
            "Precision_Hash":  0.0, "Recall_Hash":  0.0,
        }

    # Sum across passes for micro-averaged precision/recall
    sum_tp_ip     = sum(s["tp_ip"]     for s in per_pass_scores)
    sum_fp_ip     = sum(s["fp_ip"]     for s in per_pass_scores)
    sum_fn_ip     = sum(s["fn_ip"]     for s in per_pass_scores)

    sum_tp_domain = sum(s["tp_domain"] for s in per_pass_scores)
    sum_fp_domain = sum(s["fp_domain"] for s in per_pass_scores)
    sum_fn_domain = sum(s["fn_domain"] for s in per_pass_scores)

    sum_tp_hash   = sum(s["tp_hash"]   for s in per_pass_scores)
    sum_fp_hash   = sum(s["fp_hash"]   for s in per_pass_scores)
    sum_fn_hash   = sum(s["fn_hash"]   for s in per_pass_scores)

    def safe_precision(tp, fp):
        return round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0

    def safe_recall(tp, fn):
        return round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0

    return {
        # Ground truth sizes — context for interpreting miss rates
        "GT_IP_Count":     len(gt_ips),
        "GT_Domain_Count": len(gt_domains),
        "GT_Hash_Count":   len(gt_hashes),

        # Average per-pass counts
        "Avg_TP_IP":     round(sum_tp_ip     / success_count, 2),
        "Avg_FP_IP":     round(sum_fp_ip     / success_count, 2),
        "Avg_FN_IP":     round(sum_fn_ip     / success_count, 2),

        "Avg_TP_Domain": round(sum_tp_domain / success_count, 2),
        "Avg_FP_Domain": round(sum_fp_domain / success_count, 2),
        "Avg_FN_Domain": round(sum_fn_domain / success_count, 2),

        "Avg_TP_Hash":   round(sum_tp_hash   / success_count, 2),
        "Avg_FP_Hash":   round(sum_fp_hash   / success_count, 2),
        "Avg_FN_Hash":   round(sum_fn_hash   / success_count, 2),

        # Micro-averaged per-type precision and recall
        "Precision_IP":     safe_precision(sum_tp_ip,     sum_fp_ip),
        "Recall_IP":        safe_recall(sum_tp_ip,        sum_fn_ip),
        "Precision_Domain": safe_precision(sum_tp_domain, sum_fp_domain),
        "Recall_Domain":    safe_recall(sum_tp_domain,    sum_fn_domain),
        "Precision_Hash":   safe_precision(sum_tp_hash,   sum_fp_hash),
        "Recall_Hash":      safe_recall(sum_tp_hash,      sum_fn_hash),
    }


def grade_results():
    """Master aggregation function: parses N-pass JSONs into the empirical matrix."""
    try:
        gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    except FileNotFoundError:
        print(f"[!] CRITICAL: '{GROUND_TRUTH_CSV}' not found. Exiting.")
        return

    gt_dict = gt_df.set_index('Report_ID').to_dict('index')
    analysis_records = []

    print(f"[*] Evaluation Engine v4 starting on '{RESULTS_DIR}'...")

    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(RESULTS_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            print(f" [!] Skipping {filename}: Malformed JSON ({e})")
            continue

        report_id = data.get("report_id")
        model     = data.get("model")
        passes    = data.get("passes", [])

        if report_id not in gt_dict:
            print(f" [!] Skipping {filename}: "
                  f"report_id '{report_id}' not in ground truth")
            continue

        gt_row  = gt_dict[report_id]
        stratum = data.get("stratum") or gt_row.get("Stratum", "Unknown")

        trap_keywords = safe_split(gt_row.get("Benign_Traps"))
        gt_ips        = safe_split(gt_row.get("Verified_IPs"))
        gt_domains    = safe_split(gt_row.get("Verified_Domains"))
        gt_hashes     = safe_split(gt_row.get("Verified_Hashes"))

        successful_passes = [
            p for p in passes
            if p.get("extraction") is not None and p.get("error") is None
        ]
        failed_passes = sum(
            1 for p in passes
            if p.get("error") is not None or p.get("extraction") is None
        )

        h_sem       = calculate_semantic_entropy(successful_passes)
        failure_pct = min(100.0, max(0.0, (failed_passes / EXPECTED_PASSES) * 100))

        per_pass_scores = []
        noise_triggers  = 0
        total_escapes   = 0
        total_misses    = 0

        for p in successful_passes:
            scores = calculate_pass_f1(p, gt_ips, gt_domains, gt_hashes)
            per_pass_scores.append(scores)
            total_escapes += scores["fp"]
            total_misses  += scores["fn"]
            if check_operational_noise(p.get("extraction", {}), trap_keywords):
                noise_triggers += 1

        success_count = len(successful_passes)

        if per_pass_scores:
            f1_values         = [s["f1"]        for s in per_pass_scores]
            precision_values  = [s["precision"] for s in per_pass_scores]
            recall_values     = [s["recall"]    for s in per_pass_scores]

            mean_f1       = round(float(np.mean(f1_values)), 3)
            std_f1        = round(
                float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else 0.0, 3)
            mean_precision = round(float(np.mean(precision_values)), 3)
            mean_recall    = round(float(np.mean(recall_values)),    3)
        else:
            mean_f1 = std_f1 = mean_precision = mean_recall = 0.0
            f1_values = []

        avg_escapes_per_pass = total_escapes / success_count if success_count else 0.0
        avg_misses_per_pass  = total_misses  / success_count if success_count else 0.0
        noise_frequency_pct  = (noise_triggers / success_count) * 100 \
                               if success_count else 0.0

        # Compute per-type metrics
        type_metrics = compute_per_type_metrics(
            per_pass_scores, success_count, gt_ips, gt_domains, gt_hashes
        )

        record = {
            # --- Core identifiers ---
            "Report_ID": report_id,
            "Stratum":   stratum,
            "Model":     model,

            # --- Entropy and stability ---
            "Semantic_Entropy_Hsem": round(h_sem, 3) if not math.isnan(h_sem) else None,

            # --- Aggregate F1 metrics (v3 retained) ---
            "F1_Score":    mean_f1,
            "F1_Std_Dev":  std_f1,
            "F1_Per_Pass": f1_values,
            "Precision":   mean_precision,
            "Recall":      mean_recall,

            # --- Aggregate FP/FN (v3 retained) ---
            "Avg_Escapes_FP":      round(avg_escapes_per_pass, 2),
            "Avg_Misses_FN":       round(avg_misses_per_pass,  2),
            "Operational_Noise_Pct": round(noise_frequency_pct, 2),
            "API_Failure_Pct":     round(failure_pct, 2),
        }

        # Merge per-type metrics into record
        record.update(type_metrics)

        analysis_records.append(record)

        print(f" [+] {report_id} | {model} | "
              f"F1={mean_f1} ± {std_f1} | "
              f"H_sem={round(h_sem, 3) if not math.isnan(h_sem) else 'N/A'} | "
              f"FN_IP={type_metrics['Avg_FN_IP']} "
              f"FN_Dom={type_metrics['Avg_FN_Domain']} "
              f"FN_Hash={type_metrics['Avg_FN_Hash']}")

    if not analysis_records:
        print("\n[!] No valid records found. "
              "Check results directory and ground truth CSV.")
        return

    results_df = pd.DataFrame(analysis_records)
    results_df['Stratum'] = results_df['Stratum'].fillna('Unknown')

    # Full per-report output
    results_df.to_csv(OUTPUT_FULL_CSV, index=False)

    # Summary aggregated by Model + Stratum
    # F1_Per_Pass dropped from summary — kept in full CSV only
    agg_cols = {
        'Semantic_Entropy_Hsem':  'median',
        'F1_Score':               'mean',
        'F1_Std_Dev':             'mean',
        'Precision':              'mean',
        'Recall':                 'mean',
        'Avg_Escapes_FP':         'mean',
        'Avg_Misses_FN':          'mean',
        'Operational_Noise_Pct':  'mean',
        'API_Failure_Pct':        'mean',
        # Per-type averages in summary
        'GT_IP_Count':            'mean',
        'GT_Domain_Count':        'mean',
        'GT_Hash_Count':          'mean',
        'Avg_FN_IP':              'mean',
        'Avg_FN_Domain':          'mean',
        'Avg_FN_Hash':            'mean',
        'Avg_FP_IP':              'mean',
        'Avg_FP_Domain':          'mean',
        'Avg_FP_Hash':            'mean',
        'Recall_IP':              'mean',
        'Recall_Domain':          'mean',
        'Recall_Hash':            'mean',
        'Precision_IP':           'mean',
        'Precision_Domain':       'mean',
        'Precision_Hash':         'mean',
    }

    summary_df = results_df.drop(columns=['F1_Per_Pass']).groupby(
        ['Model', 'Stratum']
    ).agg(agg_cols).reset_index()

    summary_df.to_csv(OUTPUT_CSV, index=False)

    # Global macro F1 per model
    global_f1 = results_df.groupby('Model')['F1_Score'].mean().round(3)

    print(f"\n[+] Grading Complete.")
    print(f"    Full per-report results : {OUTPUT_FULL_CSV}")
    print(f"    Summary (Model/Stratum) : {OUTPUT_CSV}")
    print("\n--- MODEL GLOBAL MACRO F1 ---")
    print(global_f1.to_string())
    print("\n--- STRATIFIED PERFORMANCE SUMMARY ---")
    print(summary_df.to_string(index=False))

    # Per-type recall summary printed separately for quick inspection
    print("\n--- PER-TYPE RECALL SUMMARY (Global Mean) ---")
    type_recall = results_df.groupby('Model')[
        ['Recall_IP', 'Recall_Domain', 'Recall_Hash']
    ].mean().round(3)
    print(type_recall.to_string())


if __name__ == "__main__":
    grade_results()