import os
import json
import pandas as pd


RESULTS_DIR = "results_iocparser/"
GROUND_TRUTH_CSV = "Combined_Ground_Truth_Master_Cleaned.csv"
OUTPUT_CSV = "IOCParser_Performance.csv"


def safe_split(value, delimiter=';'):
    if value is None:
        return []
    str_val = str(value).strip()
    if str_val.lower() in ('nan', 'n/a', 'none', ''):
        return []
    return [item.strip().lower() for item in str_val.split(delimiter) if item.strip()]


def count_tp(extracted, gt):
    if not extracted or not gt:
        return 0
    extracted_clean = [str(e).strip().lower() for e in extracted]
    return sum(1 for g in gt if str(g).lower() in extracted_clean)


def count_fp(extracted, gt):
    if not extracted:
        return 0
    gt_clean = [str(g).lower() for g in gt]
    return sum(1 for e in extracted if str(e).strip().lower() not in gt_clean)


def count_fn(extracted, gt):
    if not gt:
        return 0
    if not extracted:
        return len(gt)
    extracted_clean = [str(e).strip().lower() for e in extracted]
    return sum(1 for g in gt if str(g).lower() not in extracted_clean)


def score(extracted, gt):
    # Zero-IOC correction: if both extracted and ground truth are empty,
    # this is a correct null extraction, not a failure.
    if not extracted and not gt:
        return 0, 0, 0, 1.0, 1.0, 1.0

    tp = count_tp(extracted, gt)
    fp = count_fp(extracted, gt)
    fn = count_fn(extracted, gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return tp, fp, fn, round(precision, 4), round(recall, 4), round(f1, 4)


def grade_iocparser():
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    gt_dict = gt_df.set_index('Report_ID').to_dict('index')

    records = []

    for filename in sorted(os.listdir(RESULTS_DIR)):
        if not filename.endswith(".json"):
            continue

        report_id = os.path.splitext(filename)[0]
        if report_id not in gt_dict:
            print(f"Skipping {report_id}: not in ground truth")
            continue

        with open(os.path.join(RESULTS_DIR, filename), "r", encoding="utf-8") as f:
            payload = json.load(f)

        if payload.get("status") != "success":
            print(f"{report_id}: non-success status, treating as empty extraction")
            data = {}
        else:
            data = payload.get("data", {})

        extracted_ips = data.get("IPv4", []) or []
        extracted_domains = data.get("DOMAIN", []) or []
        extracted_hashes = (
            (data.get("FILE_HASH_MD5", []) or [])
            + (data.get("FILE_HASH_SHA1", []) or [])
            + (data.get("FILE_HASH_SHA256", []) or [])
        )

        gt_row = gt_dict[report_id]
        gt_ips = safe_split(gt_row.get("Verified_IPs"))
        gt_domains = safe_split(gt_row.get("Verified_Domains"))
        gt_hashes = safe_split(gt_row.get("Verified_Hashes"))
        stratum = gt_row.get("Stratum", "Unknown")

        tp_ip, fp_ip, fn_ip, prec_ip, rec_ip, f1_ip = score(extracted_ips, gt_ips)
        tp_dom, fp_dom, fn_dom, prec_dom, rec_dom, f1_dom = score(extracted_domains, gt_domains)
        tp_hash, fp_hash, fn_hash, prec_hash, rec_hash, f1_hash = score(extracted_hashes, gt_hashes)

        # Zero-IOC correction at the overall (report-level) score too
        all_gt_empty = (not gt_ips and not gt_domains and not gt_hashes)
        all_extracted_empty = (not extracted_ips and not extracted_domains and not extracted_hashes)

        if all_gt_empty and all_extracted_empty:
            tp = fp = fn = 0
            precision = recall = f1 = 1.0
        else:
            tp = tp_ip + tp_dom + tp_hash
            fp = fp_ip + fp_dom + fp_hash
            fn = fn_ip + fn_dom + fn_hash
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        records.append({
            "Report_ID": report_id,
            "Stratum": stratum,
            "Model": "IOCParser",
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1_Score": round(f1, 4),
            "Precision_IP": prec_ip, "Recall_IP": rec_ip, "F1_IP": f1_ip,
            "Precision_Domain": prec_dom, "Recall_Domain": rec_dom, "F1_Domain": f1_dom,
            "Precision_Hash": prec_hash, "Recall_Hash": rec_hash, "F1_Hash": f1_hash,
        })

        print(f"{report_id}: F1={round(f1,3)} Precision={round(precision,3)} Recall={round(recall,3)}")

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {OUTPUT_CSV}")
    print(df.groupby("Stratum")["F1_Score"].mean().round(3))


if __name__ == "__main__":
    grade_iocparser()