import os
import json
import time
import re
import logging
from openai import OpenAI

"""
=========================================================================================
THESIS PIPELINE: AUTONOMOUS CTI EXTRACTION (- Final Production)
=========================================================================================
Aligned with thesis methodology as written:
- 4 models x 30 reports x 10 passes = 1,200 discrete extraction attempts
- Temperature 0.0 for deterministic output across all models
- Zero-IOC guard in system prompt (handles S1_06 and similar control rows)
- response_format={"type": "json_object"} conditional on model support
- STRATUM_MAP included; stratum written to every output JSON
- Incremental save per report/model combination
- Timestamped audit log written to execution_audit_YYYYMMDD_HHMMSS.log
- Inter-model sleep added to avoid OpenRouter rate limits
- Safer regex fallback for non-JSON-mode models (Gemini, DeepSeek)
- API failure captured per pass; error written to output JSON
- UTF-8 encoding on log file handler (Windows cp1252 compatibility fix)
=========================================================================================
"""

# --- Timestamped audit log ---
run_timestamp = time.strftime('%Y%m%d_%H%M%S')
log_filename = f"execution_audit_{run_timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# --- OpenRouter client ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="[Placeholder for api_key]",
    max_retries=0,
    timeout=180.0
)

# --- Models (4 models as per thesis methodology) ---
MODELS_TO_TEST = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-r1"
]

# --- Models that support response_format=json_object ---
JSON_MODE_SUPPORTED = {
    "openai/gpt-4o",
    "openai/gpt-4o-mini"
}

# --- Passes (10 as per thesis methodology) ---
TOTAL_PASSES = 10

# --- Stratum mapping (30 reports across 3 strata) ---
STRATUM_MAP = {
    "S1_01": "S1", "S1_02": "S1", "S1_03": "S1", "S1_04": "S1", "S1_05": "S1",
    "S1_06": "S1", "S1_07": "S1", "S1_08": "S1", "S1_09": "S1", "S1_10": "S1",
    "S2_01": "S2", "S2_02": "S2", "S2_03": "S2", "S2_04": "S2", "S2_05": "S2",
    "S2_06": "S2", "S2_07": "S2", "S2_08": "S2", "S2_09": "S2", "S2_10": "S2",
    "S3_01": "S3", "S3_02": "S3", "S3_03": "S3", "S3_04": "S3", "S3_05": "S3",
    "S3_06": "S3", "S3_07": "S3", "S3_08": "S3", "S3_09": "S3", "S3_10": "S3"
}

# --- System prompt ---
# Aligned with thesis methodology:
# - No malice judgment language (addresses prompt bias identified in ablation study)
# - Explicit zero-IOC instruction (handles control rows like S1_06)
# - No subjective filtering before extraction
# - IoC-only extraction focus to reduce attention dilution
SYSTEM_PROMPT = """You are a Cyber Threat Intelligence (CTI) extraction tool. Your only task is to extract Indicators of Compromise (IoCs) from the provided threat intelligence report.

Output strictly in this JSON format. Do not add explanations, commentary, or external knowledge. Ground your extraction entirely on the text provided. If a value is not present in the text, output an empty list [].

IMPORTANT: Some reports contain no IoCs at all. If no IP addresses, domains, or file hashes are present in the report text, return empty lists [] for all three fields. Do not invent, infer, or hallucinate IoCs that are not explicitly stated in the report text.

{
    "Indicators_of_Compromise": {
        "IPv4_Addresses": ["List all IPv4 addresses found in the text"],
        "Domains": ["List all domains or URLs found in the text"],
        "Hashes": ["List all file hashes found in the text"]
    }
}"""


def extract_json_safe(raw_content):
    """
    Attempts to parse JSON from model output.
    Uses a nested-structure-aware regex as fallback for
    models that do not support json_object response format
    (Gemini 2.5 Pro, DeepSeek R1).
    """
    # First attempt: direct parse
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError:
        pass

    # Second attempt: extract outermost JSON object with nesting awareness
    depth = 0
    start = None
    for i, char in enumerate(raw_content):
        if char == '{':
            if depth == 0:
                start = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0 and start is not None:
                candidate = raw_content[start:i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

    raise ValueError("No valid JSON structure found in model output.")


def evaluate_reports(input_dir="dataset/", output_dir="results/", n_passes=TOTAL_PASSES):
    """
    Core experiment engine.
    Iterates all 30 .md reports through all 4 models for n_passes each.
    Saves one JSON file per report/model combination.
    Logs all activity to timestamped audit file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    report_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".md")])
    total_reports = len(report_files)

    logging.info(
        f"LAUNCHING PRODUCTION RUN: {n_passes} passes | "
        f"{len(MODELS_TO_TEST)} models | "
        f"{total_reports} reports | "
        f"Total attempts: {n_passes * len(MODELS_TO_TEST) * total_reports}"
    )

    for file_index, filename in enumerate(report_files, 1):
        report_id = filename.split(".")[0]
        stratum = STRATUM_MAP.get(report_id, "Unknown")

        with open(os.path.join(input_dir, filename), 'r', encoding='utf-8') as f:
            report_text = f.read()

        if not report_text.strip():
            logging.warning(f"Skipping {filename}: empty file")
            continue

        logging.info(
            f"[{file_index}/{total_reports}] Report: {report_id} "
            f"(Stratum: {stratum}) | Length: {len(report_text.split())} words"
        )

        for model_index, model in enumerate(MODELS_TO_TEST, 1):
            safe_model_name = model.replace("/", "_")
            output_file = os.path.join(
                output_dir,
                f"{report_id}_{safe_model_name}_results.json"
            )

            # Skip if already completed (allows safe resume after interruption)
            if os.path.exists(output_file):
                logging.info(
                    f"  [{model_index}/{len(MODELS_TO_TEST)}] "
                    f"Skipping {model} - output already exists"
                )
                continue

            logging.info(
                f"  [{model_index}/{len(MODELS_TO_TEST)}] "
                f"Model: {model}"
            )

            report_results = {
                "report_id": report_id,
                "stratum": stratum,
                "model": model,
                "total_passes": n_passes,
                "passes": []
            }

            for i in range(n_passes):
                start_time = time.time()
                try:
                    api_kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": f"Report Text:\n{report_text}"}
                        ],
                        "temperature": 0.0
                    }

                    if model in JSON_MODE_SUPPORTED:
                        api_kwargs["response_format"] = {"type": "json_object"}

                    response = client.chat.completions.create(**api_kwargs)
                    latency = time.time() - start_time
                    raw_content = response.choices[0].message.content

                    extracted_json = extract_json_safe(raw_content)

                    report_results["passes"].append({
                        "pass_number": i + 1,
                        "latency_seconds": round(latency, 2),
                        "extraction": extracted_json,
                        "error": None
                    })

                    logging.info(
                        f"    Pass {i + 1}/{n_passes} OK "
                        f"({round(latency, 2)}s)"
                    )

                except Exception as e:
                    latency = time.time() - start_time
                    report_results["passes"].append({
                        "pass_number": i + 1,
                        "latency_seconds": round(latency, 2),
                        "extraction": None,
                        "error": str(e)
                    })
                    logging.error(f"    Pass {i + 1}/{n_passes} FAIL - Error: {e}")

                # Rate limiting between passes
                if i < n_passes - 1:
                    time.sleep(2)

            # Incremental save after all passes for this model
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report_results, f, indent=4)

            logging.info(f"  Saved: {output_file}")

            # Inter-model sleep to avoid rate limits on OpenRouter
            if model_index < len(MODELS_TO_TEST):
                time.sleep(5)

        logging.info(f"Completed report {report_id}")

    logging.info(
        f"PIPELINE EXECUTION COMPLETE. "
        f"All results saved to '{output_dir}/'"
    )


if __name__ == "__main__":
    evaluate_reports(
        input_dir="dataset",
        output_dir="results",
        n_passes=TOTAL_PASSES
    )