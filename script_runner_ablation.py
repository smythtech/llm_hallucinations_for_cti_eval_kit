import os
import json
import time
import re
import logging
from openai import OpenAI

run_timestamp = time.strftime('%Y%m%d_%H%M%S')
log_filename = f"ablation_audit_{run_timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()
    ]
)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="[PLACEHOLDER FOR API_KEY]",
    max_retries=0,
    timeout=300.0
)

MODELS_TO_TEST = [
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-r1"
]

# Models that do NOT support response_format=json_object
NO_JSON_FORMAT_MODELS = [
    "google/gemini-2.5-pro",
    "deepseek/deepseek-r1"
]

STRATUM_MAP = {
    "S1_01": "S1", "S1_02": "S1", "S1_03": "S1", "S1_04": "S1", "S1_05": "S1",
    "S1_06": "S1", "S1_07": "S1", "S1_08": "S1", "S1_09": "S1", "S1_10": "S1",
    "S2_01": "S2", "S2_02": "S2", "S2_03": "S2", "S2_04": "S2", "S2_05": "S2",
    "S2_06": "S2", "S2_07": "S2", "S2_08": "S2", "S2_09": "S2", "S2_10": "S2",
    "S3_01": "S3", "S3_02": "S3", "S3_03": "S3", "S3_04": "S3", "S3_05": "S3",
    "S3_06": "S3", "S3_07": "S3", "S3_08": "S3", "S3_09": "S3", "S3_10": "S3"
}

PROMPT_VARIANTS = [
    {
        "label": "PromptA_IoCOnly_Temp0",
        "temperature": 0.0,
        "system": """You are a Cyber Threat Intelligence extraction tool. Your only task is to extract Indicators of Compromise (IoCs) from the provided report.

Output strictly in this JSON format. Do not add explanations, commentary, or external knowledge. If a value is not present in the text, output an empty list [].

{
  "Indicators_of_Compromise": {
      "IPv4_Addresses": ["List all IPv4 addresses found in the text"],
      "Domains": ["List all domains or URLs found in the text"],
      "Hashes": ["List all file hashes found in the text"]
  }
}"""
    },
    {
        "label": "PromptB_NoMalice_Temp0",
        "temperature": 0.0,
        "system": """You are an expert Cyber Threat Intelligence (CTI) analyst. Extract key entities from the provided threat intelligence report.

Output strictly in the following JSON format. Do not include conversational text, explanations, or external knowledge. Ground your extraction entirely on the provided text. If a field is not present, output null or [].

{
  "Adversary": "The primary threat actor or APT group mentioned.",
  "Target_Asset": "The victim organization, sector, or system targeted.",
  "Operational_Metric": "Any measurable time, date, or duration related to the campaign.",
  "Technical_Vector": "The primary malware, CVE, or technical tool used.",
  "Indicators_of_Compromise": {
      "IPv4_Addresses": ["List all IPv4 addresses found in the text"],
      "Domains": ["List all domains or URLs found in the text"],
      "Hashes": ["List all file hashes found in the text"]
  }
}"""
    },
    {
        "label": "PromptC_Production_Temp0",
        "temperature": 0.0,
        "system": """You are an expert Cyber Threat Intelligence (CTI) analyst. Your task is to extract key entities from the provided threat intelligence report.

Extract the information strictly into the following JSON format. Do not include any conversational text, explanations, or external knowledge. Ground your extraction entirely on the provided text. If a field is not explicitly mentioned in the text, output null or an empty list [].

If the report contains no Indicators of Compromise, return empty lists [] for all IOC fields. Do not invent or infer IOCs that are not explicitly present in the text.

{
  "Adversary": "The primary threat actor, APT group, or state-sponsored entity conducting the attack.",
  "Target_Asset": "The victim organization, sector, geolocation, or specific system targeted.",
  "Operational_Metric": "A measurable time, date, or duration related to the campaign.",
  "Technical_Vector": "The primary custom malware, vulnerability (CVE), or distinct technical tool used to execute the attack.",
  "Indicators_of_Compromise": {
      "IPv4_Addresses": ["Extract all IPv4 addresses explicitly stated in the text"],
      "Domains": ["Extract all domains or URLs explicitly stated in the text"],
      "Hashes": ["Extract all file hashes explicitly stated in the text"]
  }
}"""
    },
    {
        "label": "PromptD_ZeroIOC_Temp0",
        "temperature": 0.0,
        "system": """You are an expert Cyber Threat Intelligence (CTI) analyst. Your task is to extract key entities from the provided threat intelligence report.

Extract the information strictly into the following JSON format. Do not include any conversational text, explanations, or external knowledge. Ground your extraction entirely on the provided text. If a field is not explicitly mentioned in the text, output null or an empty list [].

IMPORTANT: If this report contains no Indicators of Compromise, you MUST return empty lists [] for all IOC fields. Do not hallucinate, infer, or generate IOCs that are not explicitly present in the report text. Returning an empty list is a valid and correct response.

{
  "Adversary": "The primary threat actor, APT group, or state-sponsored entity conducting the attack.",
  "Target_Asset": "The victim organization, sector, geolocation, or specific system targeted.",
  "Operational_Metric": "A measurable time, date, or duration related to the campaign.",
  "Technical_Vector": "The primary custom malware, vulnerability (CVE), or distinct technical tool used to execute the attack.",
  "Indicators_of_Compromise": {
      "IPv4_Addresses": ["Extract all IPv4 addresses explicitly stated in the text"],
      "Domains": ["Extract all domains or URLs explicitly stated in the text"],
      "Hashes": ["Extract all file hashes explicitly stated in the text"]
  }
}"""
    }
]

TOTAL_PASSES = 5
ABLATION_INPUT_DIR = "dataset_ablation/"


def run_ablation(output_dir="results_ablation/", n_passes=TOTAL_PASSES):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    files = sorted([f for f in os.listdir(ABLATION_INPUT_DIR) if f.endswith(".md")])
    logging.info(f"Found {len(files)} reports for ablation.")

    for filename in files:
        report_id = filename.split(".")[0]
        stratum = STRATUM_MAP.get(report_id, "Unknown")

        with open(os.path.join(ABLATION_INPUT_DIR, filename), 'r', encoding='utf-8') as f:
            report_text = f.read()

        if not report_text.strip():
            logging.warning(f"Skipping {filename}: empty file")
            continue

        for prompt in PROMPT_VARIANTS:
            for model in MODELS_TO_TEST:
                safe_model = model.replace("/", "_")
                label = prompt["label"]

                output_file = os.path.join(
                    output_dir,
                    f"{report_id}_{safe_model}_{label}_results.json"
                )

                if os.path.exists(output_file):
                    logging.info(f"[SKIP] {report_id} | {model} | {label} already exists.")
                    continue

                logging.info(f"[{label}] Processing {report_id} | {model} | Stratum: {stratum}")

                report_results = {
                    "report_id": report_id,
                    "stratum": stratum,
                    "model": model,
                    "prompt_variant": label,
                    "passes": []
                }

                for i in range(n_passes):
                    start_time = time.time()
                    try:
                        # Conditional response_format
                        extra_kwargs = {}
                        if model not in NO_JSON_FORMAT_MODELS:
                            extra_kwargs["response_format"] = {"type": "json_object"}

                        response = client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": prompt["system"]},
                                {"role": "user", "content": f"Report Text:\n{report_text}"}
                            ],
                            temperature=prompt["temperature"],
                            **extra_kwargs
                        )
                        latency = time.time() - start_time
                        raw_content = response.choices[0].message.content

                        # JSON extraction with regex fallback
                        try:
                            extracted_json = json.loads(raw_content)
                        except json.JSONDecodeError:
                            matches = re.findall(r'\{.*\}', raw_content, re.DOTALL)
                            if matches:
                                extracted_json = json.loads(matches[-1])
                            else:
                                raise ValueError("No valid JSON found in output.")

                        report_results["passes"].append({
                            "pass_number": i + 1,
                            "latency_seconds": round(latency, 2),
                            "extraction": extracted_json,
                            "error": None
                        })
                        logging.info(f"  - Pass {i+1}/{n_passes} complete ({round(latency, 2)}s)")

                    except Exception as e:
                        latency = time.time() - start_time
                        report_results["passes"].append({
                            "pass_number": i + 1,
                            "latency_seconds": round(latency, 2),
                            "extraction": None,
                            "error": str(e)
                        })
                        logging.error(f"  ! Error on pass {i+1}: {e}")

                    # Longer sleep for DeepSeek
                    if i < n_passes - 1:
                        sleep_time = 5 if "deepseek" in model else 2
                        time.sleep(sleep_time)

                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(report_results, f, indent=4)
                logging.info(f"  [SAVED] {output_file}")


if __name__ == "__main__":
    logging.info("LAUNCHING ABLATION STUDY RUN")
    run_ablation()
    logging.info("ABLATION STUDY COMPLETE. Results saved to results_ablation/")