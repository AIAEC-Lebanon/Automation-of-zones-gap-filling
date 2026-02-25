"""
classify_zones.py

Live Gradio interface for zoning label classification using Azure OpenAI.

Run:
    python classify_zones.py
Then open http://127.0.0.1:7860 in your browser.

Outputs are saved to the output/ subfolder.
"""

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
from dotenv import load_dotenv
from openai import AzureOpenAI

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

load_dotenv()

BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


def generate_response(messages):
    response = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
        messages=messages,
        reasoning_effort="medium",
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a zoning GIS expert specializing in resolving map label mismatches against official zoning ordinances.

Your job: given a map label and a list of official zoning districts (code + description) for the same jurisdiction, identify which zone(s) the label refers to.

MATCHING RULES — read carefully:
1. Scan EVERY zone in the list before answering. Do not stop at the first close match.
2. Match against BOTH the zone CODE and the zone DESCRIPTION.
3. Be thorough and practical — map labels are often informal, abbreviated, or composite:
   - "MH-435" likely refers to zone "MH" (the number suffix is a parcel/plan ID, not part of the district code)
   - "Multiple" matches any zone whose description contains the word "Multiple"
   - "Park" matches any zone whose code or description contains "Park"
   - "2F/PH" may match two separate zones: "2F" and "PH"
   - "OS" matches a zone coded "OS" or described as "Open Space"
   - "Government - Park" matches a zone with "Government" or "Park" in its code/description
4. Zoning codes are character-precise: "R-1" ≠ "R1". Never strip dashes, dots, slashes, or asterisks from codes.
5. However, a label prefix CAN match a code — e.g. label "MH-435" starts with "MH" which IS a valid zone code.
6. Only return not_found when you have checked every zone and none are plausibly related to the label.
7. Return valid JSON only — no explanation, no markdown fences."""


def build_user_prompt(label: str, location_id: str, zones: list[dict]) -> str:
    zone_lines = "\n".join(
        f"  {i+1:3d}. {z['code']:<20} | {z['description']}"
        for i, z in enumerate(zones)
    )
    return f"""JURISDICTION: Location ID {location_id}
TOTAL ZONES AVAILABLE: {len(zones)}

Official zoning districts for this location:
{zone_lines}

MAP LABEL TO CLASSIFY: "{label}"

Check every zone above. Does this label refer to one or more of these official districts?
Return only this JSON (no extra text):

If matched:
{{
  "status": "resolved",
  "matched_codes": ["EXACT_CODE_FROM_LIST"]
}}

If nothing matches after checking all zones:
{{
  "status": "not_found",
  "matched_codes": []
}}

Important: matched_codes must contain the exact code strings from the list above."""


# ---------------------------------------------------------------------------
# Data loading (once at startup)
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    query_df = pd.read_csv(
        BASE_DIR / "query_output.csv",
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        keep_default_na=False,
    )
    zones_df = pd.read_csv(
        BASE_DIR / "zones.csv",
        dtype=str,
        encoding="utf-8",
        encoding_errors="replace",
        keep_default_na=False,
    )
    query_df["location_id"] = query_df["location_id"].str.strip()
    zones_df["location_id"] = zones_df["location_id"].str.strip()
    return query_df, zones_df


def build_zone_index(zones_df: pd.DataFrame) -> dict[str, list[dict]]:
    zone_index: dict[str, list[dict]] = {}
    for _, row in zones_df.iterrows():
        lid = row["location_id"]
        zone_index.setdefault(lid, []).append(
            {"id": row["id"], "code": row["code"], "description": row["description"]}
        )
    return zone_index


print("Loading data files...")
query_df, zones_df = load_data()
zone_index = build_zone_index(zones_df)
TOTAL_ROWS = len(query_df)
print(f"Ready — {TOTAL_ROWS} query rows | {len(zone_index)} locations indexed")


# ---------------------------------------------------------------------------
# AI classification
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None
    return None


def classify_label(
    label: str,
    location_id: str,
    zones: list[dict],
    retries: int = 1,
) -> dict:
    if not zones:
        return _make_result("not_found_location", [], zones)
    if not label.strip():
        return _make_result("not_found_zone", [], zones)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(label, location_id, zones)},
    ]
    valid_codes = {z["code"] for z in zones}

    for attempt in range(retries + 1):
        try:
            raw = generate_response(messages)
            parsed = extract_json(raw)
            if parsed is None:
                if attempt < retries:
                    continue
                return _make_result("error", [], zones)

            status = parsed.get("status", "not_found")
            claimed_codes = parsed.get("matched_codes", [])
            if not isinstance(claimed_codes, list):
                claimed_codes = []

            # Hallucination guard
            verified_codes = [c for c in claimed_codes if c in valid_codes]
            if status == "resolved" and not verified_codes:
                status = "not_found_zone"
            elif status == "not_found":
                status = "not_found_zone"

            return _make_result(status, verified_codes, zones)

        except Exception as e:
            if attempt < retries:
                time.sleep(1)
                continue
            return _make_result("error", [], zones)

    return _make_result("error", [], zones)


def _make_result(status: str, matched_codes: list[str], zones: list[dict]) -> dict:
    code_to_desc = {z["code"]: z["description"] for z in zones}
    code_to_id   = {z["code"]: z["id"]          for z in zones}
    return {
        "match_status":         status,
        "matched_codes":        "|".join(matched_codes),
        "matched_descriptions": "|".join(code_to_desc.get(c, "") for c in matched_codes),
        "matched_ids":          [code_to_id[c] for c in matched_codes if c in code_to_id],
    }


def make_fix_command(
    status: str,
    location_id: str,
    label: str,
    matched_ids: list,
    matched_descriptions: str,
) -> str:
    loc = int(location_id)
    if status == "resolved":
        ids = [int(i) for i in matched_ids]
        first_desc = matched_descriptions.split("|")[0] if matched_descriptions else label
        return f'copy_zone({loc}, {ids}, "{label}", "{first_desc}", None)'
    elif status == "not_found_zone":
        return f'insert_zone({loc}, "{label}", "{label}")'
    return ""


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def save_outputs(output_df: pd.DataFrame, elapsed: float) -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = OUTPUT_DIR / f"classification_{ts}.csv"
    txt_path = OUTPUT_DIR / f"classification_{ts}.txt"

    output_df.to_csv(csv_path, index=False, encoding="utf-8")

    total = len(output_df)
    resolved = (output_df["match_status"] == "resolved").sum()
    no_location = (output_df["match_status"] == "not_found_location").sum()
    no_match = (output_df["match_status"] == "not_found_zone").sum()
    errors = (output_df["match_status"] == "error").sum()

    lines = [
        "=" * 60,
        "ZONING LABEL CLASSIFICATION REPORT",
        "=" * 60,
        f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Elapsed   : {elapsed:.1f}s",
        "",
        "SUMMARY",
        "-" * 40,
        f"Total rows  : {total}",
        f"Resolved    : {resolved}  ({100*resolved/total:.1f}%)" if total else "Resolved    : 0",
        f"No Location : {no_location}  ({100*no_location/total:.1f}%)" if total else "No Location : 0",
        f"No Match    : {no_match}  ({100*no_match/total:.1f}%)" if total else "No Match    : 0",
        f"Errors      : {errors}",
    ]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(csv_path)


# ---------------------------------------------------------------------------
# Gradio UI builders
# ---------------------------------------------------------------------------

def _esc(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def build_stats_html(results: list, total_target: int, elapsed: float, done: bool) -> str:
    processed = len(results)
    resolved = sum(1 for r in results if r["match_status"] == "resolved")
    no_location = sum(1 for r in results if r["match_status"] == "not_found_location")
    no_match = sum(1 for r in results if r["match_status"] == "not_found_zone")
    errors = sum(1 for r in results if r["match_status"] == "error")
    pct = int(100 * processed / total_target) if total_target else 0
    status_label = "Complete" if done else f"Processing… {pct}%"
    status_color = "#16a34a" if done else "#2563eb"

    return f"""
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;display:flex;gap:12px;flex-wrap:wrap;padding:4px 0">
  <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;padding:14px 20px;min-width:120px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#0369a1">{processed}<span style="font-size:14px;color:#64748b"> / {total_target}</span></div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-top:3px">Processed</div>
  </div>
  <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;padding:14px 20px;min-width:120px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#16a34a">{resolved}</div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-top:3px">Resolved</div>
  </div>
  <div style="background:#f5f3ff;border:1px solid #ddd6fe;border-radius:10px;padding:14px 20px;min-width:120px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#7c3aed">{no_location}</div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-top:3px">No Location</div>
  </div>
  <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;padding:14px 20px;min-width:120px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#dc2626">{no_match}</div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-top:3px">No Match</div>
  </div>
  <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:10px;padding:14px 20px;min-width:120px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#d97706">{errors}</div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-top:3px">Errors</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 20px;min-width:120px;text-align:center">
    <div style="font-size:26px;font-weight:700;color:#475569">{elapsed:.0f}s</div>
    <div style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-top:3px">Elapsed</div>
  </div>
  <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:14px 20px;min-width:140px;text-align:center;flex:1">
    <div style="font-size:15px;font-weight:700;color:{status_color};margin-top:4px">{status_label}</div>
    <div style="background:#e2e8f0;border-radius:4px;height:6px;margin-top:8px;overflow:hidden">
      <div style="background:{status_color};height:100%;width:{pct}%;transition:width .3s"></div>
    </div>
  </div>
</div>"""


def build_table_html(results: list) -> str:
    if not results:
        return "<p style='color:#94a3b8;font-style:italic;padding:8px 0'>Results will appear here as rows are processed…</p>"

    rows_html = ""
    for r in reversed(results):  # newest first
        status = r["match_status"]
        if status == "resolved":
            row_bg = "#f0fdf4"
            badge = '<span style="background:#dcfce7;color:#15803d;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;letter-spacing:.4px">RESOLVED</span>'
        elif status == "not_found_location":
            row_bg = "#f5f3ff"
            badge = '<span style="background:#ede9fe;color:#6d28d9;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;letter-spacing:.4px">NO LOCATION</span>'
        elif status == "not_found_zone":
            row_bg = "#fef2f2"
            badge = '<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;letter-spacing:.4px">NO MATCH</span>'
        elif status == "error":
            row_bg = "#fffbeb"
            badge = '<span style="background:#fef3c7;color:#92400e;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;letter-spacing:.4px">ERROR</span>'
        else:
            row_bg = "#fef2f2"
            badge = '<span style="background:#fee2e2;color:#b91c1c;padding:3px 10px;border-radius:12px;font-size:11px;font-weight:700;letter-spacing:.4px">NO MATCH</span>'

        matched_codes = _esc(r["matched_codes"]) if r["matched_codes"] else "<span style='color:#94a3b8'>—</span>"
        matched_desc = _esc(r["matched_descriptions"]) if r["matched_descriptions"] else "<span style='color:#94a3b8'>—</span>"
        fix_cmd = _esc(r.get("fix_command", "")) or "<span style='color:#94a3b8'>—</span>"

        rows_html += f"""
        <tr style="background:{row_bg};border-bottom:1px solid #f1f5f9">
          <td style="padding:10px 12px;font-weight:600;font-size:13px">{_esc(r['label'])}</td>
          <td style="padding:10px 12px;font-size:13px;color:#64748b">{_esc(r['location_id'])}</td>
          <td style="padding:10px 12px;font-size:12px;color:#94a3b8">{_esc(r['feature_set_name'])}</td>
          <td style="padding:10px 12px">{badge}</td>
          <td style="padding:10px 12px;font-family:monospace;font-size:12px;color:#1d4ed8">{matched_codes}</td>
          <td style="padding:10px 12px;font-size:12px;color:#334155">{matched_desc}</td>
          <td style="padding:10px 12px;font-family:monospace;font-size:11px;color:#7c3aed;white-space:nowrap">{fix_cmd}</td>
        </tr>"""

    TH = "padding:11px 12px;text-align:left;font-size:12px;font-weight:600;letter-spacing:.5px;color:#ffffff"
    return f"""
<div style="overflow-x:auto;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
  <table style="width:100%;border-collapse:collapse;background:white">
    <thead>
      <tr style="background:#1e293b">
        <th style="{TH}">LABEL</th>
        <th style="{TH}">LOCATION ID</th>
        <th style="{TH}">FEATURE SET</th>
        <th style="{TH}">STATUS</th>
        <th style="{TH}">MATCHED CODES</th>
        <th style="{TH}">MATCHED DESCRIPTIONS</th>
        <th style="{TH}">FIX COMMAND</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""


# ---------------------------------------------------------------------------
# Streaming classification generator
# ---------------------------------------------------------------------------

def stream_classification(n_rows: int, n_workers: int):
    """Generator: yields (log_text, stats_html, table_html) as rows complete.

    Rows are processed in parallel batches of n_workers using a thread pool.
    Results stream back to the UI as each thread finishes (via as_completed),
    so the table and log update in real time even within a batch.
    """
    n_rows = max(1, min(int(n_rows), TOTAL_ROWS))
    n_workers = max(1, min(int(n_workers), 32))
    rows_list = list(query_df.head(n_rows).itertuples(index=False))
    total = len(rows_list)
    results = []
    log_lines = []
    start_time = time.time()

    for batch_start in range(0, total, n_workers):
        batch = rows_list[batch_start : batch_start + n_workers]
        batch_num = batch_start // n_workers + 1
        batch_end = batch_start + len(batch)
        elapsed = time.time() - start_time

        log_lines.append(
            f"── Batch {batch_num}  (rows {batch_start+1}–{batch_end}, {len(batch)} workers) ──"
        )
        yield (
            "\n".join(log_lines[-24:]),
            build_stats_html(results, total, elapsed, done=False),
            build_table_html(results),
        )

        # Submit all rows in this batch to the thread pool
        future_to_meta = {}
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for i, row in enumerate(batch):
                global_i = batch_start + i + 1
                label = str(row.label)
                location_id = str(row.location_id)
                zones = zone_index.get(location_id, [])
                future = executor.submit(classify_label, label, location_id, zones)
                future_to_meta[future] = (global_i, row, label, location_id)

            # Yield a result to the UI as each thread finishes
            for future in as_completed(future_to_meta):
                global_i, row, label, location_id = future_to_meta[future]
                result = future.result()
                elapsed = time.time() - start_time

                status = result["match_status"]
                if status == "resolved":
                    tag = "RESOLVED "
                elif status == "not_found_location":
                    tag = "NO LOCATN"
                elif status == "not_found_zone":
                    tag = "NO MATCH "
                else:
                    tag = "ERROR    "
                extra = f"  ->  {result['matched_codes']}" if result["matched_codes"] else ""
                log_lines.append(f"  [{global_i:3d}/{total}] [{tag}] '{label}'{extra}")

                fix_command = make_fix_command(
                    status,
                    location_id,
                    label,
                    result.get("matched_ids", []),
                    result["matched_descriptions"],
                )

                results.append(
                    {
                        "label": label,
                        "location_id": location_id,
                        "feature_set_name": str(row.feature_set_name),
                        "match_status": status,
                        "matched_codes": result["matched_codes"],
                        "matched_descriptions": result["matched_descriptions"],
                        "fix_command": fix_command,
                    }
                )

                yield (
                    "\n".join(log_lines[-24:]),
                    build_stats_html(results, total, elapsed, done=False),
                    build_table_html(results),
                )

    # All batches done — save outputs
    elapsed = time.time() - start_time
    output_df = pd.DataFrame(results)
    csv_path = save_outputs(output_df, elapsed)
    log_lines.append(f"\nDone!  Outputs saved -> {csv_path}")

    yield (
        "\n".join(log_lines[-24:]),
        build_stats_html(results, total, elapsed, done=True),
        build_table_html(results),
    )


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------

DESCRIPTION = """
<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;padding:4px 0 12px">
  <h1 style="font-size:22px;font-weight:700;color:#1e293b;margin:0 0 4px">🗺️ Zoning Label Classifier</h1>
  <p style="font-size:14px;color:#64748b;margin:0">
    Reads <code>query_output.csv</code> and <code>zones.csv</code>, then uses Azure OpenAI to classify
    each zoning label as <b style="color:#16a34a">RESOLVED</b> (matched to a zone) or
    <b style="color:#dc2626">NOT FOUND</b>. Results stream live as each row is processed.
    Outputs are saved to the <code>output/</code> folder with a timestamp.
  </p>
  <p style="font-size:12px;color:#94a3b8;margin-top:6px">
    Dataset: <b>{total}</b> rows available · <b>{locations}</b> locations indexed
  </p>
</div>
""".format(total=TOTAL_ROWS, locations=len(zone_index))

with gr.Blocks(title="Zoning Label Classifier") as demo:

    gr.HTML(DESCRIPTION)

    with gr.Row():
        n_rows_input = gr.Number(
            value=10,
            minimum=1,
            maximum=TOTAL_ROWS,
            step=1,
            label=f"Rows to process (max {TOTAL_ROWS})",
            scale=2,
        )
        n_workers_input = gr.Number(
            value=4,
            minimum=1,
            maximum=32,
            step=1,
            label="Parallel workers (1 = sequential)",
            scale=2,
        )
        run_btn = gr.Button("▶  Start Classification", variant="primary", scale=2)
        stop_btn = gr.Button("⏹  Stop", variant="stop", scale=1)

    stats_out = gr.HTML(
        value="<p style='color:#94a3b8;font-size:13px;padding:4px 0'>Press <b>Start Classification</b> to begin.</p>"
    )

    log_out = gr.Textbox(
        label="Live Processing Log",
        lines=12,
        max_lines=12,
        interactive=False,
    )

    table_out = gr.HTML(
        value="<p style='color:#94a3b8;font-style:italic;padding:8px 0'>Results will appear here as rows are processed…</p>"
    )

    run_event = run_btn.click(
        fn=stream_classification,
        inputs=[n_rows_input, n_workers_input],
        outputs=[log_out, stats_out, table_out],
    )

    stop_btn.click(fn=None, cancels=[run_event])


if __name__ == "__main__":
    demo.launch(
        inbrowser=True,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        
    )
