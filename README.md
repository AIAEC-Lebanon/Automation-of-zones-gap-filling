# Zoning Label Classifier

An AI-powered tool that automatically matches informal zoning map labels to official zoning district codes using Azure OpenAI.

## What it does

Given a list of map labels (e.g. `"MH-435"`, `"OS"`, `"2F/PH"`) and a database of official zoning districts, it figures out which official zone each label refers to — even when the labels are abbreviated, composite, or informally written.

Each label is classified as one of four statuses:

| Status | Meaning |
|---|---|
| **RESOLVED** | Matched to one or more official zone codes |
| **NO LOCATION** | No zone data exists for that location ID |
| **NO MATCH** | Zone data exists but the label didn't match any zone |
| **ERROR** | Something went wrong during processing |

For each result, the tool also generates a ready-to-use **fix command**:
- `RESOLVED` → `copy_zone(location_id, [zone_ids], "label", "description", None)`
- `NO MATCH` → `insert_zone(location_id, "label", "label")`

Results stream live in a browser UI and are saved to the `output/` folder.

## How it works

1. Reads `query_output.csv` (labels to classify) and `zones.csv` (official zone codes per location)
2. For each label, sends a prompt to Azure OpenAI with the full list of zones for that jurisdiction
3. The model returns the matched zone code(s) in JSON
4. All returned codes are verified against the actual zone list (hallucination guard)
5. Results stream to the browser in real time as each row completes

## Setup

**1. Install dependencies**
```bash
pip install gradio openai pandas python-dotenv
```

**2. Create a `.env` file** with your Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
```

**3. Prepare your CSV files** in the project root:

`query_output.csv` — labels to classify:
| location_id | label | feature_set_name |
|---|---|---|
| 101 | MH-435 | Parcel Layer |

`zones.csv` — official zone reference:
| location_id | id | code | description |
|---|---|---|---|
| 101 | 1 | MH | Mobile Home District |

## Run

```bash
python classify_zones.py
```

Opens a browser at `http://127.0.0.1:7860`.

- Set the number of rows to process and how many parallel workers to use
- Click **Start Classification** to begin — results stream in as each row finishes
- Click **Stop** to cancel mid-run

## Output

Two files are saved to `output/` with a timestamp:

- `classification_YYYYMMDD_HHMMSS.csv` — full results with columns: `label`, `location_id`, `feature_set_name`, `match_status`, `matched_codes`, `matched_descriptions`, `fix_command`
- `classification_YYYYMMDD_HHMMSS.txt` — summary report with counts and percentages per status
