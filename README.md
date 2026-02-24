# Zoning Label Classifier

An AI-powered tool that automatically matches informal zoning map labels to official zoning district codes using Azure OpenAI.

## What it does

Given a list of map labels (e.g. `"MH-435"`, `"OS"`, `"2F/PH"`) and a database of official zoning districts, it figures out which official zone each label refers to — even when the labels are abbreviated, composite, or informally written.

Each label is classified as:
- **RESOLVED** — matched to one or more official zone codes
- **NOT FOUND** — no plausible match found after checking all zones
- **ERROR** — something went wrong during processing

Results stream live in a browser UI and are saved to the `output/` folder as CSV + summary text files.

## How it works

1. Reads `query_output.csv` (labels to classify) and `zones.csv` (official zone codes per location)
2. For each label, sends a prompt to Azure OpenAI with the full list of zones for that jurisdiction
3. The model returns the matched zone code(s) in JSON
4. Results are verified against the actual zone list (hallucination guard) and displayed in real time

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

Opens a browser at `http://127.0.0.1:7860`. Set the number of rows and parallel workers, then click **Start Classification**.

## Output

Results are saved to `output/classification_YYYYMMDD_HHMMSS.csv` with matched codes, descriptions, and status for each row.
