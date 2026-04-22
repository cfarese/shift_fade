# Shift Fade

Live app: [shiftfade.app](https://shiftfade.app)

Shift Fade is a hockey analytics project focused on one core question:

How does player and line performance change as a shift gets longer?

The project combines NHL play-by-play ingestion, shift/stint reconstruction, expected goals modeling, and a web app for exploring player, line, and team-level fade patterns at 5v5.

## What This Project Does

Shift Fade is built around observed 5v5 on-ice results. Instead of only asking whether a player or line is good overall, the app also asks:

- Are they strong right after a line change?
- Do they hold their level as the shift gets older?
- Do they fade faster or slower than league average?

The web app includes:

- A league overview page that explains the metrics and shows league-wide distributions
- Player pages with observed shift-fade charts and individual shift logs
- A lines page for analyzing 3-forward combinations
- A team report page for deployment and roster breakdown analysis

## Live Version

A live version of the app is available at [shiftfade.app](https://shiftfade.app).

## Main Stack

- Python for ingestion, data processing, API logic, and analytics
- FastAPI for the web server
- A small React frontend bundled with `esbuild`
- Parquet files for processed season data
- R for the older RAPM workflow that still exists in the repo

## Repo Structure

```text
hockey_analytics/
├── config/                  # project settings and path helpers
├── dashboard/
│   ├── app.py               # Streamlit dashboard entrypoint
│   └── web/
│       ├── server.py        # FastAPI server for the web app
│       ├── app.jsx          # main frontend source
│       ├── app.bundle.js    # built frontend bundle
│       └── template.html    # HTML shell and site-wide CSS
├── data/
│   ├── cache/               # cached payloads and helper artifacts
│   ├── processed/           # processed parquet outputs
│   └── raw/                 # optional raw files
├── r/
│   └── rapm/                # older RAPM modeling scripts
├── src/
│   ├── api/                 # API entrypoints
│   ├── features/            # matrix/feature generation
│   ├── ingestion/           # NHL ingestion and parsing
│   ├── models/              # model logic, readers, xG model
│   └── utils/               # utility helpers
├── tests/                   # test suite
├── package.json             # frontend build step
└── requirements.txt         # Python dependencies
```

## Core Concepts

### xGD/60

`xGD/60` means expected goal differential per 60 minutes:

```text
(expected goals for - expected goals against) / time on ice * 3600
```

Positive values mean a player, line, or team is driving better chance quality than it allows. Negative values mean the opposite.

### Fade / Durability

The app compares performance early in a shift versus later in a shift.

- `Early xGD/60`: results in the first 30 seconds
- `Mid xGD/60`: results roughly in the 30 to 45 second window
- `Late xGD/60`: results after 45 seconds
- `Durability`: late xGD/60 minus early xGD/60

More negative durability means a bigger drop-off as the shift gets longer.

### Qualifying Sample Rule

To keep major charts and leaderboards from getting distorted by tiny samples, the app requires:

- at least `200` merged player shifts
- where the shift length exceeds `10` seconds

Low-sample players can still be searched and opened, but their big charts show `Not enough data`.

## Data Pipeline Overview

At a high level, the project works like this:

1. Pull NHL play-by-play and shift-related data
2. Parse events into player on-ice stint rows
3. Merge clock-adjacent player sub-stints into shifts
4. Score shot events with an xG model
5. Aggregate those stints into player, line, and team summaries
6. Serve the processed data through the web app

## Expected Goals Model

The repo now includes a repo-native shot-based xG model in:

- [src/models/xg_model.py](/Users/christianfarese/Projects/hockey_analytics/src/models/xg_model.py)

The model is designed around unblocked shot attempts and uses:

- shot location
- shot type
- score state
- rebound context
- rush context
- season effects
- rink effects

Its structure was inspired by the methodology in:

- [Building an xG model v1.0](https://hockey-statistics.com/2022/08/14/building-an-xg-model-v-1-0/)

The xG model can be trained and saved to:

- `data/processed/xg_model_v1.json`

Once trained, the parser will automatically use it when building new stint files.

## Local Setup

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Install frontend build dependency

```bash
npm install
```

## Running the Web App

The main local app entrypoint is the FastAPI server in:

- [dashboard/web/server.py](/Users/christianfarese/Projects/hockey_analytics/dashboard/web/server.py)

Run it with:

```bash
uvicorn dashboard.web.server:app --reload --port 8080
```

Then open:

- [http://localhost:8080](http://localhost:8080)

## Rebuilding the Frontend Bundle

If you change:

- `dashboard/web/app.jsx`
- `dashboard/web/template.html`

then rebuild the browser bundle with:

```bash
npm run build:web
```

## Streamlit Dashboard

The repo also still contains a Streamlit dashboard entrypoint:

- [dashboard/app.py](/Users/christianfarese/Projects/hockey_analytics/dashboard/app.py)

Run it with:

```bash
streamlit run dashboard/app.py
```

## Building Season Data

Processed season data lives in:

- `data/processed/stints_<season>.parquet`

Example season codes:

- `20232024`
- `20242025`
- `20252026`

To ingest a season:

```bash
python -m src.ingestion.pipeline --season 20242025
```

To force a rebuild:

```bash
rm -f data/processed/stints_20242025.parquet
rm -f data/cache/failed_games_20242025.txt
python -m src.ingestion.pipeline --season 20242025
```

## Training the xG Model

Train the xG model with one or more seasons:

```bash
python3 -m src.models.xg_model --seasons 20222023 20232024 20242025 --out data/processed/xg_model_v1.json
```

After training, regenerate season stints so the app uses the trained model:

```bash
rm -f data/processed/stints_20232024.parquet
python -m src.ingestion.pipeline --season 20232024
```

Repeat that for any seasons you want refreshed.

## Older RAPM Workflow

The repo still includes RAPM-era infrastructure. That flow looks like:

```bash
python -m src.features.export_matrix --season 20232024
Rscript r/rapm/rapm_model.R --season 20232024
python -m src.ingestion.resolve_names --season 20232024
```

Some older pieces of the repo still expect RAPM parquet outputs, but the web app has largely shifted toward observed xGD/60 and fade-based summaries.

## Running Tests

From the repo root:

```bash
PYTHONPATH=$(pwd) pytest -q
```

If your virtual environment is active, that is usually enough. If not:

```bash
PYTHONPATH=$(pwd) ./.venv/bin/python -m pytest -q
```

## Common Commands

### Start the app

```bash
uvicorn dashboard.web.server:app --reload --port 8080
```

### Rebuild the frontend

```bash
npm run build:web
```

### Clear cached web payloads

```bash
rm -f data/cache/web_payload_*.json
```

### Rebuild a season

```bash
rm -f data/processed/stints_20252026.parquet
rm -f data/cache/failed_games_20252026.txt
python -m src.ingestion.pipeline --season 20252026
```

## Troubleshooting

### The site loads but shows missing data

You probably do not have the necessary parquet files in `data/processed`.

### Player game labels show `Game <id>` instead of date/opponent

That season likely needs to be regenerated with the newer ingestion code so the stints parquet contains:

- `game_date`
- `home_team`
- `away_team`

### The first page load feels slow

The app uses cached season payloads in `data/cache`. If you change backend logic, clear:

```bash
rm -f data/cache/web_payload_*.json
```

### Team or player branding looks wrong

Some older seasons may rely on fallback team mapping if the stints parquet predates newer team metadata fields. Regenerating the season is the cleanest fix.

## Notes

- This project is optimized around 5v5 analysis.
- The Lines page currently analyzes 3-forward trios, not full 5-man units.
- The app is designed to be readable for both hockey analytics users and more casual visitors.

## Contact / Attribution

The live project is available at [shiftfade.app](https://shiftfade.app).

The xG-model approach was informed in part by:

- [hockey-statistics.com: Building an xG model v1.0](https://hockey-statistics.com/2022/08/14/building-an-xg-model-v-1-0/)
