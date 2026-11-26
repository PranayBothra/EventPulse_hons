# EventPulse

**Temporal Novelty-Aware Event Intelligence & Personalised News Recommendation**

EventPulse is a news intelligence and recommendation system that groups breaking news into evolving real-world events, measures what factual information is genuinely new at the sentence level, detects emerging topics in real time, and serves diverse, low-redundancy recommendations through a FastAPI backend and interactive web interface.

---

## Why EventPulse?

Traditional news recommenders often suffer from three common issues:
1. **Redundancy amplification:** Clicking one breaking article causes the feed to fill up with near-identical wire rewrites from different outlets.
2. **Popularity mistaken for novelty:** Rare items are treated as "novel" even when they do not add new facts for the reader.
3. **Article-level isolation:** Articles are scored individually instead of being tracked as evolving timelines of a shared real-world event.

EventPulse addresses these by combining online event clustering, chronological sentence-level event memory, multi-signal novelty scoring, streaming topic emergence detection, and dual-timescale user profiling with Maximal Marginal Relevance (MMR) reranking.

---

## Key Features

- **Multi-Source News Ingestion & Deduplication:** Collects news from **23 RSS/Atom feeds** (`config/feeds.yaml`) and **FreeNewsAPI.io**, normalizes HTML/text and UTC timestamps, and deduplicates by canonical URL and headline hash. Includes an offline sample dataset (`data/sample/sample_articles.json`) for instant local testing.
- **Background Live Sync & Batch Auto-Clustering:** Runs periodic background RSS (`5 min` default) and quota-safe FreeNewsAPI (`30 min` default) syncs, automatically clustering unassigned articles once they reach a configurable batch threshold (default `10`), with sidebar controls and non-disruptive feed refresh banners in the UI.
- **Online Multi-Signal Event Clustering:** Groups incoming articles into evolving events (`EMERGING`, `ACTIVE`, `DORMANT`, `INACTIVE`, `RESOLVED`) using 384-d `all-MiniLM-L6-v2` embeddings, `spaCy` named-entity overlap, temporal decay (48-hour half-life), and headline lexical similarity.
- **Chronological Sentence-Level Novelty:** Compares incoming sentences strictly against prior event sentences (`published_epoch < t`) with no lookahead leakage, combining semantic similarity, numerical delta checks, action/negation polarity flip detection, and promotional boilerplate filtering into `POTENTIALLY_NEW`, `REPEATED_INFORMATION`, `INSUFFICIENT_EVIDENCE`, or `BACKGROUND_INFORMATION`.
- **Emerging Topic Detection:** Surfaces newly forming topics from the article stream using two selectable detectors: `online_cluster` and `rolling_window`.
- **Personalised Recommendation & MMR Reranking:** Combines long-term and recency-decayed short-term user interest profiles with `EventPulse` modes (`RELEVANCE_FOCUSED`, `BALANCED`, `DISCOVERY_FOCUSED`, or custom weights + MMR $\lambda$) alongside `Baseline_A` (Chronological), `Baseline_B` (Content-Based), and `Baseline_C` (Event-Aware).
- **6-View Editorial Web UI (`static/`):**
  - **For You** — Personalised feed, mode/baseline switcher, interactive `Weights & MMR` controls, explanations, and live `Like` / `Mark Read` / `Dislike` persistence.
  - **Explore Events** — Searchable event clusters with lifecycle filters and chronological timelines.
  - **Novelty Inspector** — Sentence-level novelty scores, verdicts, and matched prior-memory evidence.
  - **Emerging Topics** — Live detector switching (`online_cluster` vs. `rolling_window`) and topic cards.
  - **My Interests** — Demo personas (`user_tech`, `user_geopolitics`, `user_science`, `user_cold_start`), topic keywords, and reading history.
  - **Benchmarks** — Archived MIND ranking comparison metrics and 50-event corpus novelty breakdown.

---

## Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Backend** | Python 3.12, FastAPI, Uvicorn, Pydantic v2 (`src/api/`) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript ES Modules (`static/`), inline SVG charts |
| **Database** | SQLite 3 (WAL mode, float32 embedding BLOB cache) |
| **NLP & ML** | `sentence-transformers` (`all-MiniLM-L6-v2`, CPU), `spaCy` (`en_core_web_sm`), `scikit-learn`, `numpy`, `scipy`, `PyTorch` |
| **Ingestion** | `feedparser`, `requests`, `beautifulsoup4`, `lxml`, `httpx`, `python-dotenv` |
| **Testing** | `pytest`, `pytest-cov`, FastAPI `TestClient` |

---

## Project Structure

```text
EventPulse_hons/
├── app.py                  # Optional Streamlit interface entry point
├── main.py                 # CLI entry point (ingest, cluster-events, stats, etc.)
├── requirements.txt        # Python dependencies
├── pytest.ini              # Pytest configuration
├── .env.example            # Environment variable template (FREENEWSAPI_API_KEY)
├── config/
│   ├── config.yaml         # Pipeline, embedding, clustering & DB settings
│   └── feeds.yaml          # 23 configured RSS/Atom news feeds
├── data/
│   └── sample/             # Offline sample news articles for quick start
├── reports/
│   ├── mind_benchmark_results.json
│   └── corpus_novelty_results.json
├── src/
│   ├── api/                # FastAPI app, routers, schemas, and services
│   ├── benchmarks/         # MIND dataset loader
│   ├── clustering/         # Online multi-signal & graph event clustering
│   ├── entities/           # Entity extraction and normalization
│   ├── evaluation/         # Ranking, clustering, ablation & statistical metrics
│   ├── ingestion/          # RSS, FreeNewsAPI, GDELT & historical loaders
│   ├── novelty/            # Chronological memory, multi-signal novelty & topic emergence
│   ├── personalisation/    # Dual-timescale user profiles & user-level novelty
│   ├── preprocessing/      # Text cleaning, UTC date parsing & deduplication
│   ├── recommendation/     # Multi-objective MMR recommender & baselines
│   ├── representation/     # Embedding generation, batching & SQLite cache
│   ├── storage/            # SQLite schema & repository layer
│   └── ui/                 # Legacy Streamlit & annotation apps
├── static/                 # HTML/CSS/JS frontend (6 interactive views)
└── tests/                  # Automated unit, API & temporal invariant tests
```

---

## Getting Started (Beginner-Friendly Guide)

### Prerequisites
- **Python 3.12** (recommended)
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/PranayBothra/EventPulse_hons.git
cd EventPulse_hons
```

### 2. Create & Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 4. Configure Environment Variables (Optional)
If you want to pull live articles from FreeNewsAPI.io, copy `.env.example` to `.env` and add your API key. Without a key, RSS feeds and the built-in sample dataset still work normally.

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
cp .env.example .env
```

Then edit `.env`:
```env
FREENEWSAPI_API_KEY=your_api_key_here
```

### 5. Seed Sample Data (Fresh Setup)
If you are running on a fresh machine without an existing `data/eventpulse.db`, initialize the database with the bundled sample articles (or live RSS feeds) and cluster them:

```bash
# Option A: Load bundled offline sample articles
python main.py ingest --sample

# Option B: Fetch live articles from the 23 configured RSS feeds
python main.py ingest

# Cluster ingested articles into events
python main.py cluster-events

# Check database counts
python main.py stats
```

### 6. Run the Web Application
Start the FastAPI server:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
*(Add `--reload` during development if you want auto-restart on code changes.)*

Open in your browser:
- **Web App:** http://127.0.0.1:8000
- **Interactive API Docs (Swagger):** http://127.0.0.1:8000/docs

*(Optional) To launch the legacy Streamlit UI instead:*
```bash
streamlit run app.py
```

---

## Running Tests

Run the full automated test suite with:
```bash
pytest tests/ -v
```

---

## Core API Endpoints

| Method & Path | Description |
| :--- | :--- |
| `GET /` | Serves the web UI (`static/index.html`) |
| `GET /api/health` | Service and SQLite health check |
| `GET /api/stats` | Corpus counts (articles, events, sources, embeddings) |
| `GET /api/recommendations` | Personalised EventPulse (`RELEVANCE_FOCUSED`, `BALANCED`, `DISCOVERY_FOCUSED`) or baseline ranking |
| `GET /api/events` | Filterable list of clustered events |
| `GET /api/events/{event_id}/articles` | Chronological article timeline for an event |
| `POST /api/clustering/run` | Clusters newly ingested unassigned articles |
| `GET /api/novelty/evolving-events` | Lists multi-article events for novelty inspection |
| `GET /api/novelty/inspect` | Sentence-level novelty verdicts and matched prior-memory evidence |
| `GET /api/topics` | Emerging topic detection (`online_cluster` or `rolling_window`) |
| `POST /api/interactions` | Saves user feedback (`LIKE`, `READ`, `DISLIKE`) and updates profile vectors |
| `GET /api/benchmarks/mind` | Archived MIND ranking comparison metrics |
| `GET /api/benchmarks/corpus-novelty` | Archived 50-event corpus novelty breakdown |

---

## Evaluation Highlights

- **Recommendation & Redundancy Reduction (MIND, N = 2,000 sessions):** Compared against a content-based baseline (`Redundancy@5 = 0.3135`), EventPulse reduces slate redundancy by **18.4%** in `BALANCED` mode (`0.2557`, `p < 0.0005`) and **38.1%** in `DISCOVERY_FOCUSED` mode (`0.1941`), while increasing Intra-List Diversity (`ILD@5`).
- **Sentence-Level Novelty (TAP-DLND 1.0, 65 held-out events):** Chronological sentence-level event memory achieves **0.8855 ROC-AUC**, outperforming whole-document dense (`0.8072`) and lexical (`0.8143`) baselines.
- **Event Clustering (WCEP-100 & UA-RU News):** Evaluated with `B-Cubed F1`, `Pairwise F1`, `ARI`, and `NMI` against pure dense and graph clustering baselines.
- **Streaming Topic Emergence (Synthetic Fire Drill v3):** Evaluated alongside `DBSTREAM` and `DenStream` baselines on streaming news corpora.
