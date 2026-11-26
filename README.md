# OpenK Data Studio (v2.0)

**OpenK** is an interactive data profiling, fuzzy clustering, and reproducible data wrangling studio — originally built as a thesis project inspired by OpenRefine, now modernized with a pure-Python clustering engine, Union-Find transitive closure, FastAPI REST backend, interactive Data Studio UI, and an automated Continuous Improvement benchmark loop.

---

## Key Capabilities

- **Multi-Format Ingestion & Export**: Load and export `CSV`, `TSV`, `JSON`, `Excel (.xlsx)`, and `Parquet` (load). Includes 1-click loading of the repository's sample datasets in `data_set/`.
- **Data Quality Profiler**: Computes a **0–100 Health Score**, missingness %, uniqueness %, leading/trailing whitespace issues, case variants, exact duplicate rows, numeric summaries, and actionable cleaning suggestions.
- **13 Fuzzy Clustering & Key-Collision Methods**:
  - **Key Collision ($O(N)$)**: `fingerprint`, `ngram_fingerprint`, `metaphone`
  - **Similarity Metrics**: `jaro`, `jaro_winkler`, `levenshtein_sim`, `damerau_sim`, `token_jaccard`, `token_cosine`, `num_sim`
  - **Edit-Distance / KNN**: `levenshtein`, `damerau`, `hamming`
  - **Exact Transitive Closure**: Powered by Disjoint-Set (`UnionFind`) with path compression and length-ratio pruning.
  - **Auto-Recommend**: Analyzes column value distributions to recommend the best clustering method and threshold automatically.
- **Interactive Merge, Undo/Redo & Reproducible Recipes**:
  - Merge clustered variants into canonical values directly in the UI or via API.
  - Full **Undo / Redo** stack.
  - Export and replay **JSON Cleaning Recipes** (`GET/POST /api/sessions/{id}/recipe`) for 100% reproducible pipelines.
- **Continuous Improvement Loop**:
  - Built-in Precision / Recall / F1 / Runtime benchmark suite (`openk benchmark --record`) logged to `benchmarks/history.jsonl` and surfaced in the UI.
  - Automated `pytest` suite and GitHub Actions CI (`.github/workflows/ci.yml`).

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -e ".[dev]"
# or:
pip install -r requirements.txt
```

### 2. Launch OpenK Data Studio
```bash
python -m openk.cli serve --host 127.0.0.1 --port 8000
```
Then open **http://127.0.0.1:8000** in your browser (or **http://127.0.0.1:8000/docs** for interactive OpenAPI/Swagger docs).

### 3. CLI Usage
```bash
# Profile any dataset from the terminal
python -m openk.cli profile data_set/airlines_final.csv

# Run and record the continuous-improvement benchmark suite
python -m openk.cli benchmark --record
```

### 4. Run Tests
```bash
pytest -v
```

---

## Project Structure

```text
OpenK/
├── openk/
│   ├── core/
│   │   ├── metrics.py       # Pure-Python string, phonetic, token, numeric metrics
│   │   ├── clustering.py    # Union-Find clustering engine & auto-recommender
│   │   ├── profiler.py      # Dataset health score & column quality profiler
│   │   └── engine.py        # DatasetSession with Undo/Redo, transforms, recipes
│   ├── api/
│   │   └── app.py           # FastAPI REST API & session manager
│   ├── web/
│   │   └── templates/
│   │       └── index.html   # Interactive Data Studio UI
│   ├── eval/
│   │   └── benchmark.py     # Precision/Recall/F1/Runtime benchmark tracker
│   └── cli.py               # CLI entrypoint (serve, profile, benchmark)
├── tests/                   # Unit, integration, and benchmark regression tests
├── docs/                    # Architecture, Algorithms, and Continuous Improvement docs
├── benchmarks/              # Recorded benchmark history (history.jsonl)
├── data_set/                # Thesis sample datasets
├── main/                    # Original thesis Flask prototype (preserved for reference)
└── .github/workflows/ci.yml # Automated CI pipeline
```

---

## Documentation

- [Architecture & Thesis Modernization](docs/ARCHITECTURE.md)
- [Clustering & Similarity Algorithms](docs/ALGORITHMS.md)
- [Continuous Improvement System](docs/CONTINUOUS_IMPROVEMENT.md)
