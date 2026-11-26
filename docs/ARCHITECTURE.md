# OpenK Architecture & Modernization Notes

## 1. Overview

OpenK v2.0 separates concerns into four clean layers:

1. **Core Engine (`openk/core/`)** — Stateless metrics (`metrics.py`), clustering (`clustering.py`), profiling (`profiler.py`), and stateful session management (`engine.py`).
2. **API Layer (`openk/api/app.py`)** — FastAPI application exposing REST endpoints, OpenAPI schema (`/docs`), and session-scoped state.
3. **Presentation Layer (`openk/web/templates/index.html`)** — Self-contained interactive Data Studio UI with Data Grid, Quality Profile, Clustering Studio, and Benchmark History.
4. **Evaluation Layer (`openk/eval/benchmark.py`)** — Ground-truth benchmark suite measuring Precision, Recall, F1, and runtime over time.

---

## 2. Improvements Over Original Thesis Prototype (`main/`)

The original thesis codebase in `main/` is preserved for historical reference. The following bugs, architectural limits, and security issues were identified and resolved in `openk/`:

| Area | Original Issue (`main/`) | Resolution (`openk/`) |
| :--- | :--- | :--- |
| **State Management** | Global mutable lists (`headings`, `values`, `group_list`) in `main/app.py` shared across all requests. | Session-scoped `DatasetSession` instances keyed by `session_id` with full Undo/Redo stacks. |
| **Clustering Correctness** | `jaro_winkler` called `jaro_sim()`; `levenshtein_sim` / `damerau_sim` passed two args to `str()`; `hamming` referenced `j` before loop; `com_check` mutated list while iterating. | Clean, unit-tested implementations in `openk/core/metrics.py` and exact `UnionFind` transitive closure in `openk/core/clustering.py`. |
| **Column Indexing** | `range(0, len(headings_data) - 1)` skipped the last column when building `group_list`. | All columns are profiled and clusterable by column name. |
| **Excel Support** | `pd.xlsx(...)` (non-existent pandas method). | Proper `pd.read_excel(...)` and `df.to_excel(...)` via `openpyxl`. |
| **Cluster Merging** | `clustered.html` only printed raw cluster lists without applying merges. | Interactive cluster selection, editable canonical names, and `POST /api/sessions/{id}/merge` that updates the dataset and logs a reproducible recipe step. |
| **Dependencies** | `requirements.txt` listed Python 2 `urllib2_file` and omitted `jellyfish`, `strsimpy`, `unidecode`, `fingerprints`. | Zero-external-C-dependency pure-Python metrics; standard modern stack (`fastapi`, `pandas`, `pydantic`). |
| **Security** | Hardcoded Azure Cosmos DB credentials in `main/retrieve.py`. | Replaced with `os.environ.get("AZURE_COSMOS_ENDPOINT")` and `os.environ.get("AZURE_COSMOS_KEY")`. |
