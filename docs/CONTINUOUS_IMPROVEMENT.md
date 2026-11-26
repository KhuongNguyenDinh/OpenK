# Continuous Improvement System

To ensure OpenK continuously improves rather than stagnating after a single release, the repository includes an automated **Measure $\rightarrow$ Benchmark $\rightarrow$ Regress-Check $\rightarrow$ Iterate** loop.

---

## 1. How the Loop Works

1. **Automated Unit & Integration Tests (`tests/`)**:
   - Every metric, Union-Find transitive closure, profiler calculation, Undo/Redo operation, recipe replay, and FastAPI endpoint is verified by `pytest`.
2. **Ground-Truth Benchmark Suite (`openk/eval/benchmark.py`)**:
   - Evaluates all clustering methods against a standardized ground-truth dataset (`SYNTHETIC_GROUND_TRUTH`) covering casing, punctuation, token swaps, Unicode accents, and typos.
   - Computes **Pairwise Precision**, **Pairwise Recall**, **F1-Score**, and **Runtime (ms)**.
3. **Append-Only History (`benchmarks/history.jsonl`)**:
   - Running `python -m openk.cli benchmark --record` appends a timestamped JSON summary to `benchmarks/history.jsonl`.
   - The **Continuous Improvement** tab in the web UI (`GET /api/benchmarks`) visualizes the latest run so improvements (or regressions) are immediately visible.
4. **CI Quality Gate (`.github/workflows/ci.yml`)**:
   - Runs `pytest -v` (including `tests/test_benchmark.py`, which enforces a minimum F1 quality floor) on every push and pull request.

---

## 2. Baseline Benchmark Results (v2.0)

| Method | Threshold | Precision | Recall | F1 | Runtime (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `metaphone` | `0.0` | `1.0000` | `0.8571` | **0.9231** | `0.353` |
| `token_jaccard` | `0.60` | `1.0000` | `0.8571` | **0.9231** | `1.591` |
| `token_cosine` | `0.65` | `1.0000` | `0.8571` | **0.9231** | `2.300` |
| `fingerprint` | `0.0` | `1.0000` | `0.7959` | `0.8864` | `0.172` |
| `jaro` | `0.85` | `1.0000` | `0.6122` | `0.7595` | `2.324` |
| `jaro_winkler` | `0.88` | `1.0000` | `0.6122` | `0.7595` | `2.545` |

---

## 3. Roadmap for Future Iterations

1. **Blocking / Locality-Sensitive Hashing (LSH)**: Add $n$-gram inverted-index blocking before pairwise similarity so columns with 100,000+ unique values cluster in sub-second time.
2. **Hybrid Ensemble Clustering**: Combine `fingerprint` + `metaphone` + `jaro_winkler` into a weighted ensemble scorer and benchmark it against the single-method baseline in `benchmarks/history.jsonl`.
3. **Optional Semantic Embedding Clustering**: Add an optional plugin using sentence embeddings for semantic synonyms (e.g., `"NYC"` $\leftrightarrow$ `"New York City"`).
4. **Persistent Storage**: Back `DatasetSession` with DuckDB or SQLite for multi-gigabyte datasets and persistent projects across server restarts.
