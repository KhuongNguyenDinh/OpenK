"""Automated clustering benchmark & continuous-improvement tracker."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

from openk.core.clustering import cluster_values

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
BENCHMARK_HISTORY_PATH = ROOT_DIR / "benchmarks" / "history.jsonl"

# Ground-truth entity groups covering casing, punctuation, token swaps, accents, and typos
SYNTHETIC_GROUND_TRUTH: list[list[str]] = [
    ["New York", "new york", "New  York", "York, New", "New-York"],
    ["San Francisco", "san francisco", "Francisco, San", "San Fransisco"],
    ["Los Angeles", "los angeles", "Angeles, Los", "Los  Angeles"],
    ["José García", "Jose Garcia", "jose garcia", "Garcia, Jose"],
    ["McDonald's", "McDonalds", "mcdonalds", "McDonald's Corp."],
    ["Apple Inc.", "apple inc", "Inc., Apple", "Apple Inc"],
    ["United Airlines", "united airlines", "United  Airlines", "Airlines, United"],
    ["Coffee Bean & Tea Leaf", "Coffee Bean and Tea Leaf", "coffee bean tea leaf"],
]


def _pairs_from_groups(groups: list[list[str]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for g in groups:
        for a, b in combinations(sorted(set(g)), 2):
            pairs.add((a, b))
    return pairs


def evaluate_clustering(
    ground_truth: list[list[str]],
    method: str,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute pairwise Precision, Recall, F1, and runtime for a clustering method."""
    all_values = [v for group in ground_truth for v in group]
    true_pairs = _pairs_from_groups(ground_truth)

    t0 = time.perf_counter()
    clusters = cluster_values(all_values, method=method, threshold=threshold)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)

    pred_groups = [[m.value for m in c.members] for c in clusters]
    pred_pairs = _pairs_from_groups(pred_groups)

    tp = len(true_pairs & pred_pairs)
    fp = len(pred_pairs - true_pairs)
    fn = len(true_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "method": method,
        "threshold": threshold if threshold is not None else "default",
        "clusters_found": len(clusters),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "runtime_ms": elapsed_ms,
    }


def run_benchmark_suite(record: bool = False) -> dict[str, Any]:
    """Sweep key-collision, similarity, and edit-distance methods and optionally record history."""
    configs: list[tuple[str, float | None]] = [
        ("fingerprint", 0.0),
        ("ngram_fingerprint", 2.0),
        ("metaphone", 0.0),
        ("jaro", 0.85),
        ("jaro_winkler", 0.88),
        ("levenshtein_sim", 0.78),
        ("damerau_sim", 0.78),
        ("token_jaccard", 0.60),
        ("token_cosine", 0.65),
        ("levenshtein", 2.0),
        ("damerau", 2.0),
    ]

    results: list[dict[str, Any]] = []
    for method, thr in configs:
        metrics = evaluate_clustering(SYNTHETIC_GROUND_TRUTH, method=method, threshold=thr)
        metrics["dataset"] = "synthetic_entities"
        results.append(metrics)

    best = max(results, key=lambda r: (r["f1"], -r["runtime_ms"]))
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_method": best["method"],
        "best_f1": best["f1"],
        "results": results,
    }

    if record:
        BENCHMARK_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with BENCHMARK_HISTORY_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(summary) + "\n")

    return summary


if __name__ == "__main__":
    out = run_benchmark_suite(record=True)
    print(json.dumps(out, indent=2))
