"""Regression tests for the continuous-improvement benchmark suite."""

from openk.eval.benchmark import run_benchmark_suite


def test_benchmark_suite_meets_quality_floor():
    summary = run_benchmark_suite(record=False)
    assert summary["best_f1"] >= 0.70
    assert len(summary["results"]) >= 8
