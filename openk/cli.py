"""Command-line interface for OpenK Data Studio."""

from __future__ import annotations

import argparse
import json
import sys

import uvicorn

from openk.core.engine import load_dataframe_from_path
from openk.core.profiler import profile_dataframe
from openk.eval.benchmark import run_benchmark_suite


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openk", description="OpenK Data Studio CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    serve_p = sub.add_parser("serve", help="Start the OpenK web UI and REST API")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true")

    prof_p = sub.add_parser("profile", help="Profile a dataset file and print JSON report")
    prof_p.add_argument("file", help="Path to CSV, TSV, JSON, XLSX, or Parquet file")

    bench_p = sub.add_parser("benchmark", help="Run the continuous-improvement benchmark suite")
    bench_p.add_argument("--record", action="store_true", help="Append results to benchmarks/history.jsonl")

    args = parser.parse_args(argv)

    if args.command == "serve":
        uvicorn.run("openk.api.app:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    if args.command == "profile":
        df = load_dataframe_from_path(args.file)
        print(json.dumps(profile_dataframe(df).to_dict(), indent=2))
        return 0

    if args.command == "benchmark":
        summary = run_benchmark_suite(record=args.record)
        print(json.dumps(summary, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
