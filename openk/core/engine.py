"""Stateful data wrangling session with Undo/Redo, facets, and reproducible recipes."""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from openk.core.clustering import cluster_values, recommend_clustering
from openk.core.profiler import DatasetProfile, profile_dataframe


@dataclass
class OperationRecord:
    """A single reproducible transformation step."""

    op_type: str
    params: dict[str, Any]
    summary: str
    rows_affected: int = 0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataframe_from_bytes(content: bytes, filename: str) -> pd.DataFrame:
    """Load a DataFrame from raw file bytes based on extension."""
    ext = Path(filename).suffix.lower()
    buf = io.BytesIO(content)
    if ext == ".csv":
        return pd.read_csv(buf)
    if ext == ".tsv":
        return pd.read_csv(buf, sep="\t")
    if ext == ".json":
        text = content.decode("utf-8")
        try:
            return pd.read_json(io.StringIO(text))
        except ValueError:
            parsed = json.loads(text)
            return pd.json_normalize(parsed)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(buf)
    if ext == ".parquet":
        return pd.read_parquet(buf)
    raise ValueError(f"Unsupported file format: '{ext}'")


def load_dataframe_from_path(path: str | Path) -> pd.DataFrame:
    """Load a DataFrame from a file path on disk."""
    p = Path(path)
    return load_dataframe_from_bytes(p.read_bytes(), p.name)


class DatasetSession:
    """Manages an active dataset with Undo/Redo history and reproducible recipes."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset") -> None:
        self.name = name
        self._df = df.copy()
        self._undo_stack: list[tuple[pd.DataFrame, OperationRecord]] = []
        self._redo_stack: list[tuple[pd.DataFrame, OperationRecord]] = []
        self._history: list[OperationRecord] = []

    @property
    def df(self) -> pd.DataFrame:
        return self._df

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    def _commit(self, new_df: pd.DataFrame, record: OperationRecord) -> OperationRecord:
        self._undo_stack.append((self._df.copy(), record))
        self._redo_stack.clear()
        self._df = new_df
        self._history.append(record)
        return record

    def undo(self) -> bool:
        if not self._undo_stack:
            return False
        prev_df, record = self._undo_stack.pop()
        self._redo_stack.append((self._df.copy(), record))
        self._df = prev_df
        if self._history:
            self._history.pop()
        return True

    def redo(self) -> bool:
        if not self._redo_stack:
            return False
        next_df, record = self._redo_stack.pop()
        self._undo_stack.append((self._df.copy(), record))
        self._df = next_df
        self._history.append(record)
        return True

    def profile(self) -> DatasetProfile:
        return profile_dataframe(self._df)

    def cluster(
        self,
        column: str,
        method: str = "fingerprint",
        threshold: float | None = None,
        min_cluster_size: int = 2,
    ) -> list[dict[str, Any]]:
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found")
        groups = cluster_values(
            self._df[column].tolist(),
            method=method,
            threshold=threshold,
            min_cluster_size=min_cluster_size,
        )
        return [g.to_dict() for g in groups]

    def recommend(self, column: str) -> dict[str, Any]:
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found")
        return recommend_clustering(self._df[column].tolist())

    def text_facet(self, column: str, limit: int = 50) -> list[dict[str, Any]]:
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found")
        series = self._df[column].fillna("<MISSING>").astype(str)
        vc = series.value_counts().head(limit)
        return [{"value": str(k), "count": int(v)} for k, v in vc.items()]

    def apply_cluster_merges(
        self, column: str, merges: list[dict[str, Any]]
    ) -> OperationRecord:
        """Merge clustered values in `column` into their canonical target values."""
        if column not in self._df.columns:
            raise KeyError(f"Column '{column}' not found")

        mapping: dict[str, str] = {}
        for m in merges:
            target = str(m["to_value"])
            for src in m["from_values"]:
                src_str = str(src)
                if src_str != target:
                    mapping[src_str] = target

        new_df = self._df.copy()
        col_str = new_df[column].astype(str)
        mask = col_str.isin(mapping.keys()) & new_df[column].notna()
        rows_affected = int(mask.sum())
        if rows_affected > 0:
            new_df.loc[mask, column] = col_str[mask].map(mapping)

        record = OperationRecord(
            op_type="cluster_merge",
            params={"column": column, "merges": merges},
            summary=f"Merged {len(mapping)} variant(s) in '{column}' ({rows_affected} row(s) updated)",
            rows_affected=rows_affected,
        )
        return self._commit(new_df, record)

    def transform_column(
        self, column: str | None, operation: str, **params: Any
    ) -> OperationRecord:
        """Apply a deterministic transformation to a column or the whole DataFrame."""
        new_df = self._df.copy()
        op = operation.strip().lower()
        rows_affected = 0

        if op == "drop_duplicates":
            before = len(new_df)
            new_df = new_df.drop_duplicates().reset_index(drop=True)
            rows_affected = before - len(new_df)
            summary = f"Removed {rows_affected} duplicate row(s)"
        else:
            if not column or column not in new_df.columns:
                raise KeyError(f"Column '{column}' not found")
            series = new_df[column]
            not_na = series.notna()
            as_str = series[not_na].astype(str)

            if op == "trim":
                transformed = as_str.str.strip()
            elif op == "collapse_whitespace":
                transformed = as_str.str.strip().str.replace(r"\s+", " ", regex=True)
            elif op == "lower":
                transformed = as_str.str.lower()
            elif op == "upper":
                transformed = as_str.str.upper()
            elif op == "title":
                transformed = as_str.str.title()
            elif op == "regex_replace":
                pattern = str(params.get("pattern", ""))
                repl = str(params.get("replacement", ""))
                if not pattern:
                    raise ValueError("regex_replace requires a non-empty 'pattern'")
                re.compile(pattern)
                transformed = as_str.str.replace(pattern, repl, regex=True)
            elif op == "fill_missing":
                fill_val = params.get("value", "")
                missing_mask = series.isna() | (series.astype(str).str.strip() == "")
                rows_affected = int(missing_mask.sum())
                new_df.loc[missing_mask, column] = fill_val
                transformed = None
            else:
                raise ValueError(f"Unknown transformation '{operation}'")

            if transformed is not None:
                changed = transformed != as_str
                rows_affected = int(changed.sum())
                new_df.loc[not_na, column] = transformed

            summary = f"Applied '{op}' on '{column}' ({rows_affected} row(s) updated)"

        record = OperationRecord(
            op_type="transform",
            params={"column": column, "operation": op, **params},
            summary=summary,
            rows_affected=rows_affected,
        )
        return self._commit(new_df, record)

    def get_recipe(self) -> list[dict[str, Any]]:
        """Return the ordered list of operations for reproducible replay."""
        return [r.to_dict() for r in self._history]

    def apply_recipe(self, recipe: list[dict[str, Any]]) -> int:
        """Replay a previously exported cleaning recipe onto the current dataset."""
        applied = 0
        for step in recipe:
            op_type = step.get("op_type")
            params = step.get("params", {})
            if op_type == "cluster_merge":
                self.apply_cluster_merges(params["column"], params["merges"])
                applied += 1
            elif op_type == "transform":
                col = params.get("column")
                op = params["operation"]
                extra = {k: v for k, v in params.items() if k not in {"column", "operation"}}
                self.transform_column(col, op, **extra)
                applied += 1
        return applied

    def export_bytes(self, fmt: str = "csv") -> tuple[bytes, str, str]:
        """Export current DataFrame as (bytes, media_type, file_extension)."""
        fmt = fmt.lower().strip()
        if fmt == "csv":
            return self._df.to_csv(index=False).encode("utf-8"), "text/csv", "csv"
        if fmt == "tsv":
            return self._df.to_csv(index=False, sep="\t").encode("utf-8"), "text/tab-separated-values", "tsv"
        if fmt == "json":
            return (
                self._df.to_json(orient="records", indent=2, force_ascii=False).encode("utf-8"),
                "application/json",
                "json",
            )
        if fmt == "xlsx":
            buf = io.BytesIO()
            self._df.to_excel(buf, index=False)
            return (
                buf.getvalue(),
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "xlsx",
            )
        raise ValueError(f"Unsupported export format '{fmt}'")
