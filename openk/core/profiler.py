"""Dataset & column quality profiling with health scoring and actionable suggestions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from openk.core.clustering import cluster_values


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    non_null_count: int
    missing_count: int
    missing_pct: float
    unique_count: int
    uniqueness_pct: float
    whitespace_issues: int
    case_variants: int
    potential_clusters: int
    top_values: list[dict[str, Any]]
    numeric_summary: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetProfile:
    row_count: int
    column_count: int
    duplicate_rows: int
    health_score: float
    columns: list[ColumnProfile]
    suggestions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "health_score": self.health_score,
            "columns": [c.to_dict() for c in self.columns],
            "suggestions": self.suggestions,
        }


def profile_dataframe(df: pd.DataFrame) -> DatasetProfile:
    """Profile a DataFrame, returning quality metrics, health score (0-100), and suggestions."""
    row_count = len(df)
    col_count = len(df.columns)
    duplicate_rows = int(df.duplicated().sum()) if row_count > 0 else 0

    col_profiles: list[ColumnProfile] = []
    suggestions: list[str] = []

    if duplicate_rows > 0:
        suggestions.append(
            f"Found {duplicate_rows} exact duplicate row(s). Consider running 'drop_duplicates'."
        )

    total_cells = max(1, row_count * max(1, col_count))
    total_missing = 0
    total_whitespace = 0
    total_case_variants = 0

    for col in df.columns:
        series = df[col]
        missing_mask = series.isna() | (series.astype(str).str.strip() == "")
        missing_count = int(missing_mask.sum())
        non_null_count = row_count - missing_count
        missing_pct = round((missing_count / row_count) * 100, 2) if row_count else 0.0
        total_missing += missing_count

        non_empty = series[~missing_mask].astype(str)
        unique_count = int(non_empty.nunique())
        uniqueness_pct = round((unique_count / non_null_count) * 100, 2) if non_null_count else 0.0

        whitespace_issues = int((non_empty != non_empty.str.strip()).sum()) if len(non_empty) else 0
        total_whitespace += whitespace_issues

        lower_unique = int(non_empty.str.lower().nunique()) if len(non_empty) else 0
        case_variants = max(0, unique_count - lower_unique)
        total_case_variants += case_variants

        potential_clusters = 0
        if pd.api.types.is_string_dtype(series) or series.dtype == object:
            if 2 <= unique_count <= 2000:
                potential_clusters = len(cluster_values(non_empty.tolist(), method="fingerprint"))

        top_counts = non_empty.value_counts().head(5)
        top_values = [{"value": str(k), "count": int(v)} for k, v in top_counts.items()]

        numeric_summary = None
        if pd.api.types.is_numeric_dtype(series) and non_null_count > 0:
            numeric_summary = {
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std(ddof=0)), 4),
            }

        if whitespace_issues > 0:
            suggestions.append(
                f"Column '{col}' has {whitespace_issues} value(s) with leading/trailing whitespace."
            )
        if potential_clusters > 0:
            suggestions.append(
                f"Column '{col}' has {potential_clusters} fingerprint cluster(s) ready to merge."
            )
        elif case_variants > 0:
            suggestions.append(
                f"Column '{col}' has {case_variants} case-only variant(s)."
            )

        col_profiles.append(
            ColumnProfile(
                name=str(col),
                dtype=str(series.dtype),
                non_null_count=non_null_count,
                missing_count=missing_count,
                missing_pct=missing_pct,
                unique_count=unique_count,
                uniqueness_pct=uniqueness_pct,
                whitespace_issues=whitespace_issues,
                case_variants=case_variants,
                potential_clusters=potential_clusters,
                top_values=top_values,
                numeric_summary=numeric_summary,
            )
        )

    completeness = 1.0 - (total_missing / total_cells)
    dup_penalty = (duplicate_rows / max(1, row_count)) * 20.0
    ws_penalty = min(10.0, (total_whitespace / total_cells) * 50.0)
    case_penalty = min(10.0, total_case_variants * 0.5)
    health_score = round(max(0.0, min(100.0, completeness * 100.0 - dup_penalty - ws_penalty - case_penalty)), 1)

    return DatasetProfile(
        row_count=row_count,
        column_count=col_count,
        duplicate_rows=duplicate_rows,
        health_score=health_score,
        columns=col_profiles,
        suggestions=suggestions[:15],
    )
