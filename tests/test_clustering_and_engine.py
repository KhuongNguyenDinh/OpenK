"""Unit tests for clustering, profiling, and DatasetSession transformations."""

import pandas as pd
import pytest

from openk.core.clustering import ALL_METHODS, cluster_values, recommend_clustering
from openk.core.engine import DatasetSession
from openk.core.profiler import profile_dataframe


def test_all_clustering_methods_run_without_error():
    sample = [
        "New York",
        "new york",
        "York, New",
        "New-York",
        "San Francisco",
        "San Fransisco",
        "100",
        "101",
    ]
    for method in ALL_METHODS:
        clusters = cluster_values(sample, method=method)
        assert isinstance(clusters, list)


def test_transitive_closure_with_union_find():
    # "abc" ~ "abd" (dist 1), "abd" ~ "acd" (dist 1) => all 3 in one cluster
    clusters = cluster_values(["abc", "abd", "acd"], method="levenshtein", threshold=1)
    assert len(clusters) == 1
    assert {m.value for m in clusters[0].members} == {"abc", "abd", "acd"}


def test_recommend_clustering_modes():
    assert recommend_clustering(["10", "20", "30", "40"])["method"] == "num_sim"
    assert recommend_clustering(["New York City", "City, New York", "San Francisco Bay"])["method"] in {
        "fingerprint",
        "token_jaccard",
    }
    assert recommend_clustering(["apple", "aple", "banana"])["method"] == "jaro_winkler"


def test_profiler_and_session_undo_redo_recipe():
    df = pd.DataFrame(
        {
            "city": ["New York", "new york", " York, New ", "Boston", None],
            "amount": [10, 20, 30, 40, 50],
        }
    )
    prof = profile_dataframe(df)
    assert prof.row_count == 5
    assert prof.column_count == 2
    assert 0 <= prof.health_score <= 100

    session = DatasetSession(df, name="test.csv")
    session.transform_column("city", "trim")
    assert session.df.loc[2, "city"] == "York, New"

    clusters = session.cluster("city", method="fingerprint")
    assert len(clusters) == 1
    merges = [{"from_values": [m["value"] for m in clusters[0]["members"]], "to_value": "New York"}]
    session.apply_cluster_merges("city", merges)
    assert (session.df["city"] == "New York").sum() == 3

    recipe = session.get_recipe()
    assert len(recipe) == 2

    # Undo & Redo
    assert session.undo() is True
    assert (session.df["city"] == "New York").sum() == 1
    assert session.redo() is True
    assert (session.df["city"] == "New York").sum() == 3

    # Replay recipe on fresh session
    fresh = DatasetSession(df.copy(), name="fresh.csv")
    applied = fresh.apply_recipe(recipe)
    assert applied == 2
    assert (fresh.df["city"] == "New York").sum() == 3

    # Export formats
    for fmt in ("csv", "tsv", "json", "xlsx"):
        raw, _, ext = session.export_bytes(fmt)
        assert len(raw) > 0
        assert ext == fmt
