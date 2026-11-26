"""Fast, deterministic value clustering with Union-Find transitive closure."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Callable, Iterable

from openk.core.metrics import (
    damerau_levenshtein_distance,
    damerau_levenshtein_similarity,
    fingerprint_key,
    hamming_distance,
    is_numeric_string,
    jaro_similarity,
    jaro_winkler_similarity,
    levenshtein_distance,
    levenshtein_similarity,
    metaphone_key,
    ngram_fingerprint_key,
    numeric_similarity,
    token_cosine_similarity,
    token_jaccard_similarity,
)


@dataclass(frozen=True)
class ClusterMember:
    """A distinct value inside a cluster along with its row count."""

    value: str
    count: int


@dataclass
class ClusterGroup:
    """A group of similar values with a suggested canonical replacement."""

    cluster_id: int
    canonical: str
    members: list[ClusterMember]
    total_rows: int
    method: str

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "canonical": self.canonical,
            "total_rows": self.total_rows,
            "method": self.method,
            "members": [asdict(m) for m in self.members],
        }


class UnionFind:
    """Disjoint-set data structure with path compression and union by rank."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


KEY_COLLISION_METHODS = {"fingerprint", "ngram_fingerprint", "n_gram", "metaphone"}
SIMILARITY_METHODS = {
    "jaro",
    "jaro_winkler",
    "levenshtein_sim",
    "damerau_sim",
    "token_jaccard",
    "token_cosine",
    "num_sim",
}
DISTANCE_METHODS = {"levenshtein", "damerau", "hamming"}
ALL_METHODS = sorted(KEY_COLLISION_METHODS | SIMILARITY_METHODS | DISTANCE_METHODS)

DEFAULT_PARAMETERS: dict[str, float] = {
    "fingerprint": 0.0,
    "ngram_fingerprint": 2.0,
    "n_gram": 2.0,
    "metaphone": 0.0,
    "jaro": 0.85,
    "jaro_winkler": 0.88,
    "levenshtein_sim": 0.80,
    "damerau_sim": 0.80,
    "token_jaccard": 0.65,
    "token_cosine": 0.70,
    "num_sim": 0.95,
    "levenshtein": 2.0,
    "damerau": 2.0,
    "hamming": 2.0,
}


def _choose_canonical(members: list[ClusterMember]) -> str:
    """Pick the best canonical value: highest frequency, then title/cleanest casing, then shortest."""
    return max(
        members,
        key=lambda m: (
            m.count,
            1 if m.value == m.value.strip() else 0,
            1 if not m.value.islower() and not m.value.isupper() else 0,
            -len(m.value),
        ),
    ).value


def _build_groups(
    buckets: Iterable[list[str]],
    counts: Counter[str],
    method: str,
    min_cluster_size: int,
) -> list[ClusterGroup]:
    groups: list[ClusterGroup] = []
    for raw_members in buckets:
        if len(raw_members) < min_cluster_size:
            continue
        members = sorted(
            [ClusterMember(value=v, count=counts[v]) for v in raw_members],
            key=lambda m: (-m.count, m.value),
        )
        canonical = _choose_canonical(members)
        groups.append(
            ClusterGroup(
                cluster_id=0,
                canonical=canonical,
                members=members,
                total_rows=sum(m.count for m in members),
                method=method,
            )
        )
    groups.sort(key=lambda g: (-g.total_rows, -len(g.members), g.canonical))
    for idx, g in enumerate(groups, start=1):
        g.cluster_id = idx
    return groups


def cluster_values(
    values: Iterable[object],
    method: str = "fingerprint",
    threshold: float | None = None,
    min_cluster_size: int = 2,
    max_unique: int = 2500,
) -> list[ClusterGroup]:
    """
    Cluster non-empty values in a column using key-collision, similarity, or distance methods.

    Parameters
    ----------
    values:
        Raw column values (None/NaN/empty strings are ignored).
    method:
        One of ALL_METHODS.
    threshold:
        Similarity threshold in [0, 1] for similarity methods, edit radius >= 0 for distance
        methods, or n-gram size >= 1 for ngram_fingerprint / n_gram. Uses sensible defaults if None.
    min_cluster_size:
        Minimum number of distinct values required to form a cluster (default 2).
    max_unique:
        Safety cap on unique values for O(N^2) pairwise methods.
    """
    method = method.strip().lower()
    if method not in ALL_METHODS:
        raise ValueError(f"Unknown clustering method '{method}'. Valid: {ALL_METHODS}")

    counts: Counter[str] = Counter()
    for v in values:
        if v is None:
            continue
        s = str(v)
        if not s or s.lower() == "nan":
            continue
        counts[s] += 1

    unique_vals = sorted(counts.keys())
    if len(unique_vals) < min_cluster_size:
        return []

    param = DEFAULT_PARAMETERS[method] if threshold is None else float(threshold)

    # 1. O(N) Key-collision methods
    if method in KEY_COLLISION_METHODS:
        key_fn: Callable[[str], str]
        if method == "fingerprint":
            key_fn = fingerprint_key
        elif method in ("ngram_fingerprint", "n_gram"):
            n = max(1, int(round(param)))
            key_fn = lambda s, _n=n: ngram_fingerprint_key(s, _n)
        else:
            key_fn = metaphone_key

        buckets_map: dict[str, list[str]] = defaultdict(list)
        for val in unique_vals:
            k = key_fn(val)
            if k:
                buckets_map[k].append(val)
        return _build_groups(buckets_map.values(), counts, method, min_cluster_size)

    # 2. Pairwise similarity / distance methods with Union-Find
    candidates = unique_vals[:max_unique]
    n = len(candidates)
    uf = UnionFind(n)
    lowered = [c.lower() for c in candidates]
    lengths = [len(c) for c in candidates]

    if method in SIMILARITY_METHODS:
        sim_fn: Callable[[str, str], float]
        use_lower = method in {"jaro", "jaro_winkler", "levenshtein_sim", "damerau_sim"}
        if method == "jaro":
            sim_fn = jaro_similarity
        elif method == "jaro_winkler":
            sim_fn = jaro_winkler_similarity
        elif method == "levenshtein_sim":
            sim_fn = levenshtein_similarity
        elif method == "damerau_sim":
            sim_fn = damerau_levenshtein_similarity
        elif method == "token_jaccard":
            sim_fn = token_jaccard_similarity
        elif method == "token_cosine":
            sim_fn = token_cosine_similarity
        else:
            sim_fn = numeric_similarity

        for i in range(n):
            si = lowered[i] if use_lower else candidates[i]
            li = lengths[i]
            for j in range(i + 1, n):
                if uf.find(i) == uf.find(j):
                    continue
                lj = lengths[j]
                # Exact length-ratio upper bound for edit-based similarities
                if method in {"levenshtein_sim", "damerau_sim"} and max(li, lj) > 0:
                    if min(li, lj) / max(li, lj) < param:
                        continue
                sj = lowered[j] if use_lower else candidates[j]
                if sim_fn(si, sj) >= param:
                    uf.union(i, j)
    else:
        radius = int(round(param))
        dist_fn: Callable[[str, str], int]
        if method == "levenshtein":
            dist_fn = levenshtein_distance
        elif method == "damerau":
            dist_fn = damerau_levenshtein_distance
        else:
            dist_fn = hamming_distance

        for i in range(n):
            si = lowered[i]
            li = lengths[i]
            for j in range(i + 1, n):
                if uf.find(i) == uf.find(j):
                    continue
                if abs(li - lengths[j]) > radius:
                    continue
                if dist_fn(si, lowered[j]) <= radius:
                    uf.union(i, j)

    comp_map: dict[int, list[str]] = defaultdict(list)
    for idx, val in enumerate(candidates):
        comp_map[uf.find(idx)].append(val)

    return _build_groups(comp_map.values(), counts, method, min_cluster_size)


def recommend_clustering(values: Iterable[object]) -> dict:
    """Analyze a column's values and recommend the best clustering method and threshold."""
    sample = [str(v) for v in values if v is not None and str(v).strip() and str(v).lower() != "nan"]
    if not sample:
        return {"method": "fingerprint", "threshold": 0.0, "reason": "Default key-collision"}

    unique = list(set(sample[:2000]))
    numeric_ratio = sum(1 for s in unique if is_numeric_string(s)) / len(unique)
    if numeric_ratio >= 0.85:
        return {
            "method": "num_sim",
            "threshold": 0.95,
            "reason": f"{numeric_ratio:.0%} numeric values detected",
        }

    avg_tokens = sum(len(s.split()) for s in unique) / len(unique)
    if avg_tokens >= 2.2:
        fp_clusters = cluster_values(sample, method="fingerprint")
        if fp_clusters:
            return {
                "method": "fingerprint",
                "threshold": 0.0,
                "reason": f"Multi-word text ({avg_tokens:.1f} tokens/val) with {len(fp_clusters)} exact key-collision clusters",
            }
        return {
            "method": "token_jaccard",
            "threshold": 0.65,
            "reason": f"Multi-word text ({avg_tokens:.1f} tokens/val); token Jaccard catches word-order/partial matches",
        }

    return {
        "method": "jaro_winkler",
        "threshold": 0.88,
        "reason": "Short/single-token text; Jaro-Winkler excels at typos and prefix variations",
    }
