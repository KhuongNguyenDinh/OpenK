"""Pure-Python string, phonetic, token, and numeric similarity & key-collision metrics."""

from __future__ import annotations

import math
import re
import string
import unicodedata
from collections import Counter

_PUNCT_RE = re.compile(f"[{re.escape(string.punctuation)}]")
_WHITESPACE_RE = re.compile(r"\s+")
_NUMBER_RE = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def ascii_fold(text: str) -> str:
    """Fold Unicode characters to their closest ASCII representation."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


def normalize_text(text: str) -> str:
    """Trim, lowercase, remove punctuation, collapse whitespace, and ASCII-fold."""
    cleaned = ascii_fold(text.strip().lower())
    cleaned = _PUNCT_RE.sub(" ", cleaned)
    return _WHITESPACE_RE.sub(" ", cleaned).strip()


def fingerprint_key(text: str) -> str:
    """OpenRefine-style key-collision fingerprint: sorted unique normalized tokens."""
    norm = normalize_text(text)
    if not norm:
        return ""
    tokens = sorted(set(norm.split()))
    return " ".join(tokens)


def ngram_fingerprint_key(text: str, n: int = 2) -> str:
    """OpenRefine-style n-gram fingerprint: sorted unique character n-grams."""
    if n < 1:
        raise ValueError("n must be >= 1")
    norm = _WHITESPACE_RE.sub("", normalize_text(text))
    if not norm:
        return ""
    if len(norm) <= n:
        return norm
    grams = sorted({norm[i : i + n] for i in range(len(norm) - n + 1)})
    return "".join(grams)


def metaphone_key(text: str) -> str:
    """Lightweight phonetic key (simplified Soundex/Metaphone hybrid) per token."""
    norm = normalize_text(text)
    if not norm:
        return ""

    def _encode_word(word: str) -> str:
        word = re.sub(r"[^a-z]", "", word)
        if not word:
            return ""
        # Collapse consecutive duplicate letters
        deduped = [word[0]]
        for ch in word[1:]:
            if ch != deduped[-1]:
                deduped.append(ch)
        w = "".join(deduped)
        # Common phonetic normalizations
        w = w.replace("ph", "f").replace("kn", "n").replace("wr", "r")
        w = re.sub(r"c([eiy])", r"s\1", w)
        w = w.replace("c", "k").replace("q", "k").replace("x", "ks")
        w = re.sub(r"g([eiy])", r"j\1", w)
        # Keep first char, drop internal vowels (a,e,i,o,u,y) and h/w
        first = w[0]
        rest = re.sub(r"[aeiouyhw]", "", w[1:])
        return (first + rest)[:6]

    encoded = [_encode_word(tok) for tok in norm.split()]
    return " ".join(sorted(tok for tok in encoded if tok))


def levenshtein_distance(a: str, b: str) -> int:
    """Compute exact Levenshtein edit distance between two strings."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity in [0.0, 1.0]."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(a, b) / max_len)


def damerau_levenshtein_distance(a: str, b: str) -> int:
    """Compute optimal string alignment (restricted Damerau-Levenshtein) distance."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    d = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        d[i][0] = i
    for j in range(lb + 1):
        d[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            val = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + cost,
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                val = min(val, d[i - 2][j - 2] + cost)
            d[i][j] = val
    return d[la][lb]


def damerau_levenshtein_similarity(a: str, b: str) -> float:
    """Normalized Damerau-Levenshtein similarity in [0.0, 1.0]."""
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - (damerau_levenshtein_distance(a, b) / max_len)


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two strings (extra trailing chars count as differences)."""
    common = min(len(a), len(b))
    diffs = sum(1 for i in range(common) if a[i] != b[i])
    return diffs + abs(len(a) - len(b))


def jaro_similarity(a: str, b: str) -> float:
    """Compute Jaro similarity in [0.0, 1.0]."""
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    match_dist = max(la, lb) // 2 - 1
    if match_dist < 0:
        match_dist = 0

    a_matches = [False] * la
    b_matches = [False] * lb
    matches = 0

    for i in range(la):
        start = max(0, i - match_dist)
        end = min(i + match_dist + 1, lb)
        for j in range(start, end):
            if b_matches[j] or a[i] != b[j]:
                continue
            a_matches[i] = True
            b_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0.0

    transpositions = 0
    k = 0
    for i in range(la):
        if not a_matches[i]:
            continue
        while not b_matches[k]:
            k += 1
        if a[i] != b[k]:
            transpositions += 1
        k += 1

    return (
        (matches / la) + (matches / lb) + ((matches - transpositions / 2.0) / matches)
    ) / 3.0


def jaro_winkler_similarity(a: str, b: str, prefix_weight: float = 0.1) -> float:
    """Compute Jaro-Winkler similarity in [0.0, 1.0]."""
    jaro = jaro_similarity(a, b)
    prefix = 0
    for ca, cb in zip(a[:4], b[:4]):
        if ca == cb:
            prefix += 1
        else:
            break
    return jaro + prefix * prefix_weight * (1.0 - jaro)


def token_jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity over normalized whitespace-delimited tokens."""
    ta = set(normalize_text(a).split())
    tb = set(normalize_text(b).split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def token_cosine_similarity(a: str, b: str) -> float:
    """Cosine similarity over normalized token term-frequency vectors."""
    ca = Counter(normalize_text(a).split())
    cb = Counter(normalize_text(b).split())
    if not ca and not cb:
        return 1.0
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb[t] for t in ca.keys() & cb.keys())
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def is_numeric_string(s: str) -> bool:
    """Return True if string represents a finite number."""
    return bool(_NUMBER_RE.match(s.strip()))


def numeric_similarity(a: str, b: str) -> float:
    """Relative numeric similarity in [0.0, 1.0], or 0.0 if either is non-numeric."""
    if not (is_numeric_string(a) and is_numeric_string(b)):
        return 0.0
    fa, fb = float(a), float(b)
    if fa == fb:
        return 1.0
    denom = abs(fa) + abs(fb)
    if denom == 0.0:
        return 1.0
    return max(0.0, 1.0 - abs(fa - fb) / denom)
