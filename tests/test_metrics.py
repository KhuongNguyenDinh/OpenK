"""Unit tests for OpenK string, phonetic, token, and numeric metrics."""

import pytest

from openk.core.metrics import (
    damerau_levenshtein_distance,
    damerau_levenshtein_similarity,
    fingerprint_key,
    hamming_distance,
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


def test_fingerprint_key_handles_order_punctuation_and_accents():
    assert fingerprint_key("Cruise, Tom") == "cruise tom"
    assert fingerprint_key("Tom   Cruise!") == "cruise tom"
    assert fingerprint_key("José García") == "garcia jose"


def test_ngram_fingerprint_key():
    assert ngram_fingerprint_key("Paris", n=2) == ngram_fingerprint_key("paris!", n=2)
    with pytest.raises(ValueError):
        ngram_fingerprint_key("abc", n=0)


def test_metaphone_key_groups_phonetic_variants():
    assert metaphone_key("Stephen") == metaphone_key("Steven") or metaphone_key("Philip") == metaphone_key("Filip")
    assert metaphone_key("Catherine") == metaphone_key("Katherine")


def test_edit_distances_and_similarities():
    assert levenshtein_distance("kitten", "sitting") == 3
    assert damerau_levenshtein_distance("fish", "ifsh") == 1
    assert levenshtein_distance("fish", "ifsh") == 2
    assert hamming_distance("abc", "abcd") == 1
    assert 0.0 <= levenshtein_similarity("abc", "abd") <= 1.0
    assert 0.0 <= damerau_levenshtein_similarity("fish", "ifsh") <= 1.0


def test_jaro_and_jaro_winkler():
    j = jaro_similarity("martha", "marhta")
    jw = jaro_winkler_similarity("martha", "marhta")
    assert j > 0.9
    assert jw >= j


def test_token_and_numeric_similarities():
    assert token_jaccard_similarity("New York City", "City of New York") > 0.5
    assert token_cosine_similarity("Apple Inc", "Inc Apple") == pytest.approx(1.0)
    assert numeric_similarity("100", "101") > 0.98
    assert numeric_similarity("100", "abc") == 0.0
