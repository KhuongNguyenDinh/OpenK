# Clustering & Similarity Algorithms

OpenK implements **13 clustering and similarity methods** grouped into three families, plus an **Auto-Recommender** (`recommend_clustering`).

---

## 1. Key-Collision Methods — $O(N)$

Key-collision methods map each string $s$ to a canonical key $k(s)$ and group all strings sharing the same key.

1. **`fingerprint`**:
   - Steps: Unicode NFKD ASCII-fold $\rightarrow$ lowercase $\rightarrow$ strip punctuation $\rightarrow$ collapse whitespace $\rightarrow$ split into tokens $\rightarrow$ deduplicate and sort tokens $\rightarrow$ join with space.
   - Example: `"José García"` and `"Garcia, Jose"` both map to `"garcia jose"`.
   - Best for: Multi-word names, addresses, organizations with word-order or punctuation differences.

2. **`ngram_fingerprint`** (default $n = 2$):
   - Steps: Normalize text, remove spaces, extract unique sorted character $n$-grams, join.
   - Best for: Catching minor spelling/spacing differences in short strings.

3. **`metaphone`**:
   - Steps: Phonetic encoding per token (collapsing duplicate letters, mapping `ph -> f`, `kn -> n`, `c[eiy] -> s`, dropping internal vowels), sorted unique token keys.
   - Example: `"Catherine"` and `"Katherine"` share the same phonetic key.
   - Best for: Names, cities, and brands with phonetic spelling variations.

---

## 2. Pairwise Similarity Methods — $O(N^2)$ with Length Pruning

Pairs $(a, b)$ with $\text{sim}(a, b) \ge \tau$ are merged via **Union-Find** (`UnionFind`) to compute exact transitive closure.

| Method | Formula / Idea | Default $\tau$ | Best Use Case |
| :--- | :--- | :---: | :--- |
| `jaro` | Matches within $\lfloor\max(\|a\|,\|b\|)/2\rfloor - 1$ + transpositions | `0.85` | Short strings, typos |
| `jaro_winkler` | Jaro boosted by common prefix length (up to 4 chars) | `0.88` | Person/company names, prefixes |
| `levenshtein_sim` | $1 - d_{\text{Lev}}(a, b) / \max(\|a\|, \|b\|)$ | `0.80` | General typo detection |
| `damerau_sim` | $1 - d_{\text{Dam}}(a, b) / \max(\|a\|, \|b\|)$ | `0.80` | Adjacent character swaps (`fish` vs `ifsh`) |
| `token_jaccard` | $\|T_a \cap T_b\| / \|T_a \cup T_b\|$ | `0.65` | Multi-word titles with extra/missing words |
| `token_cosine` | Cosine of token frequency vectors | `0.70` | Multi-word descriptions |
| `num_sim` | $1 - \|a - b\| / (\|a\| + \|b\|)$ | `0.95` | Near-duplicate numeric values |

*Optimization*: For `levenshtein_sim` and `damerau_sim`, pairs with $\min(|a|, |b|) / \max(|a|, |b|) < \tau$ are skipped in $O(1)$ without computing the DP matrix.

---

## 3. Edit-Distance (KNN) Methods — $O(N^2)$ with Radius Pruning

Pairs $(a, b)$ with $d(a, b) \le r$ are merged via Union-Find. Pairs with $||a| - |b|| > r$ are skipped in $O(1)$.

- `levenshtein` (default $r = 2$)
- `damerau` (default $r = 2$)
- `hamming` (default $r = 2$)
