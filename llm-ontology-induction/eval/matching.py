"""
Step 4 — Evaluation Harness: string matching at three strictness levels
(M1 exact-normalized, M2 fuzzy+lexicon, M3 semantic embedding) plus the
bipartite assignment that turns pairwise scores into a one-to-one matching.

Domain-agnostic per eval/PLAN.md §0 — this file must contain zero
domain-specific class/attribute/relation names from any concrete gold
schema. The one domain-bound input (the synonym lexicon) is read from
config/cre.yaml at call time, never hardcoded here. See
eval/tests/test_harness.py::test_no_domain_leakage for the enforced check.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import inflect
import numpy as np
import yaml
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cosine as _cosine_distance

_INFLECT = inflect.engine()

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize(s: str) -> str:
    """Canonicalize a class/attribute/relation-label string for exact
    (M1) comparison, and as the shared first step of M2/M3 so every level
    matches on the same normalized form (PLAN.md §8: normalize once,
    centrally -- don't re-normalize inside each matcher independently).

    Steps: split camelCase/PascalCase -> replace `_`/`-` with spaces ->
    strip punctuation -> lowercase -> tokenize -> singularize each token
    (via `inflect`, never a hand-rolled rule -- naive suffix-stripping is
    the classic "-y -> -ie" bug) -> rejoin.
    """
    s = _CAMEL_BOUNDARY.sub(" ", s)
    s = s.replace("_", " ").replace("-", " ")
    s = s.lower()
    s = _NON_ALNUM.sub(" ", s)
    tokens = s.split()
    singular_tokens = [_INFLECT.singular_noun(t) or t for t in tokens]
    return " ".join(singular_tokens)


def is_m1_match(a: str, b: str) -> bool:
    """Exact match after normalization -- the strictest level."""
    return normalize(a) == normalize(b)


# ---------------------------------------------------------------------------
# Domain-pack config (config/cre.yaml) -- loaded explicitly by callers, never
# as a hidden module-level global, so this file stays testable with a
# swapped-in config and free of domain-specific state.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Thresholds:
    m2_token_set_ratio: float
    m3_cosine_similarity: float
    frozen: bool = False


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    model_version: str | None
    cache_path: str


@dataclass(frozen=True)
class CREConfig:
    stopwords: frozenset[str]
    synonyms: dict[str, frozenset[str]]  # normalized term -> normalized synonym set (symmetric)
    thresholds: Thresholds
    embedding: EmbeddingConfig


def load_cre_config(path: str | Path) -> CREConfig:
    """Load a domain pack YAML (config/cre.yaml) into a CREConfig.

    Builds a symmetric synonym lookup at load time: if `owner` lists
    `landlord`, then `synonyms["owner"]` contains `landlord` AND
    `synonyms["landlord"]` contains `owner` -- callers never need to check
    both directions themselves. Synonym terms are stored normalized (via
    `normalize()`) so lookups at match time don't need a second pass.
    """
    raw = yaml.safe_load(Path(path).read_text())

    symmetric: dict[str, set[str]] = {}
    for key, values in raw.get("synonyms", {}).items():
        norm_key = normalize(key)
        norm_values = {normalize(v) for v in values}
        symmetric.setdefault(norm_key, set()).update(norm_values)
        for v in norm_values:
            symmetric.setdefault(v, set()).add(norm_key)

    thresholds_raw = raw.get("thresholds", {})
    thresholds = Thresholds(
        m2_token_set_ratio=thresholds_raw["m2_token_set_ratio"],
        m3_cosine_similarity=thresholds_raw["m3_cosine_similarity"],
        frozen=thresholds_raw.get("frozen", False),
    )

    embedding_raw = raw.get("embedding", {})
    embedding = EmbeddingConfig(
        model_name=embedding_raw["model_name"],
        model_version=embedding_raw.get("model_version"),
        cache_path=embedding_raw.get("cache_path", "eval/.cache/embeddings.json"),
    )

    return CREConfig(
        stopwords=frozenset(normalize(w) for w in raw.get("stopwords", [])),
        synonyms={k: frozenset(v) for k, v in symmetric.items()},
        thresholds=thresholds,
        embedding=embedding,
    )


# ---------------------------------------------------------------------------
# M2 -- fuzzy token-set similarity + synonym lexicon
# ---------------------------------------------------------------------------

def m2_score(a: str, b: str, cfg: CREConfig) -> float:
    """Similarity in [0, 1]. A lexicon hit scores 1.0 outright (bypassing
    the fuzzy threshold entirely -- a known synonym is a known synonym
    regardless of surface edit-distance); otherwise falls back to
    rapidfuzz's token-set ratio, which is robust to word-order/subset
    differences in multi-word labels in a way raw Levenshtein isn't.

    Token-set ratio alone is NOT sufficient to distinguish real synonyms
    from unrelated near-spellings on single-token strings -- e.g. "lease"
    vs "least" scores 80/100, on par with or above many genuine synonym
    pairs' fuzzy scores. That's exactly why the lexicon is checked first
    and is authoritative for the vocabulary it covers; the fuzzy fallback
    is deliberately a narrow, high-threshold safety net for near-identical
    spelling variants the lexicon and normalize() didn't already catch, not
    a general-purpose synonym detector (that's M3's job).

    Blends token_set_ratio with token_sort_ratio (takes the min) rather
    than using token_set_ratio alone. token_set_ratio scores a PERFECT
    100/100 for pure token-containment -- e.g. "widget" vs "office widget"
    -- because it only compares each string's token set against their
    intersection, discarding whatever extra tokens the longer string has.
    That's fatal for any schema where a class shares a compound name with
    its own subtype (a superclass/subclass pair like "Foo"/"OfficeFoo" is
    common taxonomy-naming practice): it would tie a superclass's
    self-match against its subclass's match at score 1.0, letting the
    bipartite assignment legally swap them -- caught empirically via a toy
    fixture built from this repo's own gold schema, which has exactly this
    shape. token_sort_ratio doesn't have that flaw (it's still just an
    edit-distance ratio, so a strict token superset is penalized for its
    extra length) while agreeing with token_set_ratio on genuine
    multi-word reorderings/synonyms -- so the min of the two keeps the
    token-set behavior's robustness to word order without its containment
    blind spot.
    """
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return 1.0
    if nb in cfg.synonyms.get(na, frozenset()):
        return 1.0
    return min(fuzz.token_set_ratio(na, nb), fuzz.token_sort_ratio(na, nb)) / 100.0


def is_m2_match(a: str, b: str, cfg: CREConfig) -> bool:
    return m2_score(a, b, cfg) >= cfg.thresholds.m2_token_set_ratio


# ---------------------------------------------------------------------------
# M3 -- semantic embedding cosine similarity
# ---------------------------------------------------------------------------

# In-process model cache, keyed by model name -- avoids reloading the
# (multi-second) sentence-transformers model on every m3_score() call within
# a run. Separate from EmbeddingCache below, which caches *vectors* to disk
# across runs.
_EMBEDDER_CACHE: dict[str, object] = {}


def get_embedder(model_name: str):
    """Lazily import sentence-transformers (heavy, pulls in torch) so
    M1/M2-only callers never pay that import cost, and cache the loaded
    model per process."""
    if model_name not in _EMBEDDER_CACHE:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


class EmbeddingCache:
    """Disk-backed cache of normalized-string -> embedding vector, keyed on
    (model_name, model_version, normalized_string) so a silent
    sentence-transformers/model upgrade can't quietly shift scores
    mid-experiment without the cache key changing under it (PLAN.md §8).

    Batch-flushed (call `.flush()` once at the end of a run) rather than
    written on every `.set()`, for I/O efficiency.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, list[float]] = {}
        self._dirty = False
        if self.path.exists():
            self._data = json.loads(self.path.read_text())

    @staticmethod
    def _key(model_name: str, model_version: str | None, normalized_string: str) -> str:
        return f"{model_name}::{model_version}::{normalized_string}"

    def get(self, model_name: str, model_version: str | None, normalized_string: str):
        return self._data.get(self._key(model_name, model_version, normalized_string))

    def set(self, model_name: str, model_version: str | None, normalized_string: str, vector) -> None:
        key = self._key(model_name, model_version, normalized_string)
        self._data[key] = list(float(x) for x in vector)
        self._dirty = True

    def flush(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data))
        self._dirty = False


def _embed(s_normalized: str, cfg: CREConfig, cache: EmbeddingCache):
    cached = cache.get(cfg.embedding.model_name, cfg.embedding.model_version, s_normalized)
    if cached is not None:
        return cached
    embedder = get_embedder(cfg.embedding.model_name)
    vector = embedder.encode(s_normalized)
    cache.set(cfg.embedding.model_name, cfg.embedding.model_version, s_normalized, vector)
    return vector


def m3_score(a: str, b: str, cfg: CREConfig, cache: EmbeddingCache) -> float:
    """Cosine similarity in [-1, 1] (in practice ~[0, 1] for sentence
    embeddings) between the sentence-transformer embeddings of `a` and `b`.
    Normalizes both first (M1), matching the "normalize once, centrally"
    rule, then embeds/caches the *normalized* string -- so surface-form
    variants that already collapse under M1 share one cache entry."""
    na, nb = normalize(a), normalize(b)
    if na == nb:
        return 1.0
    va = _embed(na, cfg, cache)
    vb = _embed(nb, cfg, cache)
    return float(1.0 - _cosine_distance(va, vb))


def is_m3_match(a: str, b: str, cfg: CREConfig, cache: EmbeddingCache) -> bool:
    return m3_score(a, b, cfg, cache) >= cfg.thresholds.m3_cosine_similarity


# ---------------------------------------------------------------------------
# Bipartite assignment -- turns pairwise scores at one strictness level into
# a one-to-one matching between a gold set and an induced set.
# ---------------------------------------------------------------------------

LEVELS = ("M1", "M2", "M3")


def _score_pair(a: str, b: str, level: str, cfg: CREConfig, cache: EmbeddingCache | None) -> float:
    if level == "M1":
        return 1.0 if normalize(a) == normalize(b) else 0.0
    if level == "M2":
        return m2_score(a, b, cfg)
    if level == "M3":
        if cache is None:
            raise ValueError("M3 scoring requires an EmbeddingCache")
        return m3_score(a, b, cfg, cache)
    raise ValueError(f"unknown level: {level!r} (expected one of {LEVELS})")


def _threshold_for(level: str, cfg: CREConfig) -> float:
    if level == "M1":
        return 1.0
    if level == "M2":
        return cfg.thresholds.m2_token_set_ratio
    if level == "M3":
        return cfg.thresholds.m3_cosine_similarity
    raise ValueError(f"unknown level: {level!r} (expected one of {LEVELS})")


@dataclass(frozen=True)
class MatchResult:
    matched: tuple[tuple[object, object, float], ...]  # (gold_item, induced_item, score)
    unmatched_gold: tuple[object, ...]
    unmatched_induced: tuple[object, ...]


def _bipartite_match(gold_items: list, induced_items: list, score_fn, threshold: float) -> MatchResult:
    """Core one-to-one matching over arbitrary items, via
    scipy.optimize.linear_sum_assignment over the pairwise similarity
    matrix (cost = 1 - similarity). `score_fn(gold_item, induced_item)`
    supplies the similarity; callers decide what an "item" is (a plain
    name string for match_sets(), a Relation object scored by its label
    for metrics.py's relation-bucket matching, etc.).

    Deliberately NOT greedy (PLAN.md §8's load-bearing pitfall: "greedy
    matching inflates recall, hardest bug to notice because the numbers
    look plausible"). This is also what makes the split-class case correct
    with no special-casing: if induced has two plausible matches for one
    gold item, an Nx1 assignment problem can only ever produce one pair --
    the other candidate is left unmatched by construction, never double-
    counted as a second TP.

    Pairs the optimizer picks are still filtered by `threshold` afterward
    -- the "best available" pair under an optimal assignment can still be
    a bad match if nothing clears the bar (D6-adjacent: a below-threshold
    pair contributes to neither TP nor a phantom match).
    """
    if not gold_items or not induced_items:
        return MatchResult(
            matched=(),
            unmatched_gold=tuple(gold_items),
            unmatched_induced=tuple(induced_items),
        )

    score_matrix = np.array(
        [[score_fn(g, i) for i in induced_items] for g in gold_items]
    )
    cost_matrix = 1.0 - score_matrix
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matched: list[tuple[object, object, float]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(row_idx, col_idx):
        score = float(score_matrix[r, c])
        if score >= threshold:
            matched.append((gold_items[r], induced_items[c], score))
            matched_rows.add(r)
            matched_cols.add(c)

    unmatched_gold = tuple(g for idx, g in enumerate(gold_items) if idx not in matched_rows)
    unmatched_induced = tuple(i for idx, i in enumerate(induced_items) if idx not in matched_cols)

    return MatchResult(
        matched=tuple(matched),
        unmatched_gold=unmatched_gold,
        unmatched_induced=unmatched_induced,
    )


def match_sets(
    gold_names,
    induced_names,
    level: str,
    cfg: CREConfig,
    cache: EmbeddingCache | None = None,
) -> MatchResult:
    """Optimal one-to-one matching between `gold_names` and `induced_names`
    (plain strings -- class names, attribute names) at the given strictness
    level. The name-string specialization of `bipartite_match_by_key()`."""
    threshold = _threshold_for(level, cfg)
    score_fn = lambda g, i: _score_pair(g, i, level, cfg, cache)  # noqa: E731
    return _bipartite_match(list(gold_names), list(induced_names), score_fn, threshold)


def bipartite_match_by_key(
    gold_items,
    induced_items,
    key,
    level: str,
    cfg: CREConfig,
    cache: EmbeddingCache | None = None,
) -> MatchResult:
    """Generic one-to-one matching over arbitrary objects (e.g. Relation
    instances), scoring each candidate pair by `key(item) -> str` at the
    given strictness level. Used by metrics.py's relations layer: matching
    happens on relation *labels* via `key=lambda r: r.label`, but the
    result carries the original Relation objects, not just their label
    strings -- important because two different relations (e.g. from two
    induced classes that both matched the same gold class in a split-class
    case) can legitimately share an identical label string, and losing
    object identity there would make it impossible to attribute TP/FP/FN
    back to specific relation instances.
    """
    threshold = _threshold_for(level, cfg)
    score_fn = lambda g, i: _score_pair(key(g), key(i), level, cfg, cache)  # noqa: E731
    return _bipartite_match(list(gold_items), list(induced_items), score_fn, threshold)
