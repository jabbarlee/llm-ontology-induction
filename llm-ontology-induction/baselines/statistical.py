"""
B1 baseline — zero-LLM schema induction (baselines/DECISIONS.md).

Reads the whole document corpus as one pooled run (D1), scores candidate terms
by C-value (D2), splits them into classes and attributes by a frequency-and-
position heuristic (D4), extracts relations from dependency-parsed SVO triples
(D5), and emits a single induced-schema JSON matching the contract in
eval/schema_ir.py::parse_induced_schema (eval/PLAN.md §2).

Every class is emitted with parent=null (D6): B1 has no mechanism to infer
hierarchy from term statistics, and a nonzero taxonomy score here would mean a
bug, not a success.

HARD RULE (D3): zero domain vocabulary in this module. Enforced by
baselines/tests/test_statistical.py::test_no_domain_vocabulary_leakage.

Usage:
    python -m baselines.statistical [--out-dir results/raw] [--limit N]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from baselines import term_extraction as te

# --- Frozen hyperparameters (DECISIONS.md D7) ----------------------------
TOP_N_CLASSES = 20
MIN_ATTR_COOC = 3
MAX_ATTRS_PER_CLASS = 10
MIN_RELATION_FREQ = 3

CONDITION = "B1"
MODEL_ID = "statistical-tfidf-cvalue"

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CORPUS_ROOT = _REPO_ROOT / "data" / "documents"
_SUBDIRS = ("csv_exports", "lease_texts", "notes", "messages")
DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "raw"


# ---------------------------------------------------------------------------
# Corpus loading (D1)
# ---------------------------------------------------------------------------

def _flatten_csv(path: Path) -> list[str]:
    """One pseudo-sentence per row: "{col}: {val}, {col}: {val}, ...".

    Empty cells are dropped rather than emitted as "col: " -- a blank value is
    an absent fact, and keeping it would inject a bare colon that the attribute
    patterns (D4) would try to read.
    """
    sentences: list[str] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            parts = [
                f"{col}: {val}".strip()
                for col, val in row.items()
                if col and val and str(val).strip()
            ]
            if parts:
                sentences.append(", ".join(parts))
    return sentences


def _split_text(path: Path) -> list[str]:
    """Sentence-split a prose/notes/messages file.

    Uses the loaded pipeline's parser-based sentence boundaries. Blank-line
    blocks are split first so a message thread's turns cannot be glued into one
    sentence by a missing terminal period.
    """
    nlp = te.get_nlp()
    raw = path.read_text(encoding="utf-8")
    sentences: list[str] = []
    for block in raw.split("\n\n"):
        block = " ".join(block.split())
        if not block:
            continue
        for sent in nlp(block).sents:
            text = sent.text.strip()
            if text:
                sentences.append(text)
    return sentences


def load_corpus(
    root: Path = _CORPUS_ROOT, limit: int | None = None
) -> tuple[list[str], list[str]]:
    """Read every document under the corpus subdirectories (D1).

    Returns (sentences, source_documents) where source_documents holds the
    repo-relative path of each file actually read, in the order read -- that
    list is what lands in metadata.source_documents, so it must reflect real
    reads rather than a glob pattern.

    `limit` caps files per subdirectory; it exists for smoke-testing the
    pipeline end-to-end and is never used for a reported run.
    """
    sentences: list[str] = []
    sources: list[str] = []
    for subdir in _SUBDIRS:
        directory = root / subdir
        if not directory.is_dir():
            continue
        paths = sorted(p for p in directory.iterdir() if p.is_file())
        for path in paths[:limit] if limit else paths:
            if path.suffix.lower() == ".csv":
                extracted = _flatten_csv(path)
            elif path.suffix.lower() == ".txt":
                extracted = _split_text(path)
            else:
                continue
            sentences.extend(extracted)
            sources.append(f"{subdir}/{path.name}")
    return sentences, sources


# ---------------------------------------------------------------------------
# Induction (D2, D4, D5, D6)
# ---------------------------------------------------------------------------

def induce_schema(sentences: list[str]) -> dict:
    """Run the full B1 pipeline over pooled sentences -> schema dict.

    Split from main() so the whole induction is testable and re-runnable
    without touching argv or disk.
    """
    # --- D2/D3: candidate terms, C-value ranked ---
    terms = te.extract_candidate_terms(sentences)
    forms = te.surface_forms(sentences)
    nesting = te.build_nesting_map(terms)
    scores = te.c_value(terms, nesting)

    # --- D4: top-N by C-value become classes ---
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    class_keys = [term for term, _score in ranked[:TOP_N_CLASSES]]
    class_set = set(class_keys)

    def name_of(key: str) -> str:
        """Emitted name: the observed surface form (D3a), never normalized."""
        return forms.get(key, key)

    # --- D4: attributes per class ---
    attr_hits = te.extract_attribute_candidates(sentences, class_set, set(terms))
    classes = []
    for key in class_keys:
        counted = attr_hits.get(key, Counter())
        kept = [
            attr
            for attr, count in sorted(counted.items(), key=lambda kv: (-kv[1], kv[0]))
            if count >= MIN_ATTR_COOC
        ][:MAX_ATTRS_PER_CLASS]
        classes.append(
            {
                "name": name_of(key),
                "parent": None,  # D6 -- B1 detects no hierarchy, ever.
                "attributes": [name_of(a) for a in kept],
            }
        )

    # --- D5: relations from SVO triples above the frequency floor ---
    triple_counts = Counter(te.extract_svo_triples(sentences, class_set))
    relations = [
        {"source": name_of(subj), "label": verb, "target": name_of(obj)}
        for (subj, verb, obj), count in sorted(
            triple_counts.items(), key=lambda kv: (-kv[1], kv[0])
        )
        if count >= MIN_RELATION_FREQ
    ]

    return {"classes": classes, "relations": relations}


def build_output(sentences: list[str], sources: list[str], run_id: str) -> dict:
    schema = induce_schema(sentences)
    schema["metadata"] = {
        "condition": CONDITION,
        "model": MODEL_ID,
        "run_id": run_id,
        "source_documents": sources,
    }
    return schema


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:4]}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the B1 statistical baseline.")
    parser.add_argument("--corpus", type=Path, default=_CORPUS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap files per subdirectory (smoke test only -- not for a reported run).",
    )
    args = parser.parse_args(argv)

    sentences, sources = load_corpus(args.corpus, limit=args.limit)
    print(f"corpus: {len(sources)} documents -> {len(sentences)} sentences")

    run_id = _run_id()
    output = build_output(sentences, sources, run_id)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / f"{run_id}_b1.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    n_attrs = sum(len(c["attributes"]) for c in output["classes"])
    print(
        f"induced: {len(output['classes'])} classes, {n_attrs} attributes, "
        f"{len(output['relations'])} relations"
    )
    print(f"wrote {os.path.relpath(out_path, _REPO_ROOT)}")


if __name__ == "__main__":
    main()
