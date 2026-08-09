"""
Step 4 — Evaluation Harness: CLI entry point + long-format reporting.

    python -m eval.report --gold schema/gold_schema.ttl --induced <path.json> \\
        --level {M1,M2,M3,all} [--allow-inverse] [--out-dir results/tables_figures]

Scores an induced schema against the gold schema at one or all three
strictness levels, and writes:

  - a long-format CSV to <out-dir>/<run_id>.csv -- one row per
    (run_id, model, condition, level, layer, metric, value). Deliberately
    long (not wide) format: turning it into the paper's tables is a
    groupby, not a reshape.
  - a raw class-match-decisions JSON to <raw-dir>/<run_id>_matches.json --
    which gold/induced class names matched, and at what score, at each
    level. Feeds the Definition-of-Done "100 sampled M3 decisions
    hand-verified" step and eval/error_analysis.py once that's filled in.

Domain-agnostic per eval/PLAN.md §0 -- this file must contain zero
domain-specific class/attribute/relation names from any concrete gold
schema. (Not one of the three files test_no_domain_leakage enforces --
schema_ir.py/matching.py/metrics.py are the core engine -- but held to the
same standard as a matter of design discipline: report.py is pure
orchestration and shouldn't need domain knowledge either.)
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from eval.matching import EmbeddingCache, load_cre_config
from eval.metrics import PRF1, ScoringResult, score_schema
from eval.schema_ir import load_gold_ttl, load_induced_json, load_induced_metadata

DEFAULT_CRE_CONFIG = Path("eval/config/cre.yaml")
DEFAULT_OUT_DIR = Path("results/tables_figures")
DEFAULT_RAW_DIR = Path("results/raw")
DEFAULT_EMBEDDING_CACHE = Path("eval/.cache/embeddings.json")

LEVELS = ("M1", "M2", "M3")


def _prf1_rows(layer: str, score: PRF1 | None, run_id: str, model: str, condition: str, level: str) -> list[dict]:
    if score is None:
        return []
    values = {
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "tp": score.tp,
        "fp": score.fp,
        "fn": score.fn,
    }
    return [
        {
            "run_id": run_id,
            "model": model,
            "condition": condition,
            "level": level,
            "layer": layer,
            "metric": metric,
            "value": value,
        }
        for metric, value in values.items()
    ]


def result_to_rows(result: ScoringResult, run_id: str, model: str, condition: str) -> list[dict]:
    """Flattens one ScoringResult into the long-format row shape."""
    layers = [
        ("classes", result.classes),
        ("taxonomy", result.taxonomy),
        ("attributes_effective_micro", result.attributes.effective_micro),
        ("attributes_effective_macro", result.attributes.effective_macro),
        ("attributes_declared_micro", result.attributes.declared_micro),
        ("attributes_declared_macro", result.attributes.declared_macro),
        ("relations", result.relations),
        ("relations_allow_inverse", result.relations_allow_inverse),
    ]
    rows: list[dict] = []
    for layer_name, score in layers:
        rows.extend(_prf1_rows(layer_name, score, run_id, model, condition, result.level))
    return rows


def result_to_match_decisions(result: ScoringResult) -> dict:
    """Raw class-match decisions for one level -- gold/induced pairs and
    their score, plus the unmatched leftovers on each side. Extend this
    (attribute- and relation-label match decisions) when
    error_analysis.py's sample_m3_decisions() needs them."""
    return {
        "level": result.level,
        "classes_matched": [
            {"gold": g, "induced": i, "score": s} for g, i, s in result.class_match.matched
        ],
        "classes_unmatched_gold": list(result.class_match.unmatched_gold),
        "classes_unmatched_induced": list(result.class_match.unmatched_induced),
    }


def run(
    gold_path: Path,
    induced_path: Path,
    level_arg: str,
    cre_config_path: Path,
    embedding_cache_path: Path,
    allow_inverse: bool,
    out_dir: Path,
    raw_dir: Path,
) -> tuple[Path, Path]:
    gold = load_gold_ttl(gold_path)
    induced = load_induced_json(induced_path)
    metadata = load_induced_metadata(induced_path)

    run_id = metadata.get("run_id", induced_path.stem)
    model = metadata.get("model", "unknown")
    condition = metadata.get("condition", "unknown")

    cfg = load_cre_config(cre_config_path)
    cache = EmbeddingCache(embedding_cache_path)

    levels = list(LEVELS) if level_arg == "all" else [level_arg]

    all_rows: list[dict] = []
    all_match_decisions: list[dict] = []
    for level in levels:
        result = score_schema(gold, induced, level, cfg, cache, allow_inverse=allow_inverse)
        all_rows.extend(result_to_rows(result, run_id, model, condition))
        all_match_decisions.append(result_to_match_decisions(result))

    cache.flush()

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{run_id}.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "model", "condition", "level", "layer", "metric", "value"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    matches_path = raw_dir / f"{run_id}_matches.json"
    matches_path.write_text(
        json.dumps({"run_id": run_id, "model": model, "condition": condition, "results": all_match_decisions}, indent=2)
    )

    return csv_path, matches_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score an induced schema against the gold schema (eval/PLAN.md Step 4)."
    )
    parser.add_argument("--gold", required=True, type=Path, help="Path to the gold TTL schema.")
    parser.add_argument("--induced", required=True, type=Path, help="Path to an induced-schema JSON file (PLAN.md §2 contract).")
    parser.add_argument("--level", default="all", choices=[*LEVELS, "all"], help="Strictness level to score at (default: all).")
    parser.add_argument("--allow-inverse", action="store_true", help="Also compute the D3 direction-tolerant relations variant.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path, help="Directory for the long-format CSV.")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, type=Path, help="Directory for the raw match-decisions JSON.")
    parser.add_argument("--cre-config", default=DEFAULT_CRE_CONFIG, type=Path, help="Path to the domain pack YAML.")
    parser.add_argument("--embedding-cache", default=DEFAULT_EMBEDDING_CACHE, type=Path, help="Path to the M3 embedding cache file.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    csv_path, matches_path = run(
        gold_path=args.gold,
        induced_path=args.induced,
        level_arg=args.level,
        cre_config_path=args.cre_config,
        embedding_cache_path=args.embedding_cache,
        allow_inverse=args.allow_inverse,
        out_dir=args.out_dir,
        raw_dir=args.raw_dir,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {matches_path}")


if __name__ == "__main__":
    main()
