"""
Read-only viewer for the latest B1 run — the induced schema, the score table,
and the per-level match decisions.

Purely a display tool: it computes nothing and writes nothing, so it can never
affect a reported result. Run `python -m baselines.statistical` to produce a
schema and `python -m eval.report ...` to score it before using this.

Usage:
    python -m baselines.show_results              # schema + scores + matches
    python -m baselines.show_results --schema     # just the induced schema
    python -m baselines.show_results --scores     # just the F1 table
    python -m baselines.show_results --matches    # just the match decisions
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RAW = _REPO_ROOT / "results" / "raw"
_TABLES = _REPO_ROOT / "results" / "tables_figures"

LAYERS = [
    "classes",
    "taxonomy",
    "attributes_effective_micro",
    "attributes_declared_micro",
    "relations",
]
LEVELS = ("M1", "M2", "M3")


def _latest(directory: Path, pattern: str) -> Path | None:
    hits = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    return hits[-1] if hits else None


def _missing(what: str, how: str) -> None:
    print(f"{what} — run:\n    {how}\n")


def _latest_run_id() -> str | None:
    """run_id of the most recent B1 schema, which anchors everything else.

    Scores and match decisions are looked up BY THIS ID rather than by "latest
    file of that type". Picking each independently silently pairs a fresh
    schema with a stale score table -- numbers that never described the schema
    printed above them.
    """
    path = _latest(_RAW, "*_b1.json")
    return path.name[: -len("_b1.json")] if path else None


def _score_command(run_id: str) -> str:
    return (
        "python3 -m eval.report --gold schema/gold_schema.ttl \\\n"
        f"        --induced results/raw/{run_id}_b1.json --level all"
    )


def show_schema() -> None:
    path = _latest(_RAW, "*_b1.json")
    if path is None:
        return _missing("no B1 schema found", "python3 -m baselines.statistical")
    data = json.loads(path.read_text())

    print(f"=== INDUCED SCHEMA ({path.name}) ===\n")
    print(f"{'class (C-value rank order)':32} parent  attributes")
    for cls in data["classes"]:
        attrs = ", ".join(cls["attributes"]) or "-"
        print(f"  {cls['name']!r:30} {str(cls['parent']):6}  {attrs}")

    print(f"\n{'relations':32}")
    for rel in data["relations"]:
        print(f"  {rel['source']}  --{rel['label']}->  {rel['target']}")
    if not data["relations"]:
        print("  (none)")

    meta = data["metadata"]
    n_attrs = sum(len(c["attributes"]) for c in data["classes"])
    print(
        f"\n  {meta['condition']} / {meta['model']} / run {meta['run_id']}"
        f"\n  {len(meta['source_documents'])} source documents"
        f" | {len(data['classes'])} classes, {n_attrs} attributes,"
        f" {len(data['relations'])} relations\n"
    )


def show_scores() -> None:
    run_id = _latest_run_id()
    if run_id is None:
        return _missing("no B1 schema found", "python3 -m baselines.statistical")
    path = _TABLES / f"{run_id}.csv"
    if not path.exists():
        stale = _latest(_TABLES, "*.csv")
        note = f" (found {stale.name}, a DIFFERENT run — not shown)" if stale else ""
        return _missing(f"no scores for run {run_id}{note}", _score_command(run_id))
    cells: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    with path.open() as fh:
        for row in csv.DictReader(fh):
            cells[(row["level"], row["layer"])][row["metric"]] = row["value"]

    print(f"=== F1 (tp/fp/fn) — {path.name} ===\n")
    print(f"{'layer':30}" + "".join(lv.rjust(22) for lv in LEVELS))
    for layer in LAYERS:
        line = f"{layer:30}"
        for level in LEVELS:
            m = cells.get((level, layer))
            if not m:
                line += "n/a".rjust(22)
                continue
            line += f"{float(m['f1']):.3f} ({m['tp']}/{m['fp']}/{m['fn']})".rjust(22)
        print(line)
    print()


def show_matches() -> None:
    run_id = _latest_run_id()
    if run_id is None:
        return _missing("no B1 schema found", "python3 -m baselines.statistical")
    path = _RAW / f"{run_id}_matches.json"
    if not path.exists():
        return _missing(f"no match decisions for run {run_id}", _score_command(run_id))
    results = json.loads(path.read_text())["results"]

    print(f"=== CLASS MATCH DECISIONS ({path.name}) ===\n")
    for block in results:
        print(f"  {block['level']}:")
        for pair in block["classes_matched"]:
            print(
                f"    gold {pair['gold']!r:22} <- induced {pair['induced']!r:22}"
                f" score {pair['score']:.3f}"
            )
        print(f"    missed gold : {', '.join(block['classes_unmatched_gold'])}")
        print(f"    spurious    : {', '.join(block['classes_unmatched_induced'])}\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show the latest B1 run.")
    parser.add_argument("--schema", action="store_true")
    parser.add_argument("--scores", action="store_true")
    parser.add_argument("--matches", action="store_true")
    args = parser.parse_args(argv)

    show_all = not (args.schema or args.scores or args.matches)
    if args.schema or show_all:
        show_schema()
    if args.scores or show_all:
        show_scores()
    if args.matches or show_all:
        show_matches()


if __name__ == "__main__":
    main()
