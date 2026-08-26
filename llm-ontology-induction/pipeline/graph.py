"""
P1 pipeline -- LangGraph assembly (architecture plan §3, §7).

Six nodes, one linear chain: extract -> consolidate_types -> consolidate_attrs
-> reconcile_relations -> induce_taxonomy -> assemble. Each node's own module
carries the actual justification for its stage; this file only wires them
together and owns the two things a graph-level file should own: the
checkpointer, and the CLI.

**Why LangGraph for a linear chain, honestly stated (plan §3):** the graph
abstraction itself is not the draw for six stages that always run in the same
order -- plain Python functions called in sequence would do that. Two things
are: checkpointing (192 extraction calls plus four more paid single calls,
across two models; if a call late in the chain fails, resume rather than
re-pay for everything before it) and state inspection between nodes (every
intermediate artifact -- partial schemas, the merge log, the pre-taxonomy
class list -- addressable via `graph.get_state()`, not buried in a variable
that dies with the process). If neither mattered, this would be four function
calls in `pipeline/nodes/`. They do matter at this call count and price.

**Resumability, concretely.** A run's checkpoint is keyed by its own run_id
(`thread_id == run_id` -- one sqlite file per run, not one shared file with
many threads, so a checkpoint's lifetime is legible from its filename alone).
`--resume RUN_ID` reconnects to an existing run_id's checkpoint file and its
existing raw-call log (opened in append mode, so calls already paid for are
never re-logged, let alone re-paid-for) and continues from whichever node the
graph's own checkpoint says comes next -- see the LangGraph
`get_state()`/`invoke(None, config=...)` pattern this relies on.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from baselines.b3_single_shot.single_shot import PROMPT_PATH as B3_PROMPT_PATH
from baselines.b3_single_shot.single_shot import _CORPUS_ROOT, load_documents, load_prompt_template, render_prompt
from baselines.shared.model_clients import MODELS

from pipeline.nodes.assemble import assemble
from pipeline.nodes.consolidate_attrs import consolidate_attrs
from pipeline.nodes.consolidate_types import consolidate_types
from pipeline.nodes.extract import extract
from pipeline.nodes.induce_taxonomy import induce_taxonomy
from pipeline.nodes.reconcile_relations import reconcile_relations
from pipeline.state import P1State

CONDITION = "P1"

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = _REPO_ROOT / "results" / "raw"
CHECKPOINT_DIR = DEFAULT_OUT_DIR / ".checkpoints"

STAGE_ORDER = (
    "extract",
    "consolidate_types",
    "consolidate_attrs",
    "reconcile_relations",
    "induce_taxonomy",
    "assemble",
)


def build_graph_definition() -> StateGraph:
    """The uncompiled graph -- a checkpointer is bound at compile time
    (`.compile(checkpointer=...)`), not here, so tests can compile the same
    definition against an in-memory saver without touching disk."""
    builder = StateGraph(P1State)
    builder.add_node("extract", extract)
    builder.add_node("consolidate_types", consolidate_types)
    builder.add_node("consolidate_attrs", consolidate_attrs)
    builder.add_node("reconcile_relations", reconcile_relations)
    builder.add_node("induce_taxonomy", induce_taxonomy)
    builder.add_node("assemble", assemble)

    builder.add_edge(START, "extract")
    builder.add_edge("extract", "consolidate_types")
    builder.add_edge("consolidate_types", "consolidate_attrs")
    builder.add_edge("consolidate_attrs", "reconcile_relations")
    builder.add_edge("reconcile_relations", "induce_taxonomy")
    builder.add_edge("induce_taxonomy", "assemble")
    builder.add_edge("assemble", END)
    return builder


def _new_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stamp}-{uuid4().hex[:4]}"


def _raw_log_path(run_id: str, model_key: str, out_dir: Path) -> Path:
    return out_dir / f"{run_id}_p1_{model_key}_calls.jsonl"


def _dry_run(spec, documents: list[tuple[str, str]]) -> None:
    """Per-stage call counts and token estimates for the full graph. Makes no
    API call, spends nothing. Stage 1's estimate is a real character count
    against the actual corpus; Stages 2-5 genuinely cannot be estimated this
    way -- their input is the *output* of the stage before them, which does
    not exist without a live call. Reported as N/A rather than guessed, the
    same honesty standard B3-D3's own token predictions were held to before
    a real run existed to check them against.
    """
    extraction_template = load_prompt_template(B3_PROMPT_PATH)
    sample_prompt = render_prompt(extraction_template, [documents[0]])
    total_chars = sum(len(render_prompt(extraction_template, [d])) for d in documents)
    est_tokens = round(total_chars / 3.5)

    print(f"DRY RUN -- no API calls. model={spec.key} ({spec.model_id})")
    print(f"output cap: {spec.max_output_tokens} tokens\n")
    print(f"{len(documents)} documents in corpus\n")
    print("Per-stage call plan:")
    print(f"  1. extract              -- {len(documents)} calls (one per document)")
    print("  2. consolidate_types    -- 1 call")
    print("  3. consolidate_attrs    -- 1 call")
    print("  4. reconcile_relations  -- 1 call (0 if no relations survive endpoint remapping)")
    print("  5. induce_taxonomy      -- 1 call")
    print("  6. assemble             -- 0 calls (deterministic)")
    print(f"  total: {len(documents) + 4}-{len(documents) + 5} calls\n")
    print(f"Stage 1 total input: {total_chars} chars, ~{est_tokens} est. input tokens across all {len(documents)} calls")
    print("Stages 2-5 input size: not estimable without Stage 1's real output -- each stage's")
    print("  prompt is built from the previous stage's actual response, not from the corpus directly.\n")
    print(f"--- rendered extraction prompt, document 1 ({len(sample_prompt)} chars) ---\n")
    print(sample_prompt)


def run(
    model_key: str,
    corpus: Path = _CORPUS_ROOT,
    out_dir: Path = DEFAULT_OUT_DIR,
    checkpoint_dir: Path | None = None,
    limit: int | None = None,
    resume_run_id: str | None = None,
) -> dict:
    """Runs the full six-stage graph to completion (or resumes one already in
    progress) and returns the assembled, contract-verified output dict. Does
    not write it to disk -- `main()` owns file I/O, this owns orchestration,
    so tests can call `run()` against a temp checkpoint dir without needing
    `main()`'s CLI-parsing and argv handling in the way. `checkpoint_dir` is a
    parameter (not hardcoded to the module constant) for exactly that reason
    -- pipeline/tests points it at a pytest tmp_path rather than monkeypatching
    a module-level constant.

    Deliberately `None`, not `= CHECKPOINT_DIR`, as the default: a mutable
    module-level constant bound as a default *value* is captured once, at
    function-definition time -- a test that reassigns `graph.CHECKPOINT_DIR`
    after import (to redirect into a tmp dir) would silently be ignored, and
    write into the real `results/raw/.checkpoints/` anyway. Resolving inside
    the function body keeps the lookup live.
    """
    if model_key not in MODELS:
        raise ValueError(f"unknown model {model_key!r} -- must be one of {sorted(MODELS)}")
    if checkpoint_dir is None:
        checkpoint_dir = CHECKPOINT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    run_id = resume_run_id or _new_run_id()
    raw_log_path = _raw_log_path(run_id, model_key, out_dir)
    checkpoint_path = checkpoint_dir / f"{run_id}.sqlite"

    builder = build_graph_definition()
    with SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": run_id}}

        if resume_run_id is not None and graph.get_state(config).values:
            print(f"resuming run {run_id} from checkpoint {checkpoint_path}")
            print(f"  next stage: {graph.get_state(config).next}")
            result = graph.invoke(None, config=config)
        else:
            documents = load_documents(corpus, limit=limit)
            if not documents:
                raise SystemExit(f"no documents under {corpus}")
            print(f"corpus: {len(documents)} documents -> run_id={run_id}")
            initial_state: P1State = {
                "documents": documents,
                "model_key": model_key,
                "run_id": run_id,
                "raw_log_path": str(raw_log_path),
                "usage": [],
            }
            result = graph.invoke(initial_state, config=config)

    return result["output"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the P1 six-stage decomposed pipeline.")
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--corpus", type=Path, default=_CORPUS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap files per subdirectory (smoke test only -- not for a reported run).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print per-stage call counts and Stage 1 token estimates; make no API call.",
    )
    parser.add_argument(
        "--resume",
        metavar="RUN_ID",
        default=None,
        help="Resume a previous run's checkpoint instead of starting a new one.",
    )
    args = parser.parse_args(argv)

    spec = MODELS[args.model]

    if args.dry_run:
        documents = load_documents(args.corpus, limit=args.limit)
        if not documents:
            raise SystemExit(f"no documents under {args.corpus}")
        _dry_run(spec, documents)
        return

    output = run(
        model_key=args.model,
        corpus=args.corpus,
        out_dir=args.out_dir,
        limit=args.limit,
        resume_run_id=args.resume,
    )

    run_id = output["metadata"]["run_id"]
    out_path = args.out_dir / f"{run_id}_p1_{args.model}.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    n_attributes = sum(len(c["attributes"]) for c in output["classes"])
    skipped = output["metadata"]["extraction_skipped"]
    print(
        f"induced: {len(output['classes'])} classes, {n_attributes} attributes, "
        f"{len(output['relations'])} relations"
        + (f" ({skipped} extraction calls skipped)" if skipped else "")
    )
    print(f"wrote {out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
