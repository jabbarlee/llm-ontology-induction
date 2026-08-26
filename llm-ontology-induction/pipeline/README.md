# P1 — Decomposed Pipeline

The paper's contribution, not a baseline — `baselines/` holds what P1 is measured
against. Every frozen decision here lives in [`DECISIONS.md`](DECISIONS.md); this
file is only a map and a set of commands.

```
pipeline/
├── DECISIONS.md          # the record. Read this before changing anything here.
├── state.py               # P1State -- the TypedDict threaded through every node
├── graph.py               # LangGraph assembly, checkpointing, the CLI
├── nodes/                 # one module per stage
│   ├── extract.py                # Stage 1 -- per-document extraction
│   ├── consolidate_types.py      # Stage 2 -- type consolidation (the novelty)
│   ├── consolidate_attrs.py      # Stage 3 -- attribute consolidation
│   ├── reconcile_relations.py    # Stage 4 -- relation reconciliation
│   ├── induce_taxonomy.py        # Stage 5 -- taxonomy induction (B3-impossible)
│   └── assemble.py               # Stage 6 -- assemble + validate, no model call
├── prompts/               # one frozen prompt per LLM stage (Stage 1's is B3's own, by reference)
└── tests/                 # validation suite
```

## The six stages

| # | Stage | LLM? | Calls |
|---|---|---|---|
| 1 | Per-document extraction | Yes | 192 |
| 2 | Type consolidation | Yes | 1 |
| 3 | Attribute consolidation | Yes | 1 |
| 4 | Relation reconciliation | Yes | 1 (0 if nothing survives endpoint remapping) |
| 5 | Taxonomy induction | Yes | 1 |
| 6 | Assemble + validate | No | 0 |

Stage 1 reuses B3's frozen extraction prompt by path reference, never a copy
(P1-D2) — the extraction stage is doing exactly what a B3 call does, just with
one document at a time. Stages 2-5 each get their own frozen prompt, scoped to
one job apiece, so a failure in one is never conflated with a failure in another
(P1-D1) — see `DECISIONS.md` for why this replaced the earlier two-stage design
(still documented, not deleted, in `baselines/DECISIONS.md`'s own P1-D1..D5).

## Conditions

Same three models and transports as B3 — this pipeline reads `baselines/shared/
model_clients.py`'s registry directly, so it always reflects whatever conditions
are actually frozen there rather than a hardcoded list (P1-D3).

| `--model` | Transport | Tier |
|---|---|---|
| `haiku45` | AWS Bedrock | budget |
| `opus5` | Anthropic (direct API) | frontier |
| `llama318b` | AWS Bedrock (native Meta shape) | open-weight |

## Running it

```bash
# Costs nothing, calls nothing. Always do this first -- per-stage call counts
# and Stage 1's real token estimate for the actual corpus.
python3 -m pipeline.graph --model haiku45 --dry-run
python3 -m pipeline.graph --model opus5 --dry-run
python3 -m pipeline.graph --model llama318b --dry-run

# Smoke test: caps files per subdirectory. Not a reportable run.
python3 -m pipeline.graph --model haiku45 --limit 8

# A real run: 192 extraction calls + up to 4 consolidation-stage calls.
python3 -m pipeline.graph --model opus5

# Resume a run that died partway through -- reconnects to its checkpoint and
# its existing raw-call log; already-paid-for calls are never repeated.
python3 -m pipeline.graph --model opus5 --resume 2026-08-21T20:38:13Z-4776
```

Credentials: same as B3 — `opus5` needs `ANTHROPIC_API_KEY`; `haiku45` and
`llama318b` both need AWS credentials with Bedrock access to their respective Geo
inference profiles. All read from the repo root's `.env`.

**`llama318b`'s 4,096-token output cap is worth watching across every LLM stage,
not only Stage 1** — Stage 2 (type consolidation) in particular hands the model
every partial schema's classes at once, which is the stage most likely to test
that ceiling. If it truncates, that's a real finding about this condition under
decomposition, not a bug — see `DECISIONS.md`'s P1-D3 revision.

## Checkpointing

Each run gets its own sqlite checkpoint file at
`results/raw/.checkpoints/<run_id>.sqlite`, keyed by `run_id` as the LangGraph
`thread_id` — one file per run, not one shared database, so a checkpoint's
lifetime is legible from its filename alone. The raw per-call log
(`<run_id>_p1_<model>_calls.jsonl`) is appended to incrementally, one line per
call as it resolves, not batched at the end — a call already paid for survives a
crash on the very next line of code, and a `--resume` run reopens the same file
in append mode rather than starting a new one.

## Scoring and viewing

```bash
python3 -m eval.report --gold schema/gold_schema.ttl --induced results/raw/<run_id>_p1_<model>.json --level all
python3 -m baselines.shared.show_results
```

## Tests

```bash
python3 -m pytest pipeline/tests -q
```

Every test runs offline — no network call of any kind, no live LangGraph
checkpoint left behind (in-memory or temp-dir checkpointers only). Some are
tripwires on the rules that make this pipeline valid, not ordinary regression
tests: no gold-schema vocabulary in any of the four new prompts or in any node
module (Critical Rule 1), Stage 1's prompt genuinely imported from B3 rather
than copied (P1-D2), a dangling relation or taxonomy endpoint dropped rather
than assembled (P1-D1, P1-D6), Stage 5 defaulting to `null` without concrete
support (P1-D6). Failing one of those does not mean the code got worse; it
means the pipeline stopped being the thing the paper claims it is.
