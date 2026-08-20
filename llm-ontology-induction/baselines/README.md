# Baselines

Two baselines and a staged pipeline the paper compares against each other. Every
frozen decision behind all three lives in [`DECISIONS.md`](DECISIONS.md); this file
is only a map and a set of commands.

```
baselines/
├── DECISIONS.md          # the record. Read this before changing anything here.
├── b1_statistical/       # B1 -- statistical, zero-LLM
├── b3_single_shot/       # B3 -- single-shot LLM, plus its frozen extraction prompt
├── p1_pipeline/           # P1 -- staged pipeline: per-doc extraction + LLM consolidation
├── shared/                # model-calling code and read-only viewers used by all three
└── tests/                 # validation suites for all three
```

## Model matrix (current, as of 2026-08-19)

| Condition | Models | Provider | Calls |
|---|---|---|---|
| B1 | none — statistical (C-value + dependency-parse SVO) | — | 0 (no LLM) |
| B3 | Haiku 4.5, Opus 5 | Haiku via AWS Bedrock; Opus 5 via Anthropic (direct API) | 1 whole-corpus call per model |
| P1 | Haiku 4.5, Opus 5 | Haiku via AWS Bedrock; Opus 5 via Anthropic (direct API) | N per-doc extraction calls + 1 consolidation call per model |

**Mixed transport, deliberately.** The rule of thumb: Haiku 4.5 runs on AWS Bedrock,
Opus 5 runs on the direct Anthropic API. Groq is retired — it existed only to route
around a free-tier rate limit, and that problem disappears once the account has its
own Anthropic API key for the frontier model. Bedrock stays, scoped to Haiku only.
The historical Groq run and the decisions that governed the earlier direct-API-only
attempt are not deleted — see `DECISIONS.md` B3-D1 (revised, twice) for the full
history and what each move cost.

**Two conditions, not three.** An earlier revision of this rework also wired up
`opus48` (Opus 4.8) alongside `opus5`, since the brief that drove it named "Opus
4.8/5" without disambiguating. A later correction narrowed the grid to Opus 5 only;
`opus48` is dropped from the active registry (not deleted from history — see
`DECISIONS.md`).

## B1 — statistical, no LLM

C-value term extraction, a frequency-and-position class/attribute split, and
dependency-parse SVO triples. No model, no API key, no network. It is the floor:
whatever an LLM does, it has to beat word counting.

```bash
python3 -m baselines.b1_statistical.statistical
```

## B3 — single-shot LLM

Hands a model the raw corpus under one frozen prompt, in **one call**, and asks
once for a schema. Merges with nothing cleverer than dropping malformed elements
and collapsing literal case/whitespace duplicates within that one response —
consolidating across *wordings* is P1's job, not B3's (B3-D4).

### Conditions

| `--model` | Provider ID | Transport | Tier | Output cap | Sampling |
|---|---|---|---|---|---|
| `haiku45` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | AWS Bedrock | budget | 16000 | `temperature=0.0` only — Bedrock 400s if `top_p` is also present |
| `opus5` | `claude-opus-5` | Anthropic (direct API) | frontier | 32000 | none — rejected by the API; adaptive thinking, `effort=high` instead |

Opus 5 rejects `temperature`/`top_p` outright (a documented Anthropic API constraint
on Opus 4.7+, not a per-model choice) — see B3-D3 for what replaces sampling control
on it. Haiku's Bedrock constraint is separate and narrower: `temperature` alone is
fine, but the request 400s if `top_p` is present alongside it — confirmed empirically,
not documented, see B3-D3 (revised).

### Running it

```bash
# Costs nothing, calls nothing. Always do this first.
python3 -m baselines.b3_single_shot.single_shot --model opus5 --dry-run

# Smoke test: caps files per subdirectory. Not a reportable run.
python3 -m baselines.b3_single_shot.single_shot --model haiku45 --limit 2

# A real whole-corpus run.
python3 -m baselines.b3_single_shot.single_shot --model opus5
```

## P1 — staged pipeline

Reads the corpus one document at a time (N extraction calls, same frozen prompt
B3 uses, same document order), then makes **one more call** asking the model
itself to reconcile the N partial schemas into a single one — resolving
cross-wording the way B3-D4 deliberately refuses to. That reconciliation is the
thing P1 exists to test the value of.

### Conditions

Same two models, same transports, same output caps and sampling rules as B3
(above) — P1 reuses `baselines/shared/model_clients.py` unchanged; only the call
shape differs.

### Running it

```bash
python3 -m baselines.p1_pipeline.pipeline --model opus5 --dry-run

# Smoke test.
python3 -m baselines.p1_pipeline.pipeline --model haiku45 --limit 2

# A real run: 192 extraction calls + 1 consolidation call.
python3 -m baselines.p1_pipeline.pipeline --model opus5
```

Credentials: both baselines read from `.env` via `load_dotenv()`. `opus5` needs
`ANTHROPIC_API_KEY` — get one at
[console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
`haiku45` needs AWS credentials resolvable by boto3's default chain (`AWS_PROFILE`
and `AWS_REGION` in `.env`, or your usual `~/.aws` setup) with Bedrock access to
the Haiku 4.5 Geo inference profile.

## Scoring and viewing

```bash
python3 -m eval.report --gold schema/gold_schema.ttl --induced results/raw/<run_id>_<condition>_<model>.json --level all
python3 -m baselines.shared.show_results
```

## Tests

```bash
python3 -m pytest baselines/tests -q
```

Every test runs offline — no network call of any kind. Some of them are not
ordinary regression tests but tripwires on the rules that make these conditions
valid (no gold-schema vocabulary, frozen prompts, no pre-cleaning, a truncated or
refused response never scored). Failing one of those does not mean the code got
worse; it means the baseline stopped being the thing the paper claims it is.
