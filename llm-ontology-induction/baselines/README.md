# Baselines

Two baselines the paper's contribution is measured against. Every frozen decision
behind both lives in [`DECISIONS.md`](DECISIONS.md); this file is only a map and a
set of commands.

```
baselines/
├── DECISIONS.md          # the record. Read this before changing anything here.
├── b1_statistical/       # B1 -- statistical, zero-LLM
├── b3_single_shot/       # B3 -- single-shot LLM, plus its frozen extraction prompt
├── shared/                # model-calling code and read-only viewers, shared with pipeline/
└── tests/                 # validation suites for both
```

**P1, the paper's actual contribution, moved to top-level [`pipeline/`](../pipeline/)
on 2026-08-21** — it isn't a baseline the way B1 and B3 are, and living under
`baselines/` stopped being accurate once that distinction was made explicit. Its
own two-stage predecessor is still documented in this file's `DECISIONS.md` (P1-D1
through P1-D5), marked superseded rather than deleted; see `pipeline/DECISIONS.md`
for the six-stage architecture that replaced it and `pipeline/README.md` for how to
run it.

## Model matrix (current, as of 2026-08-21)

| Condition | Models | Provider | Calls |
|---|---|---|---|
| B1 | none — statistical (C-value + dependency-parse SVO) | — | 0 (no LLM) |
| B3 | Haiku 4.5, Opus 5, Llama 3.1 8B | Haiku + Llama via AWS Bedrock; Opus 5 via Anthropic (direct API) | 1 whole-corpus call per model |

P1's own matrix (same three models, same transports) is documented in
[`pipeline/README.md`](../pipeline/README.md), not repeated here.

**Mixed transport, deliberately.** The rule of thumb: the budget and open-weight
conditions run on AWS Bedrock, the frontier condition runs on the direct Anthropic
API. Groq is retired — it existed only to route around a free-tier rate limit for
Llama, and that problem disappears once the same weights are reachable on Bedrock
instead. See `DECISIONS.md` B3-D1 (revised three times) for the full history and
what each move cost.

**Three conditions, not two, and not the three you might expect.** `opus48` (Opus
4.8) was briefly wired up alongside `opus5` early in this rework, then dropped when
the brief narrowed to Opus 5 only. Llama was dropped entirely in the mixed-transport
correction, then restored on Bedrock — the same condition, same transport, same
4,096-token cap as the original five-model grid. Neither change is deleted from
history — see `DECISIONS.md`.

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
| `llama318b` | `us.meta.llama3-1-8b-instruct-v1:0` | AWS Bedrock (native Meta shape) | open-weight | 4096 — the model card's documented ceiling | `temperature=0.0`, `top_p=1.0` — both accepted together |

Opus 5 rejects `temperature`/`top_p` outright (a documented Anthropic API constraint
on Opus 4.7+, not a per-model choice) — see B3-D3 for what replaces sampling control
on it. Haiku's Bedrock constraint is separate and narrower: `temperature` alone is
fine, but the request 400s if `top_p` is present alongside it — confirmed empirically,
not documented, see B3-D3 (revised). Llama's Bedrock request/response shape is a
third thing entirely — native `prompt`/`max_gen_len` in, `generation`/`stop_reason`
out, nothing like either Anthropic transport — see B3-D3's third revision.

### Running it

```bash
# Costs nothing, calls nothing. Always do this first.
python3 -m baselines.b3_single_shot.single_shot --model opus5 --dry-run

# Smoke test: caps files per subdirectory. Not a reportable run.
python3 -m baselines.b3_single_shot.single_shot --model haiku45 --limit 2

# A real whole-corpus run.
python3 -m baselines.b3_single_shot.single_shot --model opus5
python3 -m baselines.b3_single_shot.single_shot --model llama318b
```

Credentials: `opus5` needs `ANTHROPIC_API_KEY` — get one at
[console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys).
`haiku45` and `llama318b` both need AWS credentials resolvable by boto3's default
chain (`AWS_PROFILE` and `AWS_REGION` in `.env`, or your usual `~/.aws` setup) with
Bedrock access to the Haiku 4.5 Geo inference profile and the Llama 3.1 8B Geo
inference profile respectively. `pipeline/` (P1) reads the same `.env` and needs
the same credential sets — see its own README for its commands.

## Scoring and viewing

```bash
python3 -m eval.report --gold schema/gold_schema.ttl --induced results/raw/<run_id>_<condition>_<model>.json --level all
python3 -m baselines.shared.show_results
```

## Tests

```bash
python3 -m pytest baselines/tests -q
```

`pipeline/tests` (P1) is a separate suite, run the same way — see `pipeline/README.md`.

Every test runs offline — no network call of any kind. Some of them are not
ordinary regression tests but tripwires on the rules that make these conditions
valid (no gold-schema vocabulary, frozen prompts, no pre-cleaning, a truncated or
refused response never scored). Failing one of those does not mean the code got
worse; it means the baseline stopped being the thing the paper claims it is.
