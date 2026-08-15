# Baselines

Two baselines the staged P1 pipeline has to beat to be worth having. Every frozen
decision behind both lives in [`DECISIONS.md`](DECISIONS.md); this file is only a map
and a set of commands.

```
baselines/
├── DECISIONS.md          # the record. Read this before changing anything here.
├── b1_statistical/       # B1 -- statistical, zero-LLM
├── b3_single_shot/       # B3 -- single-shot LLM, plus its frozen prompt
├── shared/               # read-only viewers used by both
└── tests/                # validation suites for both
```

## B1 — statistical, no LLM

C-value term extraction, a frequency-and-position class/attribute split, and
dependency-parse SVO triples. No model, no API key, no network. It is the floor:
whatever an LLM does, it has to beat word counting.

```bash
python3 -m baselines.b1_statistical.statistical
```

## B3 — single-shot LLM

Hands a model the raw corpus under one frozen prompt, asks once for a schema, and
merges with nothing cleverer than exact-string deduplication. Consolidating across
wordings is P1's Stage 6 novelty; if B3 did that job too, the B3-vs-P1 comparison
would measure nothing (B3-D4).

**The current shape is whole-corpus: all 192 documents in a single call.** Batching
was a Groq free-tier artifact, not a property of single-shot prompting (B3-D2).

### Conditions

| `--model` | Provider ID | Host | Output cap |
|---|---|---|---|
| `fable5` | `us.anthropic.claude-fable-5` | Bedrock | 16000 |
| `haiku45` | `us.anthropic.claude-haiku-4-5-20251001-v1:0` | Bedrock | 16000 |
| `sol` | `openai.gpt-5.6-sol` | Bedrock — **not runnable yet, see below** | 16000 |
| `luna` | `openai.gpt-5.6-luna` (`reasoning_effort="low"`) | Bedrock — **not runnable yet, see below** | 16000 |
| `llama318b_bedrock` | `us.meta.llama3-1-8b-instruct-v1:0` | Bedrock | **4096** |
| `llama318b_groq` | `llama-3.1-8b-instant` | Groq free tier | 2048 |

The open-weight model appears twice on purpose. The Groq entry reproduces the
2026-08-12 batched run, which B3-D6 keeps as the batched arm of a same-model
batched-vs-whole-corpus comparison. Llama's cap is lower than the rest because
Bedrock serves the model at a documented 4K ceiling — see B3-D3, which also says what
to do if a run hits it.

`fable5` and `haiku45` use the **Geo inference ID** (`us.` prefix), not the bare model
ID — Bedrock rejects the bare ID for on-demand invoke on both (confirmed for
`haiku45` at the first real call, an actual `ValidationException`; matched
proactively for `fable5` from the identical pattern in its model card).

**`sol` and `luna` do not work yet.** Both are `bedrock-mantle`-only — the model card
shows `bedrock-runtime` unsupported entirely (Invoke ✗, Converse ✗) — reachable only
through the Responses API via the `openai` SDK, a different request/response shape
than anything in `model_clients.py` today. `invoke()` raises `ValueError` for both
rather than doing the wrong thing silently, but there is no working backend for them.

### Running it

```bash
# Costs nothing, calls nothing. Always do this first.
python3 -m baselines.b3_single_shot.single_shot --model haiku45 --dry-run

# Smoke test: caps files per subdirectory. Not a reportable run.
python3 -m baselines.b3_single_shot.single_shot --model llama318b_bedrock --limit 2

# A real whole-corpus run.
python3 -m baselines.b3_single_shot.single_shot --model llama318b_bedrock

# The legacy batched shape, only for reproducing the historical Groq run.
python3 -m baselines.b3_single_shot.single_shot --model llama318b_groq --batch-size 7
```

Credentials: Bedrock uses the standard AWS chain (`AWS_REGION` selects the region
unless a spec overrides it); Groq reads `GROQ_API_KEY` from `.env`.

## Scoring and viewing

```bash
python3 -m eval.report --induced results/raw/<run_id>_b3_<model>.json --level all
python3 -m baselines.shared.show_results
```

## Tests

```bash
python3 -m pytest baselines/tests -q
```

Every test runs offline — no Bedrock call, no Groq call, no network of any kind. Some
of them are not ordinary regression tests but tripwires on the rules that make these
baselines valid (no gold-schema vocabulary, one frozen prompt, naive consolidation,
batching that cannot depend on the model). Failing one of those does not mean the code
got worse; it means the baseline stopped being the thing the paper claims it is.
