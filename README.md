# Decision-model benchmarks: CLEF-Flash vs Jev vs Laya vs CLM

This project compares four SystemOne decision models, **CLEF-Flash**, **Jev**,
**Laya**, and **CLM**, by sending each one identical native `/v1/systemone`
requests on five suites: three deterministic public benchmark subsets, a
90-case model-routing set, and the official 200-case BFCL v3 multiple split
adapted for tool selection. Prepared inputs and per-case outputs are committed
so every analysis can be reproduced without calling any provider.

## Saved headline results

| Suite (primary metric) | CLEF-Flash | Jev | Laya | CLM | Chance / random |
| --- | ---: | ---: | ---: | ---: | ---: |
| Banking77 inbox triage (accuracy) | **93.5%** (144/154) | 80.5% (124/154) | 45.5% (70/154) | 1.3% (2/154) | 1.3% |
| Amazon ESCI reranking (nDCG@10) | 0.9139 | **0.9183** | 0.8553 | 0.8012 | 0.7940 |
| MMLU (accuracy) | 92.5% (185/200) | **93.5%** (187/200) | 34.5% (69/200) | 29.5% (59/200) | 25.0% |
| Model routing (accuracy) | 96.7% (87/90) | **97.8%** (88/90) | 47.8% (43/90) | 35.6% (32/90) | 33.3% |
| BFCL tool selection (accuracy) | 98.5% (197/200) | **100.0%** (200/200) | 94.0% (188/200) | 65.5% (131/200) | — |

Pairwise tests are Holm-adjusted across the six provider pairs in each suite:

- **CLEF-Flash vs Jev:** CLEF-Flash is significantly better on Banking77
  (McNemar exact p = 0.000011). Jev is slightly ahead on ESCI, MMLU, model
  routing, and BFCL, but none of those margins is significant (Holm p = 0.84,
  0.82, 1.0, and 0.25).
- **Top tier vs Laya and CLM:** CLEF-Flash and Jev are each significantly
  better than Laya and CLM on every suite.
- **Laya vs CLM:** Laya is significantly better on Banking77, ESCI, and BFCL.
  The two are not distinguishable on MMLU (Holm p = 0.64) or routing
  (Holm p = 0.22), where both are close to chance.

Median client-side latency was about 8–20 ms for Laya, 26–69 ms for CLM,
69–323 ms for CLEF-Flash, and 309–327 ms for Jev, whose figures include the
network round trip to its hosted API.

## Fairness protocol

- **Identical requests.** Every provider receives the same `state`,
  `questions`, option text, and deterministic option order. The only
  provider-specific fields are each vendor's model id and Laya's documented
  token-budget fields (below). Provider handling lives in one module,
  `scripts/common.py`.
- **Identical client.** One sequential request at a time, a 120 s timeout,
  3 attempts with the same backoff, and one warm-up request that is not part of
  any dataset, so no server-side cache can answer a measured case.
- **Fresh servers on the same GPU.** The three local models ran one at a time
  on the same RTX 4090 (GPU 1), each from a freshly restarted server. Jev ran
  remotely in the same session.
- **Vendor-recommended configurations.**
  - *Laya* ran through its official `laya.serve` app (`docker/laya/`), with
    the Router choosing the checkpoint (English for every input here) and
    `head_max_len=1024`, `max_len=2048`. At Laya's suggested `head_max_len=512`,
    Banking77's 77 options collapse to 72 distinct spans; 1024 is the smallest
    tested budget at which Laya reports none merged.
  - *CLM* ran with `clm-serve` and the Qwen3-8B pooling encoder, using the
    command from its model card.
  - *CLEF-Flash* ran through its native SystemOne decision head.
- **Run manifest.** Each provider run appends its endpoint, model id, extra
  request fields, retry policy, `/health` response, and GPU state to
  `results/*/run_manifest.jsonl`.

## Benchmark design

- **Banking77 inbox triage:** 154 test messages, two from each of 77 intents.
  Primary metric: accuracy; secondary: macro-F1.
- **Amazon ESCI reranking:** 50 US queries with ten products each, including at
  least one `Exact` and one `Irrelevant`. Primary metric: nDCG@10 with grades
  Exact 3, Substitute 2, Complement 1, Irrelevant 0 and exponential gain
  2^grade − 1 (7/3/1/0); secondary: MRR of the first exact result.
- **MMLU:** 200 test questions, 20 from each of ten fixed subjects.
- **Model routing:** 90 curated prompts, 30 per expected `small`/`medium`/
  `large` tier. Metrics include per-tier recall, log loss, Brier score, and ECE.
- **BFCL v3 multiple tool selection:** all 200 cases; each request selects one
  function name from 2–4 candidate schemas.

Sampling uses seed `20261009`. Source hashes and sampling details are in
`data/public_benchmarks/manifest.json` and `data/tool_selection/manifest.json`.

## Setup

Python 3.10 or newer is recommended. On Debian/Ubuntu, `python3 -m venv`
requires the `python3-venv` package (for example,
`sudo apt install python3.12-venv`).

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
cp .env.example .env
```

Set `JEV_API_KEY` in `.env`. `.env` is ignored by git; never commit
credentials.

### Local servers

[`RUNNING_MODELS.md`](RUNNING_MODELS.md) walks through starting, testing,
benchmarking, and stopping each model. Each local model has a compose setup
under `docker/`. Run only one at a time
on the benchmark GPU, which defaults to GPU 1 (override with `BENCH_GPU`).

| Provider | Default endpoint | Compose directory | Containers |
| --- | --- | --- | --- |
| CLEF-Flash | `http://127.0.0.1:8710/v1/systemone` | `docker/clef-flash` | `bench-clef-flash` |
| Laya | `http://127.0.0.1:8004/v1/systemone` | `docker/laya` | `bench-laya` |
| CLM | `http://127.0.0.1:8700/v1/systemone` | `docker/clm` | `bench-clm-encoder`, `bench-clm-api` (encoder on port 8090) |

Weights are mounted from `../../../models/weights/` relative to each compose
directory, which is `models/weights/` next to this project:

- `clef-flash/`: the CLEF-Flash backbone, joint head, and
  `joint_schema_model.py`.
- `laya/`: the English root checkpoint plus the `multilingual/` and
  `typed-decisions/` subfolders.
- `Qwen3-8B/` and `CLM-v0.1-8B/CLM_v0.1-8B.pt`: CLM's encoder and heads.

The CLEF-Flash image builds on top of the Laya image, so build Laya first:

```bash
(cd docker/laya && docker compose build)
cd docker/clef-flash && docker compose up -d --build
# when finished with this provider:
docker compose down
```

Stop any other server already using a port (8004, 8700, 8090, or 8710) before
starting the benchmark container, and restart a server before each run so no
cached answers carry over.

## Run

Each runner resumes by skipping saved rows and rerunning any saved row that
recorded an error. `--force` reruns the selected provider's selected cases
(with `--limit`, only the first N cases of each suite), `--limit` runs a smoke
test, and `--output-dir` keeps smoke output separate. Run one provider at a
time while its server is up:

```bash
python3 scripts/run_public_benchmarks.py --providers laya
python3 scripts/run_model_routing.py --providers laya
python3 scripts/run_tool_selection.py --providers laya
```

Then regenerate the public-benchmark statistics (confidence intervals, chance
tests, MMLU subjects, all-pairs McNemar, and nDCG bootstrap with Holm
correction):

```bash
python3 scripts/analyze_public_benchmarks.py
python3 -m pytest -q
```

The routing and tool-selection runners regenerate their own summaries and
all-pairs McNemar tables from the saved rows at the end of every run.

## Results notebook

Open `decision_model_benchmarks_results.ipynb` from the repository root. It
loads the committed result files and shows the headline metrics, every
pairwise test, calibration and latency tables, the MMLU subject breakdown, and
per-tier routing recall. The notebook is generated by
`scripts/build_results_notebook.py`; executing it saves every figure and table:

```bash
python3 scripts/build_results_notebook.py
jupyter nbconvert --to notebook --execute --inplace decision_model_benchmarks_results.ipynb
```

| Folder | Files |
| --- | --- |
| `figures/` (PNG and PDF) | `primary_metrics`, `mmlu_by_subject`, `routing_by_tier` |
| `tables/` (CSV) | `primary_metrics`, `pairwise_classification_mcnemar`, `pairwise_reranking_bootstrap`, `calibration_and_latency`, `latency_p50_ms`, `log_loss`, `brier_score`, `mmlu_by_subject`, `routing_summary`, `tool_selection_summary` |

## Caveats

- **Diagnostic subsets, not leaderboard scores.** Banking77 has two examples
  per intent, ESCI reranks ten sampled products per query, MMLU covers ten of
  57 subjects, routing targets are curated judgments, and BFCL is scored on
  function-name selection only.
- **ESCI's random floor is high.** 218 of the 500 sampled products are
  `Exact`, so a random ordering already scores nDCG@10 0.794 and MRR 0.628.
  Read provider differences against that floor; CLM's 0.801 is close to
  random.
- **Routing option order is fixed.** The routing question always lists
  `small`, `medium`, `large` in that order, while the other suites shuffle
  options per case. Every provider saw the same order, so the comparison is
  fair, but position bias is not controlled.
- **Laya** was given a larger token budget than its default (192/512),
  following its guidance for many-option questions; defaults would score lower
  on Banking77. Its checkpoint warns that the calibration temperature for
  choices with 11+ options is invalid and was clamped, so its Banking77
  probabilities are uncalibrated.
- **CLM** was evaluated as the zero-shot `CLM-v0.1-8B` checkpoint; its model
  card says the published agentic results use fine-tuned heads. On Banking77
  it predicted only two of the 77 intents. Reproducing the card's own example
  gives the same decision but probability 0.988 instead of the documented
  0.939, so a small encoder numerics difference cannot be ruled out.
- **Jev** is a hosted model whose version can change between runs. Its
  Banking77 score moved from 79.2% in the earlier two-way run to 80.5% here;
  CLEF-Flash reproduced exactly. The routing and tool-selection rows record
  `jev-1.13.0`; the saved Banking77, ESCI, and MMLU rows were written before
  the runner stored a per-row model id, so their Jev version is known only
  from running in the same session.
- **Unrecorded versions.** The run manifests do not record model-weight
  revisions or the `accelerate` and `pillow` versions inside the CLEF-Flash
  image, which the Dockerfile leaves unpinned.
- **Latency** is not a deployment comparison: Jev's numbers include network
  time, and local numbers depend on this GPU and on request size (Banking77
  sends 77 options; BFCL sends full parameter schemas).
- Dataset licenses and upstream revisions remain governed by the named public
  sources.
