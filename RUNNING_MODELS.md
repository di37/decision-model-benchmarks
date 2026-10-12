# Running the four decision models

All four models speak the same native SystemOne protocol: `POST /v1/systemone`
with a `state` and typed `questions`, answered with calibrated probabilities.
Jev is a hosted API; CLEF-Flash, Laya, and CLM run locally in Docker using the
compose setups in `docker/`.

| Model | Where it runs | Endpoint | GPU memory used here | Weights |
| --- | --- | --- | --- | --- |
| Jev | TypeSafe hosted API | `https://api.typesafe.ai/v1/systemone` | — | — |
| CLEF-Flash | `docker/clef-flash` | `http://127.0.0.1:8710/v1/systemone` | ~19 GB | [`Cloudflare/clef-flash`](https://huggingface.co/Cloudflare/clef-flash) |
| Laya | `docker/laya` | `http://127.0.0.1:8004/v1/systemone` | small (~1.5 GB of weights preloaded) | [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) |
| CLM | `docker/clm` | `http://127.0.0.1:8700/v1/systemone` | ~22 GB (encoder) | [`Contrastive-LM/CLM-v0.1-8B`](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) + [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) |

CLEF-Flash and CLM each need nearly a whole 24 GB GPU, so run **one local
model at a time**. Every compose file uses GPU 1 by default; set `BENCH_GPU`
to choose another, for example `BENCH_GPU=0 docker compose up -d`.

## Prerequisites

- Docker with Compose v2 and the NVIDIA Container Toolkit.
- An NVIDIA GPU with 24 GB of memory (the results were produced on an
  RTX 4090).
- Model weights under `models/weights/`, next to this project. Each compose
  file mounts `../../../models/weights/<name>`.
- Python 3.10+ with `pip install -r requirements.txt` to run the benchmark
  scripts.

### Downloading weights

With the Hugging Face CLI (`pip install -U huggingface_hub`), from the
directory that contains this project:

```bash
hf download Cloudflare/clef-flash --local-dir models/weights/clef-flash
hf download convaiinnovations/laya --local-dir models/weights/laya
hf download Contrastive-LM/CLM-v0.1-8B --local-dir models/weights/CLM-v0.1-8B
hf download Qwen/Qwen3-8B --local-dir models/weights/Qwen3-8B
```

The Laya download must include the `multilingual/` and `typed-decisions/`
subfolders as well as the English root checkpoint; its Router chooses among
them per request.

## A request to try

Save this as `/tmp/request.json`. It is used in the health-check examples
below.

```json
{
  "state": "Hi, I was charged twice for my March invoice. Please refund the duplicate today.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, errors",
        "other": "everything else"
      }
    },
    "refund": {"type": "noul", "instructions": "Does the customer ask for a refund?"},
    "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "today"]}
  }
}
```

---

## Jev (hosted)

Jev needs no local setup, only an API key.

1. Put your key in `.env`:

   ```bash
   cp .env.example .env
   # then set JEV_API_KEY=...
   ```

2. Send a request:

   ```bash
   curl -s https://api.typesafe.ai/v1/systemone \
     -H "Authorization: Bearer $JEV_API_KEY" \
     -H "Content-Type: application/json" \
     -d "$(jq '. + {model: "jev-latest"}' /tmp/request.json)"
   ```

Every Jev request uses API credits; a full benchmark run sends about 700.
Jev's latency includes the network round trip, so it is not directly
comparable with the local models.

---

## CLEF-Flash

CLEF-Flash is a Qwen3.5-9B backbone with a joint schema head. The server
(`docker/clef-flash/server.py`) loads it with the model's own
`load_release_model` and answers through its `systemone` function. Because
those come from `joint_schema_model.py` in the downloaded weights, the server
runs code shipped with the model; review that file after each download. The
image builds on the Laya image, so build Laya first.

```bash
(cd docker/laya && docker compose build)    # once: base image
cd docker/clef-flash
docker compose up -d --build
```

Wait until it is healthy (usually under a minute):

```bash
docker inspect -f '{{.State.Health.Status}}' bench-clef-flash   # healthy
curl -s http://127.0.0.1:8710/health
# {"status":"ok","model":"clef-flash","device":"cuda:0","cuda_available":true}
```

Try a request:

```bash
curl -s http://127.0.0.1:8710/v1/systemone \
  -H "Content-Type: application/json" \
  -d "$(jq '. + {model: "clef-flash"}' /tmp/request.json)"
```

Stop it when finished:

```bash
docker compose down
```

To require a key, set `CLEF_API_KEY` in the compose `environment` and
`CLEF_FLASH_API_KEY` in `.env`.

---

## Laya

Laya is served by its official `laya.serve` app. `docker/laya/server.py` only
points Laya's Router at the local checkpoints and preloads the English and
multilingual ones on the GPU.

```bash
cd docker/laya
docker compose up -d --build
docker inspect -f '{{.State.Health.Status}}' bench-laya        # healthy
curl -s http://127.0.0.1:8004/health
# {"status":"ok","loaded":["english","multilingual"],"device":"cuda",...}
```

Try a request. Omit `model` so the Router chooses the checkpoint (sending
`"model": "laya"` pins the English one). The benchmark adds the raised token
budget below to every Laya request; include it here to reproduce benchmark
behavior:

```bash
curl -s http://127.0.0.1:8004/v1/systemone \
  -H "Content-Type: application/json" \
  -d "$(jq '. + {max_len: 2048, head_max_len: 1024}' /tmp/request.json)"
```

The response's `routing.model` shows which checkpoint answered, and
`usage.options` appears when the token budget merged some options, which is a
sign `head_max_len` is too small.

Stop it when finished:

```bash
docker compose down
```

Configuration lives in `.env`: `LAYA_MAX_LEN` and `LAYA_HEAD_MAX_LEN` set the
budget (2048 and 1024 by default; leave either blank to use Laya's own
default), and `LAYA_MODEL` pins a checkpoint (`english`, `multilingual`, or
`typed-decisions`).

---

## CLM

CLM runs as two containers: a vLLM server that produces Qwen3-8B last-token
embeddings (`bench-clm-encoder`, port 8090) and the `clm-serve` API with the
CLM projection heads (`bench-clm-api`, port 8700). The API waits for the
encoder to become healthy, which can take a few minutes on first start.

```bash
cd docker/clm
docker compose up -d --build
docker inspect -f '{{.State.Health.Status}}' bench-clm-api     # healthy
curl -s http://127.0.0.1:8700/health
# {"ok":true,"embedder":true,"models":["clm-latest","clm-raw"],"cache":null}
```

Try a request:

```bash
curl -s http://127.0.0.1:8700/v1/systemone \
  -H "Content-Type: application/json" \
  -d "$(jq '. + {model: "clm-latest"}' /tmp/request.json)"
```

Stop both containers when finished:

```bash
docker compose down
```

`clm-latest` uses the trained heads; `clm-raw` skips them and compares raw
encoder embeddings. CLM keeps a cache of answers across requests, so restart
it (`docker compose restart`) before a benchmark run.

---

## Running the benchmark against a model

With one model's server up (or with `JEV_API_KEY` set for Jev), run the three
benchmark scripts for that provider. Use `jev`, `clef-flash`, `laya`, or
`clm`:

```bash
python3 scripts/run_public_benchmarks.py --providers laya
python3 scripts/run_model_routing.py --providers laya
python3 scripts/run_tool_selection.py --providers laya
```

Then stop that server, start the next one, and repeat. When all four are
done, regenerate the statistics, notebook, figures, and tables:

```bash
python3 scripts/analyze_public_benchmarks.py
python3 scripts/build_results_notebook.py
jupyter nbconvert --to notebook --execute --inplace decision_model_benchmarks_results.ipynb
```

Useful options for every runner:

- `--limit 2 --output-dir /tmp/smoke`: a quick smoke test that does not touch
  the saved results.
- `--force`: replace the selected provider's saved rows instead of resuming.
  With `--limit`, only those first N cases are replaced.

Without `--force`, a rerun skips saved rows but retries any row that recorded
an error.

For a fair rerun, restart each local server just before its run, run the local
models one at a time on the same GPU, and keep the settings in `.env`
unchanged. The runners record each provider's settings and server health in
`results/*/run_manifest.jsonl`.

## Troubleshooting

- **Port already in use:** another server is on 8004, 8700, 8090, or 8710.
  Stop it first; `docker ps` shows which container holds the port.
- **CUDA out of memory:** another model is still on the GPU. Check with
  `nvidia-smi` and stop it, or choose a free GPU with `BENCH_GPU`.
- **CLEF-Flash build fails with a missing `local/laya-server` image:** build
  Laya first with `(cd docker/laya && docker compose build)`.
- **Laya answers but `usage.options` reports merged options:** raise
  `LAYA_HEAD_MAX_LEN` (and `LAYA_MAX_LEN` so the state still fits).
- **CLM API stays unhealthy:** the encoder is still loading or failed; check
  `docker logs bench-clm-encoder`.
