# RiskMiner experiment visualization and hyperparameter sweeps

RiskMiner uses two complementary local-first tools:

1. **MLflow** is the primary experiment tracker. It shows live scalar metrics,
   system CPU/memory, run parameters, replay tables, policy checkpoints and final
   reports. Every standalone RiskMiner run is one MLflow run. During an Optuna
   sweep, one MLflow parent run contains one child run per Optuna trial.
2. **Optuna + Optuna Dashboard** owns hyperparameter optimization. It provides TPE
   sampling, pruning, optimization history, parameter importance and parameter
   interaction/contour views. The default local study uses JournalStorage, which
   is safe for multiple processes on one host.

The complete token-level MCTS stream remains in `OUTPUT_DIR/trace/events_*.jsonl`.
It is also uploaded to the MLflow run as an artifact. This keeps the lossless
trace without flooding MLflow with one metric row per token decision.

## Install

The packages are normal project dependencies now, so from the repo root:

```bash
pip install -e .
```

installs `mlflow`, `optuna`, `optuna-dashboard`, and `psutil`.

## Live MLflow dashboard for one RiskMiner run

Start the server first:

```bash
mlflow server \
  --backend-store-uri sqlite:////tmp/riskminer-inputdata/mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

Then in another terminal:

```bash
RISKMINER_MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
PYTHONPATH=src python scripts/run_riskminer_inputdata.py
```

Open `http://127.0.0.1:5000`. Useful metric groups include:

- `iteration/*`: pool Sharpe/size, reward quantile, reward distribution.
- `pool/*`: every terminal candidate's resulting score and additive delta.
- `candidate/*`: max/mean/min orthogonal candidate score per candidate batch.
- `mcts/*`: tree nodes, archive size, invalid rollouts and pool updates.
- `episode/*`: trajectory reward and formula length.
- `policy/*`: GRU loss and risk-seeking quantile evolution.
- `system/*`: CPU and memory from MLflow system metrics.
- `timing/*`: cpp_stream compile/native/run timing for pool and orthogonal work.

The Artifacts tab contains replay snapshots, policy checkpoints, the final report,
and the full JSONL trace.

If no server is running, the default tracker writes directly to
`sqlite:///<OUTPUT_DIR>/mlflow.db`. You can start the server later against that DB.

## Run an Optuna sweep

A standard local sweep is:

```bash
RISKMINER_SWEEP_TRIALS=30 \
RISKMINER_ROWS=500000 \
RISKMINER_MAX_DEPTH=8 \
PYTHONPATH=src python scripts/sweep_riskminer.py
```

The sweep optimizes **validation pool Sharpe**, never final test Sharpe. The
standard preset searches simulations, rollouts, PUCT exploration, progressive
widening K/alpha, rollout END probability, policy learning rate, risk quantile,
quantile learning rate, replay capacity and policy batch size.

Presets:

```bash
RISKMINER_SWEEP_PRESET=quick     # cheap range-finding
RISKMINER_SWEEP_PRESET=standard  # default
RISKMINER_SWEEP_PRESET=wide      # expensive broad search
```

The Optuna study is persistent. By default it is stored at:

```text
/tmp/riskminer-sweeps/riskminer.journal
```

so rerunning the sweep continues the study rather than forgetting prior trials.

## Optuna Dashboard

While the sweep is running, in another terminal:

```bash
optuna-dashboard /tmp/riskminer-sweeps/riskminer.journal
```

Open the URL printed by Optuna Dashboard (normally port 8080). The most useful
views are optimization history, hyperparameter importance, parallel coordinates,
and contour/interaction plots.

## MLflow view of the same sweep

The sweep's default MLflow DB is:

```text
/tmp/riskminer-sweeps/mlflow.db
```

Serve it with:

```bash
mlflow server \
  --backend-store-uri sqlite:////tmp/riskminer-sweeps/mlflow.db \
  --host 127.0.0.1 \
  --port 5000
```

The top-level `sweep-riskminer` run is the parent. Each Optuna trial is a child
run containing its complete RiskMiner time-series metrics and artifacts.

## Parallel trials

```bash
RISKMINER_SWEEP_JOBS=4 PYTHONPATH=src python scripts/sweep_riskminer.py
```

The default JournalStorage supports concurrent trials on one host. Each trial is
itself a substantial cpp_stream/JAX workload, so parallel trials can be slower if
they compete for memory bandwidth or compiler RAM. When `SWEEP_JOBS > 1` and you
have not explicitly set `RISKMINER_THREADS`, the sweep script divides visible CPUs
among trials to reduce oversubscription.

Sequential sweeps share derived `roll_rets` and `vol` files. Parallel sweeps do
not share first-time derived materialization to avoid file races.

For multi-machine sweeps set `RISKMINER_SWEEP_STORAGE_URL` to a shared RDB URL
(PostgreSQL/MySQL is preferable) rather than the local JournalStorage file.

## Pruning

Median pruning is enabled by default after a few startup trials. The sweep reads
`depth_iteration_done` records from each child process and reports the current
validation pool Sharpe to Optuna. A clearly weak trial can therefore be stopped
before it reaches every requested depth.

Disable this while diagnosing search behavior with:

```bash
RISKMINER_SWEEP_PRUNING=0 PYTHONPATH=src python scripts/sweep_riskminer.py
```

## Test the dashboards without market data

```bash
RISKMINER_SWEEP_DRY_RUN=1 \
RISKMINER_SWEEP_TRIALS=10 \
PYTHONPATH=src python scripts/sweep_riskminer.py
```

This exercises Optuna + MLflow only with a tiny synthetic objective. It is useful
for checking the dashboards and storage paths before launching expensive searches.
