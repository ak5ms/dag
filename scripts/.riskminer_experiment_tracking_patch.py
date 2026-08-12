from pathlib import Path
import textwrap

ROOT = Path('.')

tracking = r'''from __future__ import annotations

"""Experiment tracking adapters for RiskMiner.

MLflow stores low-frequency scalar progress, system metrics, replay snapshots,
policy checkpoints and final artifacts. The complete event stream is always kept
as JSONL because per-token MCTS traces are much higher cardinality than useful
MLflow scalar metrics.
"""

from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Mapping


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _serialize_param(value: object) -> object:
    if value is None:
        return "<none>"
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


@dataclass(frozen=True)
class MLflowTrackerConfig:
    enabled: bool
    tracking_uri: str
    experiment_name: str
    run_name: str | None = None
    parent_run_id: str | None = None
    log_system_metrics: bool = True
    system_metrics_interval: int = 5
    tags: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_env(cls, output_dir: str | Path) -> "MLflowTrackerConfig":
        output = Path(output_dir).resolve()
        default_uri = f"sqlite:///{output / 'mlflow.db'}"
        tags_raw = os.environ.get("RISKMINER_MLFLOW_TAGS_JSON", "").strip()
        tags: dict[str, object] = {}
        if tags_raw:
            parsed = json.loads(tags_raw)
            if not isinstance(parsed, dict):
                raise ValueError("RISKMINER_MLFLOW_TAGS_JSON must decode to an object")
            tags = parsed
        return cls(
            enabled=_truthy(os.environ.get("RISKMINER_MLFLOW_ENABLED"), default=True),
            tracking_uri=os.environ.get("RISKMINER_MLFLOW_TRACKING_URI", default_uri),
            experiment_name=os.environ.get("RISKMINER_MLFLOW_EXPERIMENT", "riskminer"),
            run_name=os.environ.get("RISKMINER_MLFLOW_RUN_NAME", "").strip() or None,
            parent_run_id=(
                os.environ.get("RISKMINER_MLFLOW_PARENT_RUN_ID", "").strip() or None
            ),
            log_system_metrics=_truthy(
                os.environ.get("RISKMINER_MLFLOW_SYSTEM_METRICS"), default=True
            ),
            system_metrics_interval=max(
                1,
                int(os.environ.get("RISKMINER_MLFLOW_SYSTEM_METRICS_INTERVAL", "5")),
            ),
            tags=tags,
        )


class RiskMinerExperimentTracker:
    """Composite experiment sink: lossless JSONL trace + MLflow summaries."""

    def __init__(self, config: MLflowTrackerConfig, output_dir: str | Path) -> None:
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_dir = self.output_dir / "trace"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path: Path | None = None
        self.trace_file = None
        self.mlflow = None
        self.active_run = None
        self.run_id: str | None = None
        self.started = time.time()
        self.counters: dict[str, int] = {}

    def _next(self, name: str) -> int:
        value = self.counters.get(name, 0) + 1
        self.counters[name] = value
        return value

    def start(
        self,
        *,
        params: Mapping[str, object],
        tags: Mapping[str, object] | None = None,
    ) -> str | None:
        suffix = str(int(self.started * 1000))
        if self.config.enabled:
            import mlflow
            from mlflow.system_metrics import set_system_metrics_sampling_interval

            self.mlflow = mlflow
            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            if self.config.log_system_metrics:
                set_system_metrics_sampling_interval(self.config.system_metrics_interval)
            self.active_run = mlflow.start_run(
                run_name=self.config.run_name,
                parent_run_id=self.config.parent_run_id,
                log_system_metrics=self.config.log_system_metrics,
            )
            self.run_id = self.active_run.info.run_id
            suffix = self.run_id
            mlflow.log_params(
                {str(key): _serialize_param(value) for key, value in params.items()}
            )
            merged_tags = {
                "riskminer.framework": "cpp_stream",
                **{str(k): str(v) for k, v in self.config.tags.items()},
                **{str(k): str(v) for k, v in (tags or {}).items()},
            }
            if sha := _git_sha():
                merged_tags["git.commit"] = sha
            mlflow.set_tags(merged_tags)
        self.trace_path = self.trace_dir / f"events_{suffix}.jsonl"
        self.trace_file = self.trace_path.open("w", encoding="utf-8", buffering=1)
        return self.run_id

    def _write_trace(self, event: str, payload: Mapping[str, object]) -> None:
        if self.trace_file is None:
            return
        record = {
            "timestamp": time.time(),
            "elapsed_seconds": time.time() - self.started,
            "event": event,
            **dict(payload),
        }
        self.trace_file.write(json.dumps(record, sort_keys=True, default=str) + "\n")

    def _log_metrics(self, metrics: Mapping[str, object], *, step: int) -> None:
        if self.mlflow is None or self.active_run is None:
            return
        clean = {
            str(key): value
            for key, raw in metrics.items()
            if (value := _finite(raw)) is not None
        }
        if clean:
            self.mlflow.log_metrics(clean, step=int(step))

    def emit(self, event: str, payload: Mapping[str, object]) -> None:
        payload = dict(payload)
        self._write_trace(event, payload)
        if self.mlflow is None or self.active_run is None:
            return

        if event == "mcts_simulation_done":
            step = self._next("simulation")
            self._log_metrics(
                {
                    "mcts/tree_nodes": payload.get("tree_nodes"),
                    "mcts/archive_size": payload.get("archive_size"),
                    "mcts/invalid_rollouts": payload.get("invalid_rollouts"),
                    "mcts/pool_updates": payload.get("pool_updates"),
                },
                step=step,
            )
        elif event == "mcts_episode_done":
            step = self._next("episode")
            self._log_metrics(
                {
                    "episode/reward": payload.get("total_reward"),
                    "episode/action_count": payload.get("action_count"),
                    "episode/pool_changed": int(bool(payload.get("pool_changed"))),
                },
                step=step,
            )
        elif event == "mcts_candidates_scored":
            scores = [
                score
                for item in payload.get("candidates", [])
                if (score := _finite(item.get("score"))) is not None
            ]
            if scores:
                step = self._next("candidate_batch")
                self._log_metrics(
                    {
                        "candidate/max_score": max(scores),
                        "candidate/mean_score": sum(scores) / len(scores),
                        "candidate/min_score": min(scores),
                        "candidate/count": len(scores),
                    },
                    step=step,
                )
        elif event == "mcts_terminal_result":
            step = self._next("terminal")
            self._log_metrics(
                {
                    "pool/terminal_reward": payload.get("terminal_reward"),
                    "pool/resulting_score": payload.get("resulting_score"),
                    "pool/additive_delta": payload.get("additive_delta"),
                    "pool/size": payload.get("pool_size"),
                    "pool/committed": int(bool(payload.get("committed"))),
                },
                step=step,
            )
        elif event == "replay_quantile_update":
            step = self._next("quantile")
            self._log_metrics(
                {
                    "policy/trajectory_reward": payload.get("reward"),
                    "policy/quantile_before": payload.get("threshold_before"),
                    "policy/quantile_after": payload.get("threshold_after"),
                    "policy/selected_for_risk_update": int(
                        bool(payload.get("selected_for_risk_update"))
                    ),
                },
                step=step,
            )
        elif event == "policy_train_batch_done":
            step = self._next("policy_batch")
            self._log_metrics({"policy/loss": payload.get("loss")}, step=step)
        elif event == "mining_iteration_done":
            step = int(payload.get("iteration", self._next("iteration")))
            self._log_metrics(
                {
                    "iteration/pool_score": payload.get("pool_score"),
                    "iteration/pool_size": payload.get("pool_size"),
                    "iteration/pool_updates": payload.get("pool_updates"),
                    "iteration/trajectories": payload.get("trajectories"),
                    "iteration/quantile": payload.get("quantile_after"),
                    "iteration/mean_reward": payload.get("mean_reward"),
                    "iteration/max_reward": payload.get("max_reward"),
                },
                step=step,
            )
            checkpoint = payload.get("checkpoint")
            if checkpoint and Path(str(checkpoint)).is_file():
                self.mlflow.log_artifact(str(checkpoint), artifact_path="policy")
        elif event == "replay_snapshot":
            iteration = int(payload.get("iteration", self._next("replay")))
            rows = payload.get("trajectories", [])
            if rows:
                table = {
                    "index": [row.get("index") for row in rows],
                    "reward": [row.get("reward") for row in rows],
                    "terminal_rpn": [row.get("terminal_rpn") for row in rows],
                    "pool_changed": [row.get("pool_changed") for row in rows],
                    "actions": [json.dumps(row.get("actions", [])) for row in rows],
                    "step_rewards": [
                        json.dumps(row.get("step_rewards", [])) for row in rows
                    ],
                }
                self.mlflow.log_table(
                    data=table,
                    artifact_file=f"tables/replay_iteration_{iteration:04d}.json",
                )
        elif event in {
            "pool_compile_done", "pool_run_done",
            "orthogonal_compile_done", "orthogonal_run_done",
        }:
            step = self._next(event)
            timing = {
                f"timing/{event}/compile_seconds": payload.get("compile_seconds"),
                f"timing/{event}/run_seconds": payload.get("run_seconds"),
                f"timing/{event}/native_seconds": payload.get("native_seconds"),
            }
            self._log_metrics(timing, step=step)

    def finalize(self, report_path: str | Path | None = None) -> None:
        if self.trace_file is not None:
            self.trace_file.flush()
            self.trace_file.close()
            self.trace_file = None
        if self.mlflow is None or self.active_run is None:
            return
        if self.trace_path is not None and self.trace_path.is_file():
            self.mlflow.log_artifact(str(self.trace_path), artifact_path="trace")
        if report_path is not None and Path(report_path).is_file():
            self.mlflow.log_artifact(str(report_path), artifact_path="reports")
        self.mlflow.end_run(status="FINISHED")
        self.active_run = None


__all__ = ["MLflowTrackerConfig", "RiskMinerExperimentTracker"]
'''
(ROOT / 'src/flows/riskminer/tracking.py').write_text(tracking)

sweep = r'''from __future__ import annotations

"""Optuna + MLflow hyperparameter sweep for RiskMiner.

Everything is configured with environment variables rather than argparse. Each
real trial runs RiskMiner in a fresh subprocess so JAX/cpp_stream memory and
compiler state do not leak across trials.

Typical use:

    RISKMINER_SWEEP_TRIALS=30 PYTHONPATH=src python scripts/sweep_riskminer.py

Then visualize the Optuna journal with:

    optuna-dashboard /tmp/riskminer-sweeps/riskminer.journal

and detailed child runs with the MLflow UI/server documented in
``docs/riskminer_experiment_tracking.md``.
"""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import mlflow
import optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = Path(os.environ.get("RISKMINER_SWEEP_DIR", "/tmp/riskminer-sweeps"))
STUDY_NAME = os.environ.get("RISKMINER_SWEEP_STUDY", "riskminer")
TRIALS = int(os.environ.get("RISKMINER_SWEEP_TRIALS", "30"))
JOBS = int(os.environ.get("RISKMINER_SWEEP_JOBS", "1"))
PRESET = os.environ.get("RISKMINER_SWEEP_PRESET", "standard").strip().lower()
SEED = int(os.environ.get("RISKMINER_SWEEP_SEED", "42"))
PRUNING = os.environ.get("RISKMINER_SWEEP_PRUNING", "1").lower() in {
    "1", "true", "yes", "on"
}
DRY_RUN = os.environ.get("RISKMINER_SWEEP_DRY_RUN", "0").lower() in {
    "1", "true", "yes", "on"
}
STORAGE_URL = os.environ.get("RISKMINER_SWEEP_STORAGE_URL", "").strip()
JOURNAL_PATH = Path(
    os.environ.get(
        "RISKMINER_SWEEP_JOURNAL",
        str(SWEEP_DIR / f"{STUDY_NAME}.journal"),
    )
)
MLFLOW_TRACKING_URI = os.environ.get(
    "RISKMINER_MLFLOW_TRACKING_URI",
    f"sqlite:///{(SWEEP_DIR / 'mlflow.db').resolve()}",
)
MLFLOW_EXPERIMENT = os.environ.get(
    "RISKMINER_MLFLOW_EXPERIMENT", "riskminer-sweeps"
)

if TRIALS <= 0 or JOBS <= 0:
    raise ValueError("RISKMINER_SWEEP_TRIALS and RISKMINER_SWEEP_JOBS must be positive")
if PRESET not in {"quick", "standard", "wide"}:
    raise ValueError("RISKMINER_SWEEP_PRESET must be quick, standard, or wide")


def _storage():
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    if STORAGE_URL:
        return STORAGE_URL
    JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    return JournalStorage(JournalFileBackend(str(JOURNAL_PATH)))


def _suggest(trial: optuna.Trial) -> dict[str, str]:
    if PRESET == "quick":
        simulations = [8, 16, 32]
        rollouts = [1, 2]
        replay = [128, 256]
        batches = [16, 32]
        exploration = (0.75, 1.75)
        widening_k = (2.5, 6.0)
        widening_alpha = (0.35, 0.70)
        end_probability = (0.15, 0.40)
        policy_lr = (3.0e-4, 2.0e-3)
        quantile_cdf = (0.75, 0.90)
        quantile_lr = (5.0e-3, 2.0e-2)
    elif PRESET == "wide":
        simulations = [16, 32, 64, 128, 256]
        rollouts = [1, 2, 4, 8]
        replay = [128, 256, 512, 1024]
        batches = [16, 32, 64, 128]
        exploration = (0.25, 3.0)
        widening_k = (1.5, 10.0)
        widening_alpha = (0.20, 0.90)
        end_probability = (0.05, 0.60)
        policy_lr = (5.0e-5, 5.0e-3)
        quantile_cdf = (0.60, 0.97)
        quantile_lr = (1.0e-3, 7.5e-2)
    else:
        simulations = [16, 32, 64, 128]
        rollouts = [1, 2, 4]
        replay = [128, 256, 512]
        batches = [16, 32, 64]
        exploration = (0.50, 2.25)
        widening_k = (2.0, 8.0)
        widening_alpha = (0.30, 0.80)
        end_probability = (0.10, 0.50)
        policy_lr = (1.0e-4, 3.0e-3)
        quantile_cdf = (0.70, 0.95)
        quantile_lr = (2.0e-3, 4.0e-2)

    return {
        "RISKMINER_SIMULATIONS": str(
            trial.suggest_categorical("simulations", simulations)
        ),
        "RISKMINER_ROLLOUTS": str(trial.suggest_categorical("rollouts", rollouts)),
        "RISKMINER_EXPLORATION": str(
            trial.suggest_float("exploration", *exploration)
        ),
        "RISKMINER_PROGRESSIVE_WIDENING_K": str(
            trial.suggest_float("progressive_widening_k", *widening_k, log=True)
        ),
        "RISKMINER_PROGRESSIVE_WIDENING_ALPHA": str(
            trial.suggest_float("progressive_widening_alpha", *widening_alpha)
        ),
        "RISKMINER_ROLLOUT_END_PROBABILITY": str(
            trial.suggest_float("rollout_end_probability", *end_probability)
        ),
        "RISKMINER_POLICY_LEARNING_RATE": str(
            trial.suggest_float("policy_learning_rate", *policy_lr, log=True)
        ),
        "RISKMINER_QUANTILE_CDF": str(
            trial.suggest_float("quantile_cdf", *quantile_cdf)
        ),
        "RISKMINER_QUANTILE_LEARNING_RATE": str(
            trial.suggest_float("quantile_learning_rate", *quantile_lr, log=True)
        ),
        "RISKMINER_REPLAY_CAPACITY": str(
            trial.suggest_categorical("replay_capacity", replay)
        ),
        "RISKMINER_POLICY_BATCH_SIZE": str(
            trial.suggest_categorical("policy_batch_size", batches)
        ),
    }


def _parse_record(line: str) -> dict | None:
    marker = "] {"
    if marker not in line:
        return None
    start = line.find("{")
    if start < 0:
        return None
    try:
        return json.loads(line[start:])
    except json.JSONDecodeError:
        return None


def _dry_objective(trial: optuna.Trial, parent_run_id: str) -> float:
    env_params = _suggest(trial)
    numeric = {key: float(value) for key, value in env_params.items()}
    score = -(
        (numeric["RISKMINER_EXPLORATION"] - 1.25) ** 2
        + (numeric["RISKMINER_PROGRESSIVE_WIDENING_ALPHA"] - 0.5) ** 2
        + (math.log10(numeric["RISKMINER_POLICY_LEARNING_RATE"]) + 3.0) ** 2
    )
    with mlflow.start_run(run_name=f"trial-{trial.number:04d}", nested=True):
        mlflow.log_params(trial.params)
        mlflow.log_metric("validation/pool_score", score)
        mlflow.set_tag("optuna.trial_number", trial.number)
        trial.set_user_attr("mlflow_run_id", mlflow.active_run().info.run_id)
    return float(score)


def _objective(parent_run_id: str):
    def objective(trial: optuna.Trial) -> float:
        if DRY_RUN:
            return _dry_objective(trial, parent_run_id)

        trial_env = _suggest(trial)
        output_dir = SWEEP_DIR / "trials" / f"trial_{trial.number:04d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env.update(trial_env)
        env["RISKMINER_OUTPUT_DIR"] = str(output_dir)
        env["RISKMINER_MLFLOW_ENABLED"] = "1"
        env["RISKMINER_MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        env["RISKMINER_MLFLOW_EXPERIMENT"] = MLFLOW_EXPERIMENT
        env["RISKMINER_MLFLOW_PARENT_RUN_ID"] = parent_run_id
        env["RISKMINER_MLFLOW_RUN_NAME"] = f"{STUDY_NAME}-trial-{trial.number:04d}"
        env["RISKMINER_MLFLOW_TAGS_JSON"] = json.dumps(
            {
                "optuna.study": STUDY_NAME,
                "optuna.trial_number": trial.number,
                "sweep.preset": PRESET,
            }
        )
        env.setdefault("RISKMINER_LOG_LEVEL", "summary")
        if JOBS == 1:
            # roll_rets/vol do not depend on MCTS hyperparameters, so sequential
            # trials share them. Parallel trials deliberately use per-trial files
            # to avoid races during first materialization.
            env["RISKMINER_DERIVED_DIR"] = str(SWEEP_DIR / "shared_derived")
            env["RISKMINER_REUSE_DERIVED"] = "1"
        else:
            env["RISKMINER_REUSE_DERIVED"] = "0"
            if "RISKMINER_THREADS" not in env:
                cpus = os.cpu_count() or 1
                env["RISKMINER_THREADS"] = str(max(1, cpus // JOBS))

        command = [sys.executable, str(ROOT / "scripts" / "run_riskminer_inputdata.py")]
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        child_run_id: str | None = None
        last_score: float | None = None
        assert proc.stdout is not None
        for line in proc.stdout:
            print(f"[optuna trial={trial.number}] {line}", end="", flush=True)
            record = _parse_record(line)
            if not record:
                continue
            if record.get("event") == "mlflow_run_started":
                child_run_id = record.get("run_id")
                if child_run_id:
                    trial.set_user_attr("mlflow_run_id", child_run_id)
            if record.get("event") == "depth_iteration_done":
                score = record.get("pool_score")
                if score is not None and math.isfinite(float(score)):
                    last_score = float(score)
                    step = int(record.get("global_iteration", record.get("depth", 0)))
                    trial.report(last_score, step=step)
                    if PRUNING and trial.should_prune():
                        proc.terminate()
                        try:
                            proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
                        if child_run_id:
                            mlflow.MlflowClient().set_terminated(
                                child_run_id, status="KILLED"
                            )
                        raise optuna.TrialPruned(
                            f"pruned after step={step}, score={last_score}"
                        )
        return_code = proc.wait()
        if return_code != 0:
            raise RuntimeError(
                f"RiskMiner trial {trial.number} exited with code {return_code}"
            )

        report_path = output_dir / "riskminer_inputdata_report.json"
        if not report_path.is_file():
            raise RuntimeError(f"missing RiskMiner report: {report_path}")
        report = json.loads(report_path.read_text())
        score = report.get("pool_score_validation")
        if score is None or not math.isfinite(float(score)):
            score = last_score if last_score is not None else -1.0e12
        score = float(score)
        trial.set_user_attr("report", str(report_path))
        trial.set_user_attr("pool_size", len(report.get("pool", [])))
        trial.set_user_attr("test_score", report.get("pool_score_test"))
        if report.get("mlflow_run_id"):
            trial.set_user_attr("mlflow_run_id", report["mlflow_run_id"])
        return score

    return objective


def main() -> None:
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    storage = _storage()
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"sweep-{STUDY_NAME}", log_system_metrics=True) as parent:
        mlflow.log_params(
            {
                "optuna.study": STUDY_NAME,
                "optuna.trials_requested": TRIALS,
                "optuna.jobs": JOBS,
                "optuna.preset": PRESET,
                "optuna.pruning": PRUNING,
                "optuna.storage": STORAGE_URL or str(JOURNAL_PATH),
                "objective": "validation_pool_sharpe",
            }
        )
        study.optimize(
            _objective(parent.info.run_id),
            n_trials=TRIALS,
            n_jobs=JOBS,
            gc_after_trial=True,
        )
        if study.best_trial is not None:
            mlflow.log_metric("best/validation_pool_score", study.best_value)
            mlflow.log_params(
                {f"best.{name}": value for name, value in study.best_trial.params.items()}
            )
            mlflow.set_tag("best.optuna_trial", study.best_trial.number)
            if run_id := study.best_trial.user_attrs.get("mlflow_run_id"):
                mlflow.set_tag("best.mlflow_child_run_id", run_id)
        mlflow.log_dict(
            {
                "study": STUDY_NAME,
                "best_value": study.best_value if len(study.trials) else None,
                "best_params": study.best_params if len(study.trials) else {},
                "trial_count": len(study.trials),
            },
            "optuna/study_summary.json",
        )

    print("\n=== OPTUNA SWEEP COMPLETE ===")
    print(f"study={STUDY_NAME}")
    print(f"trials={len(study.trials)}")
    print(f"best_value={study.best_value}")
    print(f"best_params={study.best_params}")
    if not STORAGE_URL:
        print(f"dashboard: optuna-dashboard {JOURNAL_PATH}")
    print(
        "mlflow: mlflow server --backend-store-uri "
        f"{MLFLOW_TRACKING_URI} --host 127.0.0.1 --port 5000"
    )


if __name__ == "__main__":
    main()
'''
(ROOT / 'scripts/sweep_riskminer.py').write_text(sweep)

# Add dependencies.
p = ROOT / 'pyproject.toml'
s = p.read_text()
needle = '    "deap",\n]'
replacement = '    "deap",\n    "mlflow>=3.0",\n    "optuna>=4.0",\n    "optuna-dashboard>=0.17",\n    "psutil>=5.9",\n]'
if needle not in s:
    raise SystemExit('pyproject dependency insertion point not found')
s = s.replace(needle, replacement, 1)
p.write_text(s)

# Patch the InputData runner.
p = ROOT / 'scripts/run_riskminer_inputdata.py'
s = p.read_text()
s = s.replace(
    'from flows.riskminer.semantics import DEFAULT_TYPE_GRAPH, NON_VALUE_TYPES\n',
    'from flows.riskminer.semantics import DEFAULT_TYPE_GRAPH, NON_VALUE_TYPES\n'
    'from flows.riskminer.tracking import MLflowTrackerConfig, RiskMinerExperimentTracker\n',
    1,
)

# Expose MCTS knobs needed for real sweeps.
marker = '''MAX_TOKENS = int(os.environ.get("RISKMINER_MAX_TOKENS", "30"))\n\n'''
insert = '''MAX_TOKENS = int(os.environ.get("RISKMINER_MAX_TOKENS", "30"))\n\n# Maximum number of unresolved RPN stack values. Higher values allow broader\n# expressions to be assembled before reducing them with operators, but also make\n# dead-end rollouts easier. This was previously fixed at 8; it is exposed so\n# Optuna can sweep it if you choose to extend the default sweep space.\nMAX_STACK = int(os.environ.get("RISKMINER_MAX_STACK", "8"))\n\n'''
if marker not in s:
    raise SystemExit('MAX_TOKENS marker missing')
s = s.replace(marker, insert, 1)

marker = '''EXPLORATION = float(os.environ.get("RISKMINER_EXPLORATION", "1.25"))\n\n'''
insert = '''EXPLORATION = float(os.environ.get("RISKMINER_EXPLORATION", "1.25"))\n\n# Progressive widening controls how many legal children a tree node exposes as\n# the node receives visits: roughly K * visits ** ALPHA. Larger K/ALPHA open the\n# very large 132-token action space faster; smaller values force MCTS to spend\n# more evidence on the currently exposed high-prior actions before widening.\nPROGRESSIVE_WIDENING_K = float(\n    os.environ.get("RISKMINER_PROGRESSIVE_WIDENING_K", "4.0")\n)\nPROGRESSIVE_WIDENING_ALPHA = float(\n    os.environ.get("RISKMINER_PROGRESSIVE_WIDENING_ALPHA", "0.5")\n)\n\n'''
if marker not in s:
    raise SystemExit('EXPLORATION marker missing')
s = s.replace(marker, insert, 1)

marker = '''REPLAY_CAPACITY = int(os.environ.get("RISKMINER_REPLAY_CAPACITY", "256"))\n\n'''
insert = '''REPLAY_CAPACITY = int(os.environ.get("RISKMINER_REPLAY_CAPACITY", "256"))\n\n# Reward assigned to a rollout that cannot produce a valid END-terminated formula.\n# More negative values teach the policy/tree to avoid dead ends more aggressively.\nINVALID_REWARD = float(os.environ.get("RISKMINER_INVALID_REWARD", "-5.0"))\n\n# Discount applied while backing immediate rewards up a trajectory. 1.0 gives all\n# later rewards their full weight (the current/paper-style default); values below\n# one increasingly emphasize rewards closer to the selected edge.\nDISCOUNT = float(os.environ.get("RISKMINER_DISCOUNT", "1.0"))\n\n'''
if marker not in s:
    raise SystemExit('REPLAY marker missing')
s = s.replace(marker, insert, 1)

# Derived directory can be shared across sequential Optuna trials.
marker = '''OUTPUT_DIR = Path(\n    os.environ.get("RISKMINER_OUTPUT_DIR", "/tmp/riskminer-inputdata")\n)\n\n'''
insert = '''OUTPUT_DIR = Path(\n    os.environ.get("RISKMINER_OUTPUT_DIR", "/tmp/riskminer-inputdata")\n)\n\n# Optional separate directory for roll_rets/vol materialization. Hyperparameter\n# sweeps can point every sequential trial at one shared directory because these\n# arrays depend on the input data, not on MCTS hyperparameters. Do not share it\n# across concurrently starting trials unless the files have already been built.\nDERIVED_DIR = Path(\n    os.environ.get("RISKMINER_DERIVED_DIR", str(OUTPUT_DIR / "derived"))\n)\n\n'''
if marker not in s:
    raise SystemExit('OUTPUT_DIR marker missing')
s = s.replace(marker, insert, 1)

# Replace old fixed-knob note with MLflow documentation.
old = '''# Important RiskMinerConfig knobs that are intentionally fixed in this script\n# rather than exposed as environment variables: max_stack=8, invalid_reward=-5,\n# discount=1.0. Progressive widening uses RiskMinerConfig defaults k=4.0 and\n# alpha=0.5. Change those in base_config/config.py if experimenting with them.\n'''
new = '''# MLflow experiment tracking is ON by default. It logs low-frequency scalar\n# progress (pool score, rewards, candidate score summaries, policy loss), replay\n# tables, policy checkpoints, final reports, and optional CPU/memory metrics. The\n# complete high-cardinality MCTS event stream is always written to JSONL under\n# OUTPUT_DIR/trace and uploaded as an MLflow artifact at the end.\n#\n# RISKMINER_MLFLOW_ENABLED=1|0\n#   Master switch. Disable for minimum tracking overhead. JSONL trace is still kept.\n# RISKMINER_MLFLOW_TRACKING_URI\n#   Default: sqlite:///<OUTPUT_DIR>/mlflow.db. For truly live browser monitoring,\n#   start `mlflow server` and set this to http://127.0.0.1:5000.\n# RISKMINER_MLFLOW_EXPERIMENT\n#   MLflow experiment/group name. Default: riskminer.\n# RISKMINER_MLFLOW_RUN_NAME\n#   Optional human-readable run name. MLflow generates one when omitted.\n# RISKMINER_MLFLOW_PARENT_RUN_ID\n#   Optional parent run. sweep_riskminer.py sets this so every Optuna trial appears\n#   as a child beneath one sweep run in MLflow.\n# RISKMINER_MLFLOW_SYSTEM_METRICS=1|0\n#   Log CPU, process/system memory, disk and network metrics. Recommended while\n#   diagnosing compilation/JAX memory behavior.\n# RISKMINER_MLFLOW_SYSTEM_METRICS_INTERVAL\n#   Sampling interval in seconds for MLflow system metrics. Default: 5.\n# RISKMINER_MLFLOW_TAGS_JSON\n#   Optional JSON object of searchable tags, e.g. '{"dataset":"aks_out3"}'.\nMLFLOW_CONFIG = MLflowTrackerConfig.from_env(OUTPUT_DIR)\n'''
if old not in s:
    raise SystemExit('fixed knob note missing')
s = s.replace(old, new, 1)

# Use shared derived directory.
s = s.replace('    derived_dir = OUTPUT_DIR / "derived"\n', '    derived_dir = DERIVED_DIR\n', 1)

# Wire tracker into main and shared event sink.
old = '''def main() -> None:\n    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n    progress = ConsoleProgress(prefix="riskminer-inputdata")\n\n    def event(event_name: str, payload) -> None:\n        required = (\n            2 if event_name in TRACE_EVENTS\n            else 1 if event_name in DETAIL_EVENTS\n            else 0\n        )\n        if LOG_LEVEL_RANK[LOG_LEVEL] >= required:\n            progress.emit(event_name, **dict(payload))\n\n    progress.emit(\n'''
new = '''def main() -> None:\n    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)\n    progress = ConsoleProgress(prefix="riskminer-inputdata")\n    tracker = RiskMinerExperimentTracker(MLFLOW_CONFIG, OUTPUT_DIR)\n    tracker.start(\n        params={\n            "rows_requested": ROWS,\n            "max_depth": MAX_DEPTH,\n            "max_tokens": MAX_TOKENS,\n            "max_stack": MAX_STACK,\n            "iterations_per_depth": ITERATIONS_PER_DEPTH,\n            "simulations": SIMULATIONS,\n            "rollouts": ROLLOUTS,\n            "evaluation_batch": EVALUATION_BATCH,\n            "archive_size": ARCHIVE_SIZE,\n            "pool_capacity": POOL_CAPACITY,\n            "pool_min_improvement": POOL_MIN_IMPROVEMENT,\n            "pool_importance": POOL_IMPORTANCE,\n            "ridge_recompute_every": RIDGE_RECOMPUTE_EVERY,\n            "train_fraction": TRAIN_FRACTION,\n            "validation_fraction": VALIDATION_FRACTION,\n            "policy_epochs": POLICY_EPOCHS,\n            "policy_batch_size": POLICY_BATCH_SIZE,\n            "policy_learning_rate": POLICY_LEARNING_RATE,\n            "quantile_cdf": QUANTILE_CDF,\n            "quantile_learning_rate": QUANTILE_LEARNING_RATE,\n            "exploration": EXPLORATION,\n            "progressive_widening_k": PROGRESSIVE_WIDENING_K,\n            "progressive_widening_alpha": PROGRESSIVE_WIDENING_ALPHA,\n            "rollout_end_probability": ROLLOUT_END_PROBABILITY,\n            "replay_capacity": REPLAY_CAPACITY,\n            "invalid_reward": INVALID_REWARD,\n            "discount": DISCOUNT,\n            "threads": THREADS,\n            "seed": SEED,\n            "input_glob": INPUT_GLOB,\n        },\n        tags={"riskminer.log_level": LOG_LEVEL},\n    )\n    progress.emit(\n        "mlflow_run_started",\n        enabled=MLFLOW_CONFIG.enabled,\n        run_id=tracker.run_id,\n        tracking_uri=MLFLOW_CONFIG.tracking_uri,\n        experiment=MLFLOW_CONFIG.experiment_name,\n        trace_path=str(tracker.trace_path) if tracker.trace_path else None,\n    )\n\n    def event(event_name: str, payload) -> None:\n        payload = dict(payload)\n        tracker.emit(event_name, payload)\n        required = (\n            2 if event_name in TRACE_EVENTS\n            else 1 if event_name in DETAIL_EVENTS\n            else 0\n        )\n        if LOG_LEVEL_RANK[LOG_LEVEL] >= required:\n            progress.emit(event_name, **payload)\n\n    progress.emit(\n'''
if old not in s:
    raise SystemExit('main tracker insertion point missing')
s = s.replace(old, new, 1)

# Config knobs.
s = s.replace('        max_stack=8,\n', '        max_stack=MAX_STACK,\n', 1)
s = s.replace(
    '        exploration=EXPLORATION,\n        rollout_end_probability=ROLLOUT_END_PROBABILITY,\n        invalid_reward=-5.0,\n',
    '        exploration=EXPLORATION,\n        progressive_widening_k=PROGRESSIVE_WIDENING_K,\n        progressive_widening_alpha=PROGRESSIVE_WIDENING_ALPHA,\n        rollout_end_probability=ROLLOUT_END_PROBABILITY,\n        invalid_reward=INVALID_REWARD,\n        discount=DISCOUNT,\n',
    1,
)

# Trainer events must flow through both console filtering and MLflow.
s = s.replace(
    '        on_event=lambda name, payload: progress.emit(name, **payload),\n',
    '        on_event=event,\n',
    1,
)

# Make depth completion visible to the sweep parser AND tracker.
old = '''            progress.emit(\n                "depth_iteration_done",\n                depth=depth,\n                depth_iteration=depth_iteration,\n                trajectories=report.search.metrics.trajectories,\n                pool_updates=report.search.metrics.pool_updates,\n                pool_size=len(pool.entries),\n                pool_score=(pool.score if math.isfinite(pool.score) else None),\n                quantile=trainer.quantile.value,\n                best_archive_score=(\n                    report.search.archive[0].score if report.search.archive else None\n                ),\n            )\n'''
new = '''            event(\n                "depth_iteration_done",\n                {\n                    "depth": depth,\n                    "depth_iteration": depth_iteration,\n                    "global_iteration": iteration,\n                    "trajectories": report.search.metrics.trajectories,\n                    "pool_updates": report.search.metrics.pool_updates,\n                    "pool_size": len(pool.entries),\n                    "pool_score": (pool.score if math.isfinite(pool.score) else None),\n                    "quantile": trainer.quantile.value,\n                    "best_archive_score": (\n                        report.search.archive[0].score if report.search.archive else None\n                    ),\n                },\n            )\n'''
if old not in s:
    raise SystemExit('depth done block missing')
s = s.replace(old, new, 1)

# Add tracking metadata to final report and close MLflow after logging it.
s = s.replace(
    '        "backend": "trading_dsl_engine.cpp_stream",\n',
    '        "backend": "trading_dsl_engine.cpp_stream",\n'
    '        "mlflow_run_id": tracker.run_id,\n'
    '        "mlflow_tracking_uri": MLFLOW_CONFIG.tracking_uri if MLFLOW_CONFIG.enabled else None,\n'
    '        "trace_path": str(tracker.trace_path) if tracker.trace_path else None,\n',
    1,
)
s = s.replace(
    '            "max_tokens": MAX_TOKENS,\n',
    '            "max_tokens": MAX_TOKENS,\n'
    '            "max_stack": MAX_STACK,\n'
    '            "progressive_widening_k": PROGRESSIVE_WIDENING_K,\n'
    '            "progressive_widening_alpha": PROGRESSIVE_WIDENING_ALPHA,\n'
    '            "invalid_reward": INVALID_REWARD,\n'
    '            "discount": DISCOUNT,\n',
    1,
)
old = '''    progress.emit(\n        "done",\n        report=str(report_path),\n        pool_size=len(pool.entries),\n        validation_score=result["pool_score_validation"],\n        test_score=result["pool_score_test"],\n        search_seconds=result["search_seconds"],\n    )\n    print("=== FINAL ROOT-LEVEL RIDGE POOL ===", flush=True)\n'''
new = '''    event(\n        "done",\n        {\n            "report": str(report_path),\n            "pool_size": len(pool.entries),\n            "validation_score": result["pool_score_validation"],\n            "test_score": result["pool_score_test"],\n            "search_seconds": result["search_seconds"],\n        },\n    )\n    tracker.finalize(report_path)\n    print("=== FINAL ROOT-LEVEL RIDGE POOL ===", flush=True)\n'''
if old not in s:
    raise SystemExit('done block missing')
s = s.replace(old, new, 1)
p.write_text(s)

# Documentation.
doc = r'''# RiskMiner experiment visualization and hyperparameter sweeps

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
'''
(ROOT / 'docs/riskminer_experiment_tracking.md').write_text(doc)

print('RiskMiner experiment tracking patch applied')
