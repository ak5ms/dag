from __future__ import annotations

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
