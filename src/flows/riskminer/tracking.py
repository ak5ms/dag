from __future__ import annotations

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
