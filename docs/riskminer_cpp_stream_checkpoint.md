# RiskMiner on `cpp_stream`

The feature branch now contains two search interfaces:

1. `RiskMCTS` and `CppStreamCandidateEvaluator` preserve the earlier standalone
   Sharpe search for focused experiments.
2. `RewardDenseRiskMCTS`, `RiskSeekingTrainer`, and
   `train_cpp_stream_riskminer` implement the complete pool-aware neural search
   loop.

The complete loop includes typed RPN legality, neural PUCT/rollout priors,
intermediate rewards, exact terminal Ridge-pool rewards, edge-specific reward
backup, replay, quantile tracking, policy training, policy checkpoints, pool
capacity eviction, and chronological train/validation/test evaluation.

See:

- `docs/riskminer_paper_pipeline.md`
- `docs/riskminer_inputdata.md`
- `scripts/run_riskminer_inputdata.py`
