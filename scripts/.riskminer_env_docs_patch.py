from pathlib import Path

p = Path('scripts/run_riskminer_inputdata.py')
s = p.read_text()
start = s.index('# Moderate workstation defaults.')
end = s.index('LOG_LEVEL_RANK =', start)
new = r'''# -----------------------------------------------------------------------------
# Runtime/search configuration
# -----------------------------------------------------------------------------
# Every setting below can be overridden with an environment variable, e.g.:
#
#   RISKMINER_ROWS=500000 RISKMINER_SIMULATIONS=128 \
#   PYTHONPATH=src python scripts/run_riskminer_inputdata.py
#
# The defaults are deliberately small enough to inspect the search interactively.
# They are not intended to be a full production mining budget.

# InputData glob. InputData discovers all source arrays from this pattern and the
# MCTS vocabulary is then restricted to INPUTDATA_ALPHA_KEYS. Evaluation-only
# fields such as roll_rets/vol/hs are constructed separately and cannot leak into
# generated alpha formulas.
INPUT_GLOB = os.environ.get(
    "RISKMINER_INPUT_GLOB", "/mnt/extra/qrt/data/aks_out3/*.npy"
)

# Number of contiguous rows loaded from the start of every source before the
# train/validation/test split. 0 means use every available row. This is the most
# useful knob for controlling search cost while debugging: ~100k is a smoke test;
# hundreds of thousands are more useful for mining; use all rows for a final
# validation only when the search budget/runtime is acceptable.
ROWS = int(os.environ.get("RISKMINER_ROWS", "100000"))

# Largest expression-tree depth searched. The runner performs exact-depth stages
# 1, 2, ..., MAX_DEPTH by setting min_formula_depth == max_depth for each stage.
# This is NOT the number of RPN tokens. For example, `x` has depth 1 and
# `x xs_rank` has depth 2. Larger values unlock more nested formulas but expand
# the search space very rapidly.
MAX_DEPTH = int(os.environ.get("RISKMINER_MAX_DEPTH", "8"))

# Hard cap on the full RPN episode length, including operands/operators and END.
# This controls total expression size/breadth independently of tree depth. The
# RiskMiner paper also caps episodes at 30. Increase only if valid formulas are
# frequently hitting the token ceiling; larger values make rollouts much harder.
MAX_TOKENS = int(os.environ.get("RISKMINER_MAX_TOKENS", "30"))

# Number of complete MCTS -> replay -> neural-policy-update cycles run at EACH
# exact depth. A new MCTS tree and replay buffer are created every iteration, as
# in the paper, while the learned neural policy and accepted alpha pool persist.
# Increasing this gives the policy more chances to learn/search a given depth.
ITERATIONS_PER_DEPTH = int(
    os.environ.get("RISKMINER_ITERATIONS_PER_DEPTH", "1")
)

# Number of MCTS simulations per mining iteration. One simulation starts at the
# root, follows the tree policy to a leaf, expands the tree, performs ROLLOUTS
# completions, and backpropagates their rewards. This is the primary search-budget
# knob and runtime is roughly linear in it. The paper uses 200 search cycles per
# mining iteration; the default 8 here is intentionally a diagnostic budget.
SIMULATIONS = int(os.environ.get("RISKMINER_SIMULATIONS", "8"))

# Number of stochastic BEG->END completions launched from each selected/expanded
# leaf. More rollouts give a less noisy estimate of whether a partial expression
# is promising, but every valid completion can trigger intermediate native alpha
# evaluation and an exact validation Ridge-pool trial, so this can be expensive.
ROLLOUTS = int(os.environ.get("RISKMINER_ROLLOUTS", "1"))

# Native intermediate-evaluation batch size. The orthogonal-alpha scorer batches
# up to this many formulas into cpp_stream work to amortize compile/data-scan cost.
# In the current reward-dense MCTS this is NOT a count of simulations processed
# together; tree simulations themselves remain sequential because each one can
# change tree statistics and the alpha pool.
EVALUATION_BATCH = int(os.environ.get("RISKMINER_EVALUATION_BATCH", "8"))

# Maximum number of unique scored formulas retained in the per-search FormulaArchive.
# The archive is for diagnostics/ranking only: it is NOT the alpha pool and it is
# NOT the replay buffer. Increasing it mainly costs Python memory, not Ridge work.
ARCHIVE_SIZE = int(os.environ.get("RISKMINER_ARCHIVE_SIZE", "500"))

# Maximum number K of accepted alphas in the root Ridge pool. Once a candidate
# would make K+1 alphas, coefficient-based eviction is run before admission.
# The paper uses K=100, which is therefore the default here as well.
POOL_CAPACITY = int(os.environ.get("RISKMINER_POOL_CAPACITY", "100"))

# Minimum required increase in VALIDATION pool Sharpe for a candidate/replacement
# to be committed. Admission is strict: delta must be > this value. The empty-pool
# baseline is zero, so the first alpha must have positive validation pool Sharpe
# when this is nonnegative. 1e-8 acts like "strictly positive" with a tiny numeric
# tolerance; raise it to demand economically meaningful rather than tiny gains.
POOL_MIN_IMPROVEMENT = float(
    os.environ.get("RISKMINER_POOL_MIN_IMPROVEMENT", "1e-8")
)

# How often the top-level temporal Ridge recomputes beta. k=1 solves beta every
# bar and is the reference/exact runner behavior. k>1 keeps updating Ridge state
# but reuses beta between solve bars, which can materially reduce solve cost at
# the price of changing the resulting yhat/pool Sharpe. Treat k>1 as a deliberate
# performance/accuracy tradeoff and revalidate final results with k=1.
RIDGE_RECOMPUTE_EVERY = int(
    os.environ.get("RISKMINER_RIDGE_RECOMPUTE_EVERY", "1")
)

# Contiguous fraction of rows used for TRAIN rewards. Intermediate candidate
# rewards (including cross-sectional orthogonalization against the current pool)
# are measured here. Data are not randomly shuffled across time.
TRAIN_FRACTION = float(os.environ.get("RISKMINER_TRAIN_FRACTION", "0.70"))

# Contiguous fraction immediately after TRAIN used to decide exact Ridge-pool
# admission/eviction. The remaining fraction, 1 - TRAIN - VALIDATION, is the
# untouched final TEST segment. TRAIN + VALIDATION must therefore be < 1.
VALIDATION_FRACTION = float(
    os.environ.get("RISKMINER_VALIDATION_FRACTION", "0.15")
)

# Number of passes over the current iteration's replay buffer when updating the
# neural policy. The replay buffer is intentionally RESET every mining iteration;
# only the neural-network parameters (and alpha pool) carry learned information
# forward. More epochs fit the current trajectories harder and can overfit a tiny
# replay sample.
POLICY_EPOCHS = int(os.environ.get("RISKMINER_POLICY_EPOCHS", "1"))

# Number of replay trajectories per neural optimizer step. This is also used as
# the fixed JAX batch dimension so repeated updates reuse one compiled executable.
# Larger batches use more memory and produce fewer updates per replay epoch.
POLICY_BATCH_SIZE = int(os.environ.get("RISKMINER_POLICY_BATCH_SIZE", "32"))

# Gradient-descent step size for the GRU/MLP risk policy. This corresponds to the
# paper's policy-network update learning rate (reported as 0.001). If training is
# unstable/noisy, lower it; increasing it makes the learned priors change faster.
POLICY_LEARNING_RATE = float(
    os.environ.get("RISKMINER_POLICY_LEARNING_RATE", "0.001")
)

# CDF quantile tracked by the risk-seeking optimizer. 0.80 means the running
# threshold estimates the 80th percentile of trajectory reward. The paper's
# gradient suppresses probability of trajectories at/below that threshold,
# thereby shifting probability mass toward the upper tail rather than optimizing
# only average reward. Higher values make the cutoff more demanding.
QUANTILE_CDF = float(os.environ.get("RISKMINER_QUANTILE_CDF", "0.80"))

# Step size beta for the stochastic quantile recursion (paper Eq. 11). Larger
# values let the reward threshold react faster to a changing policy/search, but
# also make the threshold noisier. The paper uses 0.01.
QUANTILE_LEARNING_RATE = float(
    os.environ.get("RISKMINER_QUANTILE_LEARNING_RATE", "0.01")
)

# PUCT exploration multiplier c. Tree selection is approximately
#   Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a)).
# Larger values give the neural prior P more influence and revisit under-sampled
# actions more aggressively; smaller values exploit actions with high observed Q.
EXPLORATION = float(os.environ.get("RISKMINER_EXPLORATION", "1.25"))

# Probability that a stochastic rollout chooses END when END is legal. Higher
# values prefer shorter formulas; lower values keep extending them. In THIS
# exact-depth staged runner its effect is limited because END is illegal below
# the requested depth and is forced once a valid expression reaches max_depth.
ROLLOUT_END_PROBABILITY = float(
    os.environ.get("RISKMINER_ROLLOUT_END_PROBABILITY", "0.25")
)

# Maximum number of trajectories retained for neural-policy training inside ONE
# mining iteration. Because the buffer is reset each iteration, this is not a
# long-term experience store. If simulations*rollouts exceed the capacity, the
# oldest trajectories from that iteration are dropped and the newest are kept.
REPLAY_CAPACITY = int(os.environ.get("RISKMINER_REPLAY_CAPACITY", "256"))

# How a time-varying Ridge beta series is reduced to one importance number when
# the pool is full and one alpha must be evicted:
#   mean_abs  -> mean_t |beta_t,j| over the validation segment (default; smoother)
#   final_abs -> |beta_T,j| at the final validation row (closer to a single fitted
#                paper-style weight, but can be much noisier for online Ridge).
# The PAPER's rule is simply "remove the alpha with the smallest absolute fitted
# linear-model weight". This mean/final choice is our adaptation because the root
# synthesis model here is an online/temporal Ridge whose coefficients vary by bar.
POOL_IMPORTANCE = os.environ.get("RISKMINER_POOL_IMPORTANCE", "mean_abs")

# Requested cpp_stream thread count. 0 means do not pass an explicit thread count
# and let the backend/default planner decide. A positive value is a request, not a
# guarantee: dependency-heavy or currently-serial graphs may still use fewer cores.
THREADS = int(os.environ.get("RISKMINER_THREADS", "0"))

# Master random seed used to derive stage seeds for MCTS rollout sampling, replay
# shuffling and policy initialization. Keep fixed for reproducibility; change it
# when measuring robustness across independent mining runs.
SEED = int(os.environ.get("RISKMINER_SEED", "42"))

# Seconds between "still running" heartbeat messages around long native stages.
# This affects logging only, not search/evaluation semantics.
HEARTBEAT_SECONDS = float(
    os.environ.get("RISKMINER_HEARTBEAT_SECONDS", "5")
)

# Working/output directory. Contains derived roll_rets/vol arrays, cpp_stream
# intermediate and validation outputs, policy checkpoints and the final JSON report.
OUTPUT_DIR = Path(
    os.environ.get("RISKMINER_OUTPUT_DIR", "/tmp/riskminer-inputdata")
)

# Reuse already-materialized roll_rets.npy and vol.npy when their shape/dtype
# match the requested run. WARNING: the current reuse check does NOT fingerprint
# the underlying InputData contents. Leave this off after changing input files if
# the replacement data could have the same shape, or stale derived arrays may be used.
REUSE_DERIVED = os.environ.get("RISKMINER_REUSE_DERIVED", "0").lower() in {
    "1", "true", "yes", "on",
}

# Optional path to a previously saved JaxGRUPolicy checkpoint. This resumes the
# learned neural prior only. The current script still starts a NEW alpha pool and
# a NEW quantile tracker; it is not a full mining-session resume. The checkpoint
# vocabulary size must match the vocabulary constructed for this run.
RESUME_POLICY = os.environ.get("RISKMINER_RESUME_POLICY", "").strip()

# Console verbosity:
#   summary -> stage/compile/pool/final progress only
#   detail  -> plus MCTS episodes, RPN candidates/scores, replay and training
#   trace   -> plus every node/PUCT choice, rollout token and backpropagated edge
# Formula logging is RPN-only; internal Expr AST reprs are intentionally omitted.
LOG_LEVEL = os.environ.get("RISKMINER_LOG_LEVEL", "trace").strip().lower()
if LOG_LEVEL not in {"summary", "detail", "trace"}:
    raise ValueError("RISKMINER_LOG_LEVEL must be summary, detail, or trace")

# Important RiskMinerConfig knobs that are intentionally fixed in this script
# rather than exposed as environment variables: max_stack=8, invalid_reward=-5,
# discount=1.0. Progressive widening uses RiskMinerConfig defaults k=4.0 and
# alpha=0.5. Change those in base_config/config.py if experimenting with them.
'''
s = s[:start] + new + s[end:]
p.write_text(s)
print('documented RiskMiner environment variables inline')
