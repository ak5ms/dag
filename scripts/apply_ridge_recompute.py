from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def replace_once(relative: str, old: str, new: str) -> None:
    path = ROOT / relative
    text = path.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{relative}: expected one exact replacement, found {count}"
        )
    path.write_text(text.replace(old, new))


def regex_once(relative: str, pattern: str, replacement: str) -> None:
    path = ROOT / relative
    text = path.read_text()
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(
            f"{relative}: expected one regex replacement for {pattern!r}, found {count}"
        )
    path.write_text(updated)


def insert_before_last(relative: str, marker: str, insertion: str) -> None:
    path = ROOT / relative
    text = path.read_text()
    index = text.rfind(marker)
    if index < 0:
        raise RuntimeError(f"{relative}: marker not found: {marker!r}")
    path.write_text(text[:index] + insertion + text[index:])


def patch_utils() -> None:
    replace_once(
        "src/trading_dsl_engine/cpp_stream/cpp/stackdsl/utils.hpp",
        """template <class T>
inline constexpr bool always_false_v = false;

template <class T>
STACKDSL_HOT bool finite(T value) noexcept {
""",
        """template <class T>
inline constexpr bool always_false_v = false;

// Reusable compile-time scheduler for expensive state refreshes.  The first
// observation of each slot refreshes immediately; subsequent refreshes occur
// after Every global rows.  Every=1 specializes to a zero-state, always-due
// path so existing operators pay no counter/modulo cost.
template <std::size_t Every, std::size_t Slots = 1>
struct PeriodicRecompute {
    static_assert(Every > 0, "PeriodicRecompute Every must be > 0");
    static_assert(Slots > 0, "PeriodicRecompute Slots must be > 0");

    std::array<std::uint64_t, Slots> last{};
    std::array<std::uint8_t, Slots> initialized{};
    std::uint64_t row = 0;

    STACKDSL_HOT bool due(std::size_t slot = 0) noexcept {
        if (!initialized[slot] || row - last[slot] >= Every) {
            initialized[slot] = 1;
            last[slot] = row;
            return true;
        }
        return false;
    }

    STACKDSL_HOT void next_row() noexcept { ++row; }
};

template <std::size_t Slots>
struct PeriodicRecompute<1, Slots> {
    static_assert(Slots > 0, "PeriodicRecompute Slots must be > 0");

    STACKDSL_HOT constexpr bool due(std::size_t = 0) const noexcept {
        return true;
    }

    STACKDSL_HOT constexpr void next_row() const noexcept {}
};

template <class T>
STACKDSL_HOT bool finite(T value) noexcept {
""",
    )


def patch_ridge_header() -> None:
    relative = "src/trading_dsl_engine/cpp_stream/cpp/stackdsl/ops/ridge.hpp"

    replace_once(
        relative,
        """template <std::size_t Groups, std::size_t K>
struct RidgeMetricState<Groups, K, true> {
    std::array<double, Groups * K * K> xx{};
    std::array<double, Groups * K> xy{};
    std::array<double, Groups> ywy{};
    std::array<double, Groups> wy{};
    std::array<double, Groups> weight{};
    std::array<double, Groups> weight_square{};
    std::array<std::uint8_t, Groups> initialized{};
    std::array<std::uint64_t, Groups> last_update{};
};

template <class Projection> struct projection_component {
""",
        """template <std::size_t Groups, std::size_t K>
struct RidgeMetricState<Groups, K, true> {
    std::array<double, Groups * K * K> xx{};
    std::array<double, Groups * K> xy{};
    std::array<double, Groups> ywy{};
    std::array<double, Groups> wy{};
    std::array<double, Groups> weight{};
    std::array<double, Groups> weight_square{};
    std::array<std::uint8_t, Groups> initialized{};
    std::array<std::uint64_t, Groups> last_update{};
};

template <
    std::size_t Groups,
    std::size_t K,
    bool NeedsMetrics,
    bool NeedsInference,
    bool Enabled
>
struct RidgeResultCache {};

template <
    std::size_t Groups,
    std::size_t K,
    bool NeedsMetrics,
    bool NeedsInference
>
struct RidgeResultCache<Groups, K, NeedsMetrics, NeedsInference, true> {
    std::array<double, Groups * K> beta{};
    std::array<std::uint8_t, Groups> initialized{};
    std::array<double, NeedsInference ? Groups * K : 0> standard_errors{};
    std::array<double, NeedsInference ? Groups * K : 0> tstats{};
    std::array<double, NeedsMetrics ? Groups : 0> sse{};
    std::array<double, NeedsMetrics ? Groups : 0> sst{};
    std::array<double, NeedsMetrics ? Groups : 0> r2{};
    std::array<double, NeedsInference ? Groups : 0> residual_variance{};
    std::array<double, NeedsInference ? Groups : 0> effective_df{};
    std::array<double, NeedsMetrics ? Groups : 0> effective_n{};

    STACKDSL_HOT void load(
        std::size_t group,
        std::array<double, K>& beta_out,
        std::array<double, K>& standard_errors_out,
        std::array<double, K>& tstats_out,
        double& sse_out,
        double& sst_out,
        double& r2_out,
        double& residual_variance_out,
        double& effective_df_out,
        double& effective_n_out
    ) const noexcept {
        const std::size_t vector_base = group * K;
        for (std::size_t j = 0; j < K; ++j) {
            beta_out[j] = beta[vector_base + j];
        }
        if constexpr (NeedsInference) {
            for (std::size_t j = 0; j < K; ++j) {
                standard_errors_out[j] =
                    standard_errors[vector_base + j];
                tstats_out[j] = tstats[vector_base + j];
            }
            residual_variance_out = residual_variance[group];
            effective_df_out = effective_df[group];
        }
        if constexpr (NeedsMetrics) {
            sse_out = sse[group];
            sst_out = sst[group];
            r2_out = r2[group];
            effective_n_out = effective_n[group];
        }
    }

    STACKDSL_HOT void store(
        std::size_t group,
        const std::array<double, K>& beta_in,
        const std::array<double, K>& standard_errors_in,
        const std::array<double, K>& tstats_in,
        double sse_in,
        double sst_in,
        double r2_in,
        double residual_variance_in,
        double effective_df_in,
        double effective_n_in
    ) noexcept {
        const std::size_t vector_base = group * K;
        for (std::size_t j = 0; j < K; ++j) {
            beta[vector_base + j] = beta_in[j];
        }
        if constexpr (NeedsInference) {
            for (std::size_t j = 0; j < K; ++j) {
                standard_errors[vector_base + j] =
                    standard_errors_in[j];
                tstats[vector_base + j] = tstats_in[j];
            }
            residual_variance[group] = residual_variance_in;
            effective_df[group] = effective_df_in;
        }
        if constexpr (NeedsMetrics) {
            sse[group] = sse_in;
            sst[group] = sst_in;
            r2[group] = r2_in;
            effective_n[group] = effective_n_in;
        }
        initialized[group] = 1;
    }
};

template <class Projection> struct projection_component {
""",
    )

    replace_once(
        relative,
        """template <std::size_t N, class Features, class Y, class Weights, class Out, std::uint64_t AlphaBits, std::uint64_t LambdaBits, bool Nonnegative, bool Stateful, class Projection, class Execution = DirectExecution<N>> struct RidgeNode;

template <std::size_t N, class Y, class Weights, class Out, std::uint64_t AlphaBits, std::uint64_t LambdaBits, bool Nonnegative, bool Stateful, class Projection, class Execution, class... FeatureSources>
struct RidgeNode<N, FeatureList<FeatureSources...>, Y, Weights, Out, AlphaBits, LambdaBits, Nonnegative, Stateful, Projection, Execution> {
""",
        """template <
    std::size_t N,
    class Features,
    class Y,
    class Weights,
    class Out,
    std::uint64_t AlphaBits,
    std::uint64_t LambdaBits,
    bool Nonnegative,
    bool Stateful,
    class Projection,
    class Execution = DirectExecution<N>,
    std::size_t RecomputeEvery = 1
>
struct RidgeNode;

template <
    std::size_t N,
    class Y,
    class Weights,
    class Out,
    std::uint64_t AlphaBits,
    std::uint64_t LambdaBits,
    bool Nonnegative,
    bool Stateful,
    class Projection,
    class Execution,
    std::size_t RecomputeEvery,
    class... FeatureSources
>
struct RidgeNode<
    N,
    FeatureList<FeatureSources...>,
    Y,
    Weights,
    Out,
    AlphaBits,
    LambdaBits,
    Nonnegative,
    Stateful,
    Projection,
    Execution,
    RecomputeEvery
> {
""",
    )

    replace_once(
        relative,
        """    static constexpr bool NeedsMetrics = Projections::needs_metrics;
    ridge_detail::RidgeState<Groups, K, Stateful> state{};
    ridge_detail::RidgeMetricState<Groups, K, Stateful && NeedsMetrics> metrics{};
    STACKDSL_HOT void setup() noexcept {}
""",
        """    static constexpr bool NeedsMetrics = Projections::needs_metrics;
    static_assert(RecomputeEvery > 0, "Ridge recompute interval must be > 0");
    ridge_detail::RidgeState<Groups, K, Stateful> state{};
    ridge_detail::RidgeMetricState<Groups, K, Stateful && NeedsMetrics> metrics{};
    PeriodicRecompute<RecomputeEvery, Groups> recompute_schedule{};
    ridge_detail::RidgeResultCache<
        Groups,
        K,
        NeedsMetrics,
        NeedsInference,
        (RecomputeEvery > 1)
    > result_cache{};
    STACKDSL_HOT void setup() noexcept {}
""",
    )

    replace_once(
        relative,
        """        for (std::size_t active = 0; active < active_count; ++active) {
            const std::size_t group = active_groups[active], local_matrix = active * K * K, local_vector = active * K;
            std::array<double, K * K> xx{};
""",
        """        for (std::size_t active = 0; active < active_count; ++active) {
            const std::size_t group = active_groups[active];
            const std::size_t local_matrix = active * K * K;
            const std::size_t local_vector = active * K;
            const bool recompute = recompute_schedule.due(group);
            std::array<double, K * K> xx{};
""",
    )

    replace_once(
        relative,
        """            std::array<double, K * K> system = xx;
""",
        """            if constexpr (RecomputeEvery > 1) {
                if (!recompute && result_cache.initialized[group]) {
                    result_cache.load(
                        group,
                        solved_betas[active],
                        standard_errors[active],
                        tstats[active],
                        sse_values[active],
                        sst_values[active],
                        r2_values[active],
                        residual_variances[active],
                        effective_df_values[active],
                        effective_n_values[active]
                    );
                    continue;
                }
            }
            std::array<double, K * K> system = xx;
""",
    )

    insert_before_last(
        relative,
        "\n        }\n        Projections::for_each(",
        """
            if constexpr (RecomputeEvery > 1) {
                result_cache.store(
                    group,
                    beta,
                    standard_errors[active],
                    tstats[active],
                    sse_values[active],
                    sst_values[active],
                    r2_values[active],
                    residual_variances[active],
                    effective_df_values[active],
                    effective_n_values[active]
                );
            }
""",
    )

    replace_once(
        relative,
        """        if constexpr (Stateful) ++state.t;
""",
        """        if constexpr (Stateful) ++state.t;
        recompute_schedule.next_row();
""",
    )


def patch_ir() -> None:
    replace_once(
        "src/trading_dsl_engine/ir/ops.py",
        """@dataclass(frozen=True, slots=True)
class RidgeOp:
    feature_widths: tuple[int, ...]
    has_weights: bool
    nonneg: bool = False
    is_stateful: bool = True

    @property
    def coefficient_width(self) -> int:
        return sum(self.feature_widths)
""",
        """@dataclass(frozen=True, slots=True)
class RidgeOp:
    feature_widths: tuple[int, ...]
    has_weights: bool
    nonneg: bool = False
    is_stateful: bool = True
    recompute_every: int = 1

    def __post_init__(self) -> None:
        if self.recompute_every < 1:
            raise ValueError("Ridge recompute_every must be >= 1")

    @property
    def coefficient_width(self) -> int:
        return sum(self.feature_widths)
""",
    )

    relative = "src/trading_dsl_engine/ir/frontend.py"
    regex_once(
        relative,
        r"""def _normalize_ridge\(
    call: Call,
\) -> tuple\[tuple\[Expr, \.\.\.\], Expr, Expr \| None, Expr, Expr, bool, bool\]:
.*?
    \)


def _resolve_universe_groups""",
        """def _normalize_ridge(
    call: Call,
) -> tuple[
    tuple[Expr, ...],
    Expr,
    Expr | None,
    Expr,
    Expr,
    bool,
    bool,
    int,
]:
    keyword_values = dict(call.kwargs)
    if len(keyword_values) != len(call.kwargs):
        raise FormulaIRCompileError("Ridge got duplicate keyword arguments")
    recompute_every = _literal_int(
        keyword_values.pop("recompute_every", Number(1.0)),
        "Ridge recompute_every",
        1,
    )
    if keyword_values:
        values = keyword_values
        if set(values) - {"y", "weights", "hl", "lambda_", "nonneg"}:
            raise FormulaIRCompileError("invalid Ridge keyword")
        if any(name not in values for name in ("y", "hl", "lambda_")):
            raise FormulaIRCompileError("Ridge missing y/hl/lambda_")
        features = call.args
        y = values["y"]
        weights = values.get("weights")
        hl = values["hl"]
        lam = values["lambda_"]
        nonneg = _literal_bool(
            values.get("nonneg", Number(0.0)), "Ridge nonneg"
        )
    else:
        args = call.args
        sentinel = (
            len(args) >= 5
            and isinstance(args[-1], Number)
            and float(args[-1].value) in (2.0, 3.0)
        )
        nonneg = _literal_bool(args[-1], "Ridge nonneg") if sentinel else False
        if sentinel:
            args = args[:-1]
        if len(args) >= 5:
            features, (y, weights, hl, lam) = args[:-4], args[-4:]
        elif len(args) >= 4:
            features, (y, hl, lam), weights = args[:-3], args[-3:], None
        else:
            raise FormulaIRCompileError(
                "Ridge expects features,y,[weights,]hl,lambda"
            )
    features = _flatten_cat_features(tuple(features))
    return (
        features,
        y,
        weights,
        hl,
        lam,
        nonneg,
        not (isinstance(hl, Number) and float(hl.value) == 0.0),
        recompute_every,
    )


def _resolve_universe_groups""",
    )

    replace_once(
        relative,
        """        if node.fn == "Ridge":
            features, y, weights, hl, lam, nonneg, stateful = _normalize_ridge(node)
""",
        """        if node.fn == "Ridge":
            (
                features,
                y,
                weights,
                hl,
                lam,
                nonneg,
                stateful,
                recompute_every,
            ) = _normalize_ridge(node)
""",
    )
    replace_once(
        relative,
        """            op = RidgeOp(widths, weights is not None, nonneg, stateful)
""",
        """            op = RidgeOp(
                widths,
                weights is not None,
                nonneg,
                stateful,
                recompute_every,
            )
""",
    )


def patch_dsl() -> None:
    relative = "src/trading_dsl_engine/base/dsl.py"
    regex_once(
        relative,
        r"""def Ridge\(\*features, y=None, weights=None, hl=None, lambda_=None, lam=None, nonneg=False\) -> Expr:  # noqa: N802
.*?


get_beta = op\("get_beta"\)""",
        """def Ridge(  # noqa: N802
    *features,
    y=None,
    weights=None,
    hl=None,
    lambda_=None,
    lam=None,
    nonneg=False,
    recompute_every=1,
) -> Expr:
    ridge_lambda = lambda_ if lambda_ is not None else lam
    recompute_kwargs = (
        {}
        if recompute_every == 1
        else {"recompute_every": recompute_every}
    )
    if y is None or hl is None or ridge_lambda is None:
        if weights is not None:
            raise TypeError(
                "Ridge positional form cannot combine positional "
                "y/hl/lambda with keyword weights"
            )
        return call(
            "Ridge",
            *features,
            3.0 if nonneg else 2.0,
            **recompute_kwargs,
        )
    if weights is None:
        return call(
            "Ridge",
            *features,
            y,
            1.0,
            hl,
            ridge_lambda,
            3.0 if nonneg else 2.0,
            **recompute_kwargs,
        )
    return call(
        "Ridge",
        *features,
        y,
        weights,
        hl,
        ridge_lambda,
        3.0 if nonneg else 2.0,
        **recompute_kwargs,
    )


get_beta = op("get_beta")""",
    )


def patch_codegen() -> None:
    replace_once(
        "src/trading_dsl_engine/cpp_stream/python/codegen.py",
        """            projection,
            execution,
        )
    if stage.kind == "groupby":
""",
        """            projection,
            execution,
            IntArg(physical.op.recompute_every),
        )
    if stage.kind == "groupby":
""",
    )


def main() -> None:
    patch_utils()
    patch_ridge_header()
    patch_ir()
    patch_dsl()
    patch_codegen()
    print("Applied Ridge recompute_every implementation")


if __name__ == "__main__":
    main()
