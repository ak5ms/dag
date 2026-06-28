#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>

#include "trading_dsl_engine/jax_ffi/nnqp/nnqp_eigen_impl.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {
constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
using Vec = Eigen::VectorXd;
using RowMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using ByteVec = Eigen::Array<uint8_t, Eigen::Dynamic, 1>;
using I64Vec = Eigen::Array<int64_t, Eigen::Dynamic, 1>;

enum class OpCode : int {
    Input,
    Literal,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,
    FloorDiv,
    Abs,
    Ln,
    Ceil,
    Floor,
    Round,
    Exp,
    Sign,
    Arctan,
    IsNan,
    Purify,
    Fraction,
    NormInv,
    XsNorm,
    Cache,
    Clip,
    XStd,
    XsRank,
    XsSort,
    Mean,
    Outer,
    Einsum,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Xor,
    FillNa,
    Where,
    Cat,
    Bspline,
    RbfBasis,
    FutureRbfBasisSum,
    Col,
    Cumsum,
    Ewm,
    RollMean,
    FFill,
    Shift,
    InstrumentBasisMean,
    Ridge,
    GetBeta,
    GetPreds,
    Group,
};

OpCode parse_opcode(const std::string& name) {
    if (name == "input") return OpCode::Input;
    if (name == "literal") return OpCode::Literal;
    if (name == "add") return OpCode::Add;
    if (name == "sub") return OpCode::Sub;
    if (name == "mul") return OpCode::Mul;
    if (name == "div") return OpCode::Div;
    if (name == "mod") return OpCode::Mod;
    if (name == "pow") return OpCode::Pow;
    if (name == "floordiv") return OpCode::FloorDiv;
    if (name == "abs") return OpCode::Abs;
    if (name == "ln") return OpCode::Ln;
    if (name == "ceil") return OpCode::Ceil;
    if (name == "floor") return OpCode::Floor;
    if (name == "round") return OpCode::Round;
    if (name == "exp") return OpCode::Exp;
    if (name == "sign") return OpCode::Sign;
    if (name == "arctan") return OpCode::Arctan;
    if (name == "isnan") return OpCode::IsNan;
    if (name == "purify") return OpCode::Purify;
    if (name == "fraction") return OpCode::Fraction;
    if (name == "norm_inv") return OpCode::NormInv;
    if (name == "xs_norm") return OpCode::XsNorm;
    if (name == "cache") return OpCode::Cache;
    if (name == "clip") return OpCode::Clip;
    if (name == "xstd") return OpCode::XStd;
    if (name == "xs_rank") return OpCode::XsRank;
    if (name == "xs_sort") return OpCode::XsSort;
    if (name == "mean") return OpCode::Mean;
    if (name == "outer") return OpCode::Outer;
    if (name == "einsum") return OpCode::Einsum;
    if (name == "eq") return OpCode::Eq;
    if (name == "ne") return OpCode::Ne;
    if (name == "lt") return OpCode::Lt;
    if (name == "gt") return OpCode::Gt;
    if (name == "le") return OpCode::Le;
    if (name == "ge") return OpCode::Ge;
    if (name == "and") return OpCode::And;
    if (name == "or") return OpCode::Or;
    if (name == "xor") return OpCode::Xor;
    if (name == "fillna") return OpCode::FillNa;
    if (name == "where") return OpCode::Where;
    if (name == "cat") return OpCode::Cat;
    if (name == "bspline") return OpCode::Bspline;
    if (name == "rbf_basis") return OpCode::RbfBasis;
    if (name == "future_rbf_basis_sum") return OpCode::FutureRbfBasisSum;
    if (name == "col") return OpCode::Col;
    if (name == "cumsum") return OpCode::Cumsum;
    if (name == "ewm") return OpCode::Ewm;
    if (name == "roll_mean") return OpCode::RollMean;
    if (name == "ffill") return OpCode::FFill;
    if (name == "shift") return OpCode::Shift;
    if (name == "instrument_basis_mean") return OpCode::InstrumentBasisMean;
    if (name == "ridge") return OpCode::Ridge;
    if (name == "get_beta") return OpCode::GetBeta;
    if (name == "get_preds") return OpCode::GetPreds;
    if (name == "group") return OpCode::Group;
    throw std::invalid_argument("unsupported C++ jax_flat opcode: " + name);
}

struct NodeSpec {
    OpCode opcode;
    std::vector<int> children;
    int input_index = -1;
    int state_index = -1;
    double literal = 0.0;
    double param = 0.0;
    int int_param = 0;
    int width = 1;
    std::vector<int> feature_widths;
    std::vector<NodeSpec> inner_nodes;
    int inner_output_id = -1;
    std::string str_param;
};


struct EinsumPlan {
    std::vector<std::vector<std::string>> inputs;
    std::vector<std::string> output;
    std::vector<std::string> summed;
};

static std::vector<std::string> parse_einsum_term(const std::string& term) {
    std::vector<std::string> labels;
    for (size_t i = 0; i < term.size();) {
        if (term.compare(i, 3, "...") == 0) {
            labels.push_back("...");
            i += 3;
        } else if (std::isalpha(static_cast<unsigned char>(term[i]))) {
            labels.emplace_back(1, term[i]);
            ++i;
        } else if (std::isspace(static_cast<unsigned char>(term[i]))) {
            ++i;
        } else {
            throw std::invalid_argument("invalid C++ jax_flat einsum label in pattern: " + term);
        }
    }
    return labels;
}

static std::vector<std::string> split_einsum_csv(const std::string& text) {
    std::vector<std::string> terms;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        terms.push_back(text.substr(start, comma == std::string::npos ? std::string::npos : comma - start));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return terms;
}

static int explicit_label_count(const std::vector<std::string>& labels) {
    return static_cast<int>(std::count_if(labels.begin(), labels.end(), [](const std::string& label) { return label != "..."; }));
}

static std::vector<std::string> expand_ellipsis(const std::vector<std::string>& labels, int ellipsis_rank, int operand_rank) {
    std::vector<std::string> out;
    const bool has_ellipsis = std::find(labels.begin(), labels.end(), "...") != labels.end();
    const int term_ellipsis_rank = has_ellipsis ? operand_rank - explicit_label_count(labels) : 0;
    if (term_ellipsis_rank < 0) throw std::invalid_argument("C++ jax_flat einsum operand rank is smaller than explicit labels");
    const int broadcast_pad = ellipsis_rank - term_ellipsis_rank;
    for (const auto& label : labels) {
        if (label == "...") {
            for (int i = 0; i < broadcast_pad; ++i) out.push_back("@" + std::to_string(i));
            for (int i = 0; i < term_ellipsis_rank; ++i) out.push_back("@" + std::to_string(broadcast_pad + i));
        } else {
            out.push_back(label);
        }
    }
    return out;
}

static EinsumPlan make_einsum_plan(const NodeSpec& spec) {
    const std::string& sub = spec.str_param;
    const size_t arrow = sub.find("->");
    if (arrow == std::string::npos) throw std::invalid_argument("C++ jax_flat einsum requires explicit output pattern: " + sub);
    const auto input_text = split_einsum_csv(sub.substr(0, arrow));
    if (input_text.size() != spec.children.size()) throw std::invalid_argument("C++ jax_flat einsum input count mismatch: " + sub);
    std::vector<std::vector<std::string>> parsed_inputs;
    parsed_inputs.reserve(input_text.size());
    int ellipsis_rank = 0;
    for (size_t i = 0; i < input_text.size(); ++i) {
        auto labels = parse_einsum_term(input_text[i]);
        parsed_inputs.push_back(labels);
        if (std::find(labels.begin(), labels.end(), "...") != labels.end()) {
            const int rank = spec.feature_widths.at(i) == 1 ? 1 : 2;
            ellipsis_rank = std::max(ellipsis_rank, rank - explicit_label_count(labels));
        }
    }
    auto parsed_output = parse_einsum_term(sub.substr(arrow + 2));
    EinsumPlan plan;
    for (size_t i = 0; i < parsed_inputs.size(); ++i) {
        const int rank = spec.feature_widths.at(i) == 1 ? 1 : 2;
        auto expanded = expand_ellipsis(parsed_inputs[i], ellipsis_rank, rank);
        const bool fixed_row_vector = rank == 2 && static_cast<int>(expanded.size()) == 1 && spec.feature_widths.at(i) > 1;
        if (static_cast<int>(expanded.size()) != rank && static_cast<int>(expanded.size()) != 0 && !fixed_row_vector) {
            throw std::invalid_argument("C++ jax_flat einsum rank mismatch: " + sub);
        }
        plan.inputs.push_back(std::move(expanded));
    }
    plan.output = expand_ellipsis(parsed_output, ellipsis_rank, explicit_label_count(parsed_output) + (std::find(parsed_output.begin(), parsed_output.end(), "...") != parsed_output.end() ? ellipsis_rank : 0));
    std::vector<std::string> all;
    for (const auto& term : plan.inputs) for (const auto& label : term) all.push_back(label);
    for (const auto& label : all) {
        if (std::find(plan.output.begin(), plan.output.end(), label) == plan.output.end()
            && std::find(plan.summed.begin(), plan.summed.end(), label) == plan.summed.end()) {
            plan.summed.push_back(label);
        }
    }
    return plan;
}


constexpr int kMaxEinsumAxes = 16;
constexpr int kMaxEinsumRank = 4;
constexpr int kMaxEinsumInputs = 8;

struct EinsumExecPlan {
    int n_inputs = 0;
    int n_labels = 0;
    int output_rank = 0;
    int summed_rank = 0;
    std::array<int, kMaxEinsumInputs> input_rank{};
    std::array<std::array<int, kMaxEinsumRank>, kMaxEinsumInputs> input_label_pos{};
    std::array<int, kMaxEinsumAxes> output_label_pos{};
    std::array<int, kMaxEinsumAxes> summed_label_pos{};
};

static int find_or_add_label(std::array<std::string, kMaxEinsumAxes>& labels, int& n_labels, const std::string& label) {
    for (int i = 0; i < n_labels; ++i) if (labels[static_cast<size_t>(i)] == label) return i;
    if (n_labels >= kMaxEinsumAxes) throw std::invalid_argument("C++ jax_flat einsum exceeds supported axis count");
    labels[static_cast<size_t>(n_labels)] = label;
    return n_labels++;
}

static EinsumExecPlan make_einsum_exec_plan(const NodeSpec& spec) {
    const EinsumPlan parsed = make_einsum_plan(spec);
    if (parsed.inputs.size() > kMaxEinsumInputs) throw std::invalid_argument("C++ jax_flat einsum exceeds supported input count");
    EinsumExecPlan plan;
    plan.n_inputs = static_cast<int>(parsed.inputs.size());
    std::array<std::string, kMaxEinsumAxes> labels{};
    for (size_t input_i = 0; input_i < parsed.inputs.size(); ++input_i) {
        const auto& term = parsed.inputs[input_i];
        if (term.size() > kMaxEinsumRank) throw std::invalid_argument("C++ jax_flat einsum exceeds supported operand rank");
        plan.input_rank[input_i] = static_cast<int>(term.size());
        for (size_t axis = 0; axis < term.size(); ++axis) {
            plan.input_label_pos[input_i][axis] = find_or_add_label(labels, plan.n_labels, term[axis]);
        }
    }
    if (parsed.output.size() > kMaxEinsumRank || parsed.summed.size() > kMaxEinsumRank) {
        throw std::invalid_argument("C++ jax_flat einsum exceeds supported output or contraction rank");
    }
    plan.output_rank = static_cast<int>(parsed.output.size());
    plan.summed_rank = static_cast<int>(parsed.summed.size());
    for (size_t axis = 0; axis < parsed.output.size(); ++axis) plan.output_label_pos[axis] = find_or_add_label(labels, plan.n_labels, parsed.output[axis]);
    for (size_t axis = 0; axis < parsed.summed.size(); ++axis) plan.summed_label_pos[axis] = find_or_add_label(labels, plan.n_labels, parsed.summed[axis]);
    return plan;
}

struct NodeValue {
    int rows_kind = 0;  // 0 => instrument-aligned rows; 1 => fixed flat rows; 2 => fixed matrix rows
    int width = 1;
    int fixed_rows = 1;
    std::vector<double> data;

    int rows(int n) const { return rows_kind == 0 ? n : fixed_rows; }
    int size(int n) const { return rows(n) * width; }
};

struct ValueState {
    Vec value;
    Vec weight;
    ByteVec initialized;
    I64Vec streak;
    ByteVec seen;

    explicit ValueState(int size = 0)
        : value(Vec::Zero(size)),
          weight(Vec::Zero(size)),
          initialized(ByteVec::Zero(size)),
          streak(I64Vec::Zero(size)),
          seen(ByteVec::Zero(size)) {}
};


struct RollingMeanState {
    RowMatrix buffer;
    Vec total;
    I64Vec valid_count;
    int pos = 0;
    int count = 0;
    int lookback = 0;

    RollingMeanState(int window = 0, int row_size = 0)
        : buffer(RowMatrix::Constant(window, row_size, NaN)),
          total(Vec::Zero(row_size)),
          valid_count(I64Vec::Zero(row_size)),
          lookback(window) {}
};

struct ShiftState {
    RowMatrix buffer;
    int pos = 0;
    int count = 0;
    int cap = 0;

    ShiftState(int capacity = 0, int row_size = 0)
        : buffer(RowMatrix::Constant(capacity, row_size, NaN)), cap(capacity) {}
};

struct RidgeState {
    RowMatrix xx;
    Vec xy;
    ByteVec has_xx;
    ByteVec has_xy;
    I64Vec last_xx;
    I64Vec last_xy;
    Vec beta;
    Vec preds;
    int64_t t = 0;
    int k = 0;

    RidgeState(int feature_count = 0, int instruments = 0)
        : xx(RowMatrix::Zero(feature_count, feature_count)),
          xy(Vec::Zero(feature_count)),
          has_xx(ByteVec::Zero(feature_count * feature_count)),
          has_xy(ByteVec::Zero(feature_count)),
          last_xx(I64Vec::Zero(feature_count * feature_count)),
          last_xy(I64Vec::Zero(feature_count)),
          beta(Vec::Zero(feature_count)),
          preds(Vec::Constant(instruments, NaN)),
          k(feature_count) {}
};

struct InstrumentBasisMeanState {
    RowMatrix num;
    RowMatrix den;
    ByteVec has_value;
    RowMatrix beta;
    Vec preds;
    int k = 0;

    InstrumentBasisMeanState(int feature_width = 0, int instruments = 0)
        : num(RowMatrix::Zero(instruments, feature_width)),
          den(RowMatrix::Zero(instruments, feature_width)),
          has_value(ByteVec::Zero(instruments * feature_width)),
          beta(RowMatrix::Zero(instruments, feature_width)),
          preds(Vec::Constant(instruments, NaN)),
          k(feature_width) {}
};

struct GroupState {
    int n_keys = 0;
    int capacity = 0;
    int n_groups = 1;
    std::vector<int> group_offsets;
    std::vector<int> group_indices;
    RowMatrix keys;
    ByteVec occupied;
    Vec cached_group_key;
    int cached_group_slot = -1;
    int cached_group_universe = -1;
    uint8_t cached_group_valid = 0;
    std::vector<Vec> inner_values;
    std::vector<Vec> inner_weights;
    std::vector<ByteVec> inner_initialized;
    std::vector<I64Vec> inner_streak;
    std::vector<ByteVec> inner_seen;

    GroupState(int key_count = 0, int slot_count = 0, int inner_state_count = 0, int instruments = 0)
        : n_keys(key_count),
          capacity(slot_count),
          keys(RowMatrix::Constant(slot_count, std::max(key_count, 1), NaN)),
          occupied(ByteVec::Zero(slot_count)),
          cached_group_key(Vec::Constant(std::max(key_count, 1), NaN)) {
        inner_values.reserve(static_cast<size_t>(inner_state_count));
        inner_weights.reserve(static_cast<size_t>(inner_state_count));
        inner_initialized.reserve(static_cast<size_t>(inner_state_count));
        inner_streak.reserve(static_cast<size_t>(inner_state_count));
        inner_seen.reserve(static_cast<size_t>(inner_state_count));
        for (int i = 0; i < inner_state_count; ++i) {
            inner_values.emplace_back(Vec::Zero(slot_count * instruments));
            inner_weights.emplace_back(Vec::Zero(slot_count * instruments));
            inner_initialized.emplace_back(ByteVec::Zero(slot_count * instruments));
            inner_streak.emplace_back(I64Vec::Zero(slot_count * instruments));
            inner_seen.emplace_back(ByteVec::Zero(slot_count * instruments));
        }
    }
};


inline bool finite(double x) { return std::isfinite(x); }

inline double norm_inv(double p) {
    if (std::isnan(p)) return NaN;
    if (p <= 0.0) return -std::numeric_limits<double>::infinity();
    if (p >= 1.0) return std::numeric_limits<double>::infinity();
    return Eigen::numext::ndtri(p);
}


inline bool same_double_key(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

int count_inner_states(const std::vector<NodeSpec>& nodes) {
    int count = 0;
    for (const auto& node : nodes) count = std::max(count, node.state_index + 1);
    return count;
}

class Runtime;
size_t count_inputs(const std::vector<NodeSpec>& nodes);

class State {
public:
    State(const Runtime* runtime, int n_instruments);
    int n_instruments() const { return n_instruments_; }

private:
    friend class Runtime;
    int n_instruments_;
    std::vector<NodeValue> values_;
    std::vector<ValueState> value_states_;
    std::vector<ShiftState> shift_states_;
    std::vector<RollingMeanState> rolling_mean_states_;
    std::vector<RidgeState> ridge_states_;
    std::vector<InstrumentBasisMeanState> instrument_basis_mean_states_;
    std::vector<GroupState> group_states_;
    std::vector<double> output_;
    std::vector<const double*> row_ptrs_;
};

class Runtime {
public:
    Runtime(std::vector<NodeSpec> nodes, int output_id, int n_states)
        : nodes_(std::move(nodes)), output_id_(output_id), n_states_(n_states) {
        if (output_id_ < 0 || output_id_ >= static_cast<int>(nodes_.size())) {
            throw std::invalid_argument("invalid C++ jax_flat output id");
        }
        assign_native_state_indices();
        einsum_plans_.reserve(nodes_.size());
        for (const auto& spec : nodes_) {
            einsum_plans_.push_back(spec.opcode == OpCode::Einsum ? make_einsum_exec_plan(spec) : EinsumExecPlan{});
        }
    }

    State init_state(int n_instruments) const { return State(this, n_instruments); }

    py::array_t<double> tick(State& state, py::args rows) const {
        py::array_t<double> out({static_cast<int64_t>(output_size(state))});
        tick_into(state, out, rows);
        return out;
    }

    void tick_into(State& state, py::array_t<double> out, py::args rows) const {
        auto out_buf = out.mutable_unchecked<1>();
        if (out_buf.shape(0) != static_cast<py::ssize_t>(output_size(state))) {
            throw std::invalid_argument("tick_into output width does not match C++ jax_flat root output size");
        }
        bind_tick_rows(state, rows);
        eval_row(state, state.row_ptrs_, &out_buf(0));
    }

    py::array_t<double> run_batch(State& state, py::args arrays) const {
        int64_t rows = -1;
        validate_batch(state, arrays, rows);
        py::array_t<double> out({rows, static_cast<int64_t>(output_size(state))});
        run_batch_into(state, out, arrays);
        return out;
    }

    void run_batch_into(State& state, py::array_t<double> out, py::args arrays) const {
        int64_t rows = -1;
        auto input_arrays = validate_batch(state, arrays, rows);
        auto out_buf = out.mutable_unchecked<2>();
        if (out_buf.shape(0) != rows || out_buf.shape(1) != static_cast<py::ssize_t>(output_size(state))) {
            throw std::invalid_argument("C++ jax_flat batch output shape mismatch");
        }
        const int n = state.n_instruments_;
        state.row_ptrs_.assign(input_arrays.size(), nullptr);
        std::vector<const double*> base_ptrs;
        base_ptrs.reserve(input_arrays.size());
        for (const auto& arr : input_arrays) base_ptrs.push_back(static_cast<const double*>(arr.request().ptr));
        for (int64_t t = 0; t < rows; ++t) {
            for (size_t i = 0; i < base_ptrs.size(); ++i) state.row_ptrs_[i] = base_ptrs[i] + t * n;
            eval_row(state, state.row_ptrs_, &out_buf(t, 0));
        }
    }

private:
    friend class State;
    std::vector<NodeSpec> nodes_;
    int output_id_;
    int n_states_;
    std::vector<EinsumExecPlan> einsum_plans_;

    int output_size(const State& state) const { return state.values_[static_cast<size_t>(output_id_)].size(state.n_instruments_); }

    void bind_tick_rows(State& state, py::args rows) const {
        if (rows.size() != state.row_ptrs_.size()) throw std::invalid_argument("wrong number of C++ jax_flat tick inputs");
        for (size_t i = 0; i < rows.size(); ++i) {
            py::array_t<double, py::array::c_style | py::array::forcecast> arr = py::cast<py::array>(rows[i]);
            auto info = arr.request();
            if (info.ndim != 1 || info.shape[0] != state.n_instruments_) {
                throw std::invalid_argument("C++ jax_flat tick inputs must be 1D float64 arrays matching n_instruments");
            }
            state.row_ptrs_[i] = static_cast<const double*>(info.ptr);
        }
    }

    std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> validate_batch(
        const State& state, py::args arrays, int64_t& rows) const {
        std::vector<py::array_t<double, py::array::c_style | py::array::forcecast>> input_arrays;
        input_arrays.reserve(arrays.size());
        for (py::handle item : arrays) {
            input_arrays.push_back(py::cast<py::array>(item));
            auto info = input_arrays.back().request();
            if (info.ndim != 2 || info.shape[1] != state.n_instruments_) {
                throw std::invalid_argument("C++ jax_flat batch inputs must be 2D float64 arrays matching n_instruments");
            }
            if (rows < 0) rows = info.shape[0];
            if (info.shape[0] != rows) throw std::invalid_argument("C++ jax_flat batch inputs must share row count");
        }
        if (rows < 0) throw std::invalid_argument("C++ jax_flat run_batch requires at least one input");
        return input_arrays;
    }

    static ValueState& value_state(State& state, const NodeSpec& spec) { return state.value_states_.at(static_cast<size_t>(spec.state_index)); }
    static ShiftState& shift_state(State& state, const NodeSpec& spec) { return state.shift_states_.at(static_cast<size_t>(spec.state_index)); }
    static RollingMeanState& rolling_mean_state(State& state, const NodeSpec& spec) { return state.rolling_mean_states_.at(static_cast<size_t>(spec.state_index)); }
    static RidgeState& ridge_state(State& state, const NodeSpec& spec) { return state.ridge_states_.at(static_cast<size_t>(spec.state_index)); }
    static InstrumentBasisMeanState& instrument_basis_mean_state(State& state, const NodeSpec& spec) { return state.instrument_basis_mean_states_.at(static_cast<size_t>(spec.state_index)); }
    static GroupState& group_state(State& state, const NodeSpec& spec) { return state.group_states_.at(static_cast<size_t>(spec.state_index)); }

    void assign_native_state_indices() {
        std::array<int, 6> next{0, 0, 0, 0, 0, 0};
        for (NodeSpec& spec : nodes_) {
            if (spec.state_index < 0) continue;
            spec.state_index = next[static_cast<size_t>(state_bucket(spec.opcode))]++;
        }
    }

    static int state_bucket(OpCode opcode) {
        switch (opcode) {
            case OpCode::RollMean: return 1;
            case OpCode::Shift: return 2;
            case OpCode::Ridge: return 3;
            case OpCode::InstrumentBasisMean: return 4;
            case OpCode::Group: return 5;
            default: return 0;
        }
    }
    static const NodeValue& child(const State& state, const NodeSpec& spec, size_t i) { return state.values_.at(static_cast<size_t>(spec.children.at(i))); }

    static double at(const NodeValue& v, int n, int row, int col = 0) {
        if (v.rows_kind == 1) return v.data[static_cast<size_t>(row < v.width ? row : 0)];
        const int width = v.width;
        const int r = v.rows(n) == 1 ? 0 : row;
        const int c = width == 1 ? 0 : col;
        return v.data[static_cast<size_t>(r * width + c)];
    }

    static void normalized_rbf_row(double x, int k, double* out) {
        if (!finite(x)) {
            for (int b = 0; b < k; ++b) out[b] = NaN;
            return;
        }
        const double clipped = std::min(std::max(x, 0.0), 1.0);
        const double denom = static_cast<double>(std::max(k - 1, 1));
        const double sigma = 1.0 / denom;
        double total = 0.0;
        for (int b = 0; b < k; ++b) {
            const double center = k == 1 ? 0.0 : static_cast<double>(b) / static_cast<double>(k - 1);
            const double z = (clipped - center) / sigma;
            const double val = std::exp(-0.5 * z * z);
            out[b] = val;
            total += val;
        }
        if (total <= 1e-18) {
            for (int b = 0; b < k; ++b) out[b] = 1.0 / static_cast<double>(k);
        } else {
            for (int b = 0; b < k; ++b) out[b] /= total;
        }
    }

    static void fill_rbf_basis(State& state, const NodeSpec& spec, NodeValue& dst_v) {
        const int n = state.n_instruments_;
        const int k = spec.width;
        const auto& ev = child(state, spec, 0);
        const auto& start = child(state, spec, 1);
        const auto& end = child(state, spec, 2);
        std::vector<double> row(static_cast<size_t>(k));
        for (int i = 0; i < n; ++i) {
            const double ts = at(ev, n, i);
            const double s = at(start, n, i);
            const double e = at(end, n, i);
            const double len = e - s;
            const bool in_session = finite(ts) && finite(s) && finite(e) && len > 0.0 && ts >= s && ts < e;
            if (!in_session) {
                for (int b = 0; b < k; ++b) dst_v.data[static_cast<size_t>(i * k + b)] = NaN;
                continue;
            }
            normalized_rbf_row((ts - s) / len, k, row.data());
            for (int b = 0; b < k; ++b) dst_v.data[static_cast<size_t>(i * k + b)] = row[static_cast<size_t>(b)];
        }
    }

    static void fill_future_rbf_basis_sum(State& state, const NodeSpec& spec, NodeValue& dst_v) {
        const int n = state.n_instruments_;
        const int k = spec.width;
        const int steps = std::max(1, static_cast<int>(std::llround(spec.param)));
        const auto& ev = child(state, spec, 0);
        const auto& start = child(state, spec, 1);
        const auto& end = child(state, spec, 2);
        std::vector<double> suffix(static_cast<size_t>((steps + 1) * k), 0.0);
        std::vector<double> vals(static_cast<size_t>(k));
        for (int g = steps - 1; g >= 0; --g) {
            normalized_rbf_row(static_cast<double>(g) / static_cast<double>(steps), k, vals.data());
            for (int b = 0; b < k; ++b) {
                suffix[static_cast<size_t>(g * k + b)] = suffix[static_cast<size_t>((g + 1) * k + b)] + vals[static_cast<size_t>(b)];
            }
        }
        for (int i = 0; i < n; ++i) {
            const double ts = at(ev, n, i);
            const double s = at(start, n, i);
            const double e = at(end, n, i);
            const double len = e - s;
            const bool valid_session = finite(ts) && finite(s) && finite(e) && len > 0.0;
            if (!valid_session) {
                for (int b = 0; b < k; ++b) dst_v.data[static_cast<size_t>(i * k + b)] = NaN;
                continue;
            }
            const double phase = (ts - s) / len;
            const double clipped = std::min(std::max(phase, 0.0), 1.0);
            int idx = static_cast<int>(std::floor(clipped * static_cast<double>(steps))) + 1;
            if (ts < s) idx = 0;
            else if (ts >= e) idx = steps;
            idx = std::min(std::max(idx, 0), steps);
            for (int b = 0; b < k; ++b) dst_v.data[static_cast<size_t>(i * k + b)] = suffix[static_cast<size_t>(idx * k + b)];
        }
    }


    int einsum_label_dim(State& state, const NodeSpec& spec, int label_pos, const EinsumExecPlan& plan) const {
        const int n = state.n_instruments_;
        int dim = 1;
        for (int input_i = 0; input_i < plan.n_inputs; ++input_i) {
            const auto& value = child(state, spec, static_cast<size_t>(input_i));
            for (int axis = 0; axis < plan.input_rank[static_cast<size_t>(input_i)]; ++axis) {
                if (plan.input_label_pos[static_cast<size_t>(input_i)][static_cast<size_t>(axis)] != label_pos) continue;
                const bool fixed_row_vector = plan.input_rank[static_cast<size_t>(input_i)] == 1 && value.rows(n) == 1 && value.width > 1;
                const int candidate = fixed_row_vector ? value.width : (axis == 0 ? value.rows(n) : value.width);
                if (dim != 1 && candidate != 1 && dim != candidate) {
                    throw std::invalid_argument("C++ jax_flat einsum label dimension mismatch");
                }
                dim = std::max(dim, candidate);
            }
        }
        return dim;
    }

    static size_t einsum_operand_offset(const NodeValue& value, int n, const EinsumExecPlan& plan, int input_i, const std::array<int, kMaxEinsumAxes>& idx) {
        if (plan.input_rank[static_cast<size_t>(input_i)] == 0) return 0;
        const int row_dim = value.rows(n);
        const int col_dim = value.width;
        int row = 0;
        int col = 0;
        for (int axis = 0; axis < plan.input_rank[static_cast<size_t>(input_i)]; ++axis) {
            const int label_pos = plan.input_label_pos[static_cast<size_t>(input_i)][static_cast<size_t>(axis)];
            const int v = idx[static_cast<size_t>(label_pos)];
            const bool fixed_row_vector = plan.input_rank[static_cast<size_t>(input_i)] == 1 && row_dim == 1 && col_dim > 1;
            if (fixed_row_vector) col = col_dim == 1 ? 0 : v;
            else if (axis == 0) row = row_dim == 1 ? 0 : v;
            else col = col_dim == 1 ? 0 : v;
        }
        return static_cast<size_t>(row * col_dim + col);
    }

    static bool increment_einsum_indices(std::array<int, kMaxEinsumAxes>& idx, const std::array<int, kMaxEinsumAxes>& label_positions, int rank, const std::array<int, kMaxEinsumAxes>& dims) {
        for (int axis = rank - 1; axis >= 0; --axis) {
            const int label_pos = label_positions[static_cast<size_t>(axis)];
            if (++idx[static_cast<size_t>(label_pos)] < dims[static_cast<size_t>(label_pos)]) return true;
            idx[static_cast<size_t>(label_pos)] = 0;
        }
        return false;
    }

    void eval_einsum(State& state, const NodeSpec& spec, const EinsumExecPlan& plan, NodeValue& dst_v) const {
        const int n = state.n_instruments_;
        std::array<int, kMaxEinsumAxes> dims{};
        for (int label_pos = 0; label_pos < plan.n_labels; ++label_pos) dims[static_cast<size_t>(label_pos)] = einsum_label_dim(state, spec, label_pos, plan);
        const int out_size = dst_v.size(n);
        int expected = 1;
        for (int axis = 0; axis < plan.output_rank; ++axis) expected *= dims[static_cast<size_t>(plan.output_label_pos[static_cast<size_t>(axis)])];
        if (expected != out_size) throw std::invalid_argument("C++ jax_flat einsum inferred output shape does not match lowered shape");
        std::array<int, kMaxEinsumAxes> idx{};
        std::array<int, kMaxEinsumAxes> output_idx{};
        for (int out_flat = 0; out_flat < out_size; ++out_flat) {
            int rem = out_flat;
            for (int axis = plan.output_rank - 1; axis >= 0; --axis) {
                const int label_pos = plan.output_label_pos[static_cast<size_t>(axis)];
                const int d = dims[static_cast<size_t>(label_pos)];
                output_idx[static_cast<size_t>(label_pos)] = rem % d;
                rem /= d;
            }
            idx.fill(0);
            for (int axis = 0; axis < plan.output_rank; ++axis) {
                const int label_pos = plan.output_label_pos[static_cast<size_t>(axis)];
                idx[static_cast<size_t>(label_pos)] = output_idx[static_cast<size_t>(label_pos)];
            }
            double sum = 0.0;
            while (true) {
                double prod = 1.0;
                for (int child_i = 0; child_i < plan.n_inputs; ++child_i) {
                    const auto& value = child(state, spec, static_cast<size_t>(child_i));
                    prod *= value.data[einsum_operand_offset(value, n, plan, child_i, idx)];
                }
                sum += prod;
                if (plan.summed_rank == 0 || !increment_einsum_indices(idx, plan.summed_label_pos, plan.summed_rank, dims)) break;
            }
            dst_v.data[static_cast<size_t>(out_flat)] = sum;
        }
    }

    void eval_instrument_basis_mean(State& state, const NodeSpec& spec, NodeValue& dst_v) const {
        auto& s = instrument_basis_mean_state(state, spec);
        const int n = state.n_instruments_;
        const int k = s.k;
        const auto& x = child(state, spec, 0);
        const auto& yv = child(state, spec, 1);
        const bool has_weights = spec.int_param != 0;
        const auto& weights = child(state, spec, has_weights ? 2 : 1);
        const auto& hlv = child(state, spec, has_weights ? 3 : 2);
        const double hl = at(hlv, n, 0, 0);
        const double rho = (!finite(hl) || hl <= 0.0) ? 0.0 : std::exp(std::log(0.5) / hl);
        const double alpha = std::min(std::max(1.0 - rho, 0.0), 1.0);
        for (int i = 0; i < n; ++i) {
            const double y = at(yv, n, i, 0);
            const double w = has_weights ? at(weights, n, i, 0) : 1.0;
            const bool valid_row = finite(y) && finite(w);
            bool finite_features = true;
            double pred = 0.0;
            for (int b = 0; b < k; ++b) {
                const double xb = at(x, n, i, b);
                finite_features = finite_features && finite(xb);
                pred += xb * s.beta(i, b);
            }
            s.preds[i] = valid_row && finite_features ? pred : NaN;
            dst_v.data[static_cast<size_t>(i)] = s.preds[i];
            for (int b = 0; b < k; ++b) {
                const double xb = at(x, n, i, b);
                const bool valid = valid_row && finite(xb);
                if (!valid) continue;
                const double num_new = xb * y * w;
                const double den_new = xb * w;
                const size_t hv_idx = static_cast<size_t>(i * k + b);
                s.num(i, b) = s.has_value[hv_idx] ? s.num(i, b) * (1.0 - alpha) + num_new * alpha : num_new;
                s.den(i, b) = s.has_value[hv_idx] ? s.den(i, b) * (1.0 - alpha) + den_new * alpha : den_new;
                s.has_value[hv_idx] = 1;
                const double candidate = s.den(i, b) != 0.0 ? s.num(i, b) / s.den(i, b) : NaN;
                if (finite(candidate)) s.beta(i, b) = candidate;
            }
        }
    }

    void eval_row(State& state, const std::vector<const double*>& input_ptrs, double* __restrict out_ptr) const {
        const int n = state.n_instruments_;
        for (size_t node_i = 0; node_i < nodes_.size(); ++node_i) {
            const NodeSpec& spec = nodes_[node_i];
            NodeValue& dst_v = state.values_[node_i];
            double* __restrict dst = dst_v.data.data();
            switch (spec.opcode) {
                case OpCode::Input: {
                    const double* __restrict src = input_ptrs.at(static_cast<size_t>(spec.input_index));
                    std::copy(src, src + n, dst);
                    break;
                }
                case OpCode::Literal:
                    std::fill(dst, dst + dst_v.size(n), spec.literal);
                    break;
                case OpCode::Add:
                case OpCode::Sub:
                case OpCode::Mul:
                case OpCode::Div:
                case OpCode::Mod:
                case OpCode::Pow:
                case OpCode::FloorDiv:
                case OpCode::Eq:
                case OpCode::Ne:
                case OpCode::Lt:
                case OpCode::Gt:
                case OpCode::Le:
                case OpCode::Ge:
                case OpCode::And:
                case OpCode::Or:
                case OpCode::Xor:
                case OpCode::FillNa:
                case OpCode::Clip:
                case OpCode::Cache: {
                    const auto& l = child(state, spec, 0);
                    const auto* r_ptr = spec.children.size() > 1 ? &child(state, spec, 1) : nullptr;
                    const int width = dst_v.width;
                    for (int i = 0; i < n; ++i) {
                        for (int c = 0; c < width; ++c) {
                            const double a = at(l, n, i, c);
                            const double b = r_ptr == nullptr ? NaN : at(*r_ptr, n, i, c);
                            double out = NaN;
                            if (spec.opcode == OpCode::Add) out = a + b;
                            else if (spec.opcode == OpCode::Sub) out = a - b;
                            else if (spec.opcode == OpCode::Mul) out = a * b;
                            else if (spec.opcode == OpCode::Div) out = b == 0.0 ? NaN : a / b;
                            else if (spec.opcode == OpCode::Mod) out = b == 0.0 ? NaN : a - std::floor(a / b) * b;
                            else if (spec.opcode == OpCode::Pow) out = std::pow(a, b);
                            else if (spec.opcode == OpCode::FloorDiv) out = b == 0.0 ? NaN : std::floor(a / b);
                            else if (spec.opcode == OpCode::FillNa) out = std::isnan(a) ? b : a;
                            else if (spec.opcode == OpCode::Clip) out = std::min(std::max(a, b), at(child(state, spec, 2), n, i, c));
                            else if (spec.opcode == OpCode::Cache) out = a;
                            else if (std::isnan(a) || std::isnan(b)) out = NaN;
                            else if (spec.opcode == OpCode::Eq) out = a == b ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Ne) out = a != b ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Lt) out = a < b ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Gt) out = a > b ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Le) out = a <= b ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Ge) out = a >= b ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::And) out = (a != 0.0 && b != 0.0) ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Or) out = (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Xor) out = ((a != 0.0) != (b != 0.0)) ? 1.0 : 0.0;
                            dst[static_cast<size_t>(i * width + c)] = out;
                        }
                    }
                    break;
                }
                case OpCode::Where: {
                    const auto& cond = child(state, spec, 0);
                    const auto& tv = child(state, spec, 1);
                    const auto& fv = child(state, spec, 2);
                    const int width = dst_v.width;
                    for (int i = 0; i < n; ++i) {
                        for (int c = 0; c < width; ++c) {
                            dst[static_cast<size_t>(i * width + c)] = at(cond, n, i, c) != 0.0 ? at(tv, n, i, c) : at(fv, n, i, c);
                        }
                    }
                    break;
                }
                case OpCode::Abs:
                case OpCode::Ln:
                case OpCode::Ceil:
                case OpCode::Floor:
                case OpCode::Round:
                case OpCode::Exp:
                case OpCode::Sign:
                case OpCode::Arctan:
                case OpCode::IsNan:
                case OpCode::Purify:
                case OpCode::Fraction:
                case OpCode::NormInv: {
                    const auto& x = child(state, spec, 0);
                    for (int i = 0; i < dst_v.size(n); ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        if (spec.opcode == OpCode::Abs) dst[i] = std::abs(v);
                        else if (spec.opcode == OpCode::Ln) dst[i] = std::log(v);
                        else if (spec.opcode == OpCode::Ceil) dst[i] = std::ceil(v);
                        else if (spec.opcode == OpCode::Floor) dst[i] = std::floor(v);
                        else if (spec.opcode == OpCode::Round) dst[i] = std::round(v);
                        else if (spec.opcode == OpCode::Exp) dst[i] = std::exp(v);
                        else if (spec.opcode == OpCode::Sign) dst[i] = (v > 0.0) - (v < 0.0);
                        else if (spec.opcode == OpCode::Arctan) dst[i] = std::atan(v);
                        else if (spec.opcode == OpCode::IsNan) dst[i] = std::isnan(v) ? 1.0 : 0.0;
                        else if (spec.opcode == OpCode::Purify) dst[i] = finite(v) ? v : NaN;
                        else if (spec.opcode == OpCode::Fraction) dst[i] = v - std::floor(v);
                        else if (spec.opcode == OpCode::NormInv) dst[i] = norm_inv(v);
                        else dst[i] = v;
                    }
                    break;
                }
                case OpCode::Mean: {
                    const auto& x = child(state, spec, 0);
                    double sum = 0.0;
                    int count = 0;
                    for (double v : x.data) if (finite(v)) { sum += v; ++count; }
                    dst[0] = count ? sum / count : NaN;
                    break;
                }
                case OpCode::Outer: {
                    const auto& x = child(state, spec, 0);
                    Eigen::Map<const Vec> x_vec(x.data.data(), n);
                    Eigen::Map<RowMatrix> out(dst, n, dst_v.width);
                    out.noalias() = x_vec * x_vec.transpose();
                    break;
                }
                case OpCode::Einsum: {
                    eval_einsum(state, spec, einsum_plans_[node_i], dst_v);
                    break;
                }
                case OpCode::XsNorm: {
                    const auto& x = child(state, spec, 0);
                    Eigen::Map<const Vec> x_vec(x.data.data(), n);
                    Eigen::Map<Vec> out(dst, n);
                    const double denom = x_vec.array().isFinite().select(x_vec.array().abs(), 0.0).sum();
                    if (denom > 0.0) out.array() = x_vec.array() / denom;
                    else out.array().setConstant(NaN);
                    break;
                }
                case OpCode::XStd: {
                    const auto& x = child(state, spec, 0);
                    double sum = 0.0;
                    int count = 0;
                    for (int i = 0; i < n; ++i) if (finite(x.data[static_cast<size_t>(i)])) { sum += x.data[static_cast<size_t>(i)]; ++count; }
                    const double denom = std::max(count, 1);
                    const double mean = sum / denom;
                    double ss = 0.0;
                    for (int i = 0; i < n; ++i) if (finite(x.data[static_cast<size_t>(i)])) { const double d = x.data[static_cast<size_t>(i)] - mean; ss += d * d; }
                    const double stdev = std::sqrt(std::max(ss / denom, 0.0));
                    for (int i = 0; i < n; ++i) dst[i] = finite(x.data[static_cast<size_t>(i)]) && stdev > 0.0 ? (x.data[static_cast<size_t>(i)] - mean) / stdev : NaN;
                    break;
                }
                case OpCode::XsSort: {
                    const auto& x = child(state, spec, 0);
                    std::copy(x.data.begin(), x.data.begin() + n, dst);
                    std::sort(dst, dst + n, [](double a, double b) {
                        if (std::isnan(a)) return false;
                        if (std::isnan(b)) return true;
                        return a < b;
                    });
                    break;
                }
                case OpCode::XsRank: {
                    const auto& x = child(state, spec, 0);
                    std::vector<double> compact;
                    compact.reserve(static_cast<size_t>(n));
                    for (int i = 0; i < n; ++i) if (finite(x.data[static_cast<size_t>(i)])) compact.push_back(x.data[static_cast<size_t>(i)]);
                    std::sort(compact.begin(), compact.end());
                    const double denom = static_cast<double>(compact.size() + 1);
                    for (int i = 0; i < n; ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        dst[i] = finite(v) ? norm_inv(static_cast<double>(std::upper_bound(compact.begin(), compact.end(), v) - compact.begin()) / denom) : NaN;
                    }
                    break;
                }
                case OpCode::Cat: {
                    int off = 0;
                    for (int child_id : spec.children) {
                        const auto& x = state.values_[static_cast<size_t>(child_id)];
                        for (int i = 0; i < n; ++i) {
                            for (int c = 0; c < x.width; ++c) dst[static_cast<size_t>(i * dst_v.width + off + c)] = at(x, n, i, c);
                        }
                        off += x.width;
                    }
                    break;
                }
                case OpCode::Bspline: {
                    const auto& x = child(state, spec, 0);
                    const int k = spec.width;
                    const double sigma = 1.0 / static_cast<double>(k);
                    for (int i = 0; i < n; ++i) {
                        const double raw = x.data[static_cast<size_t>(i)];
                        if (std::isnan(raw)) {
                            for (int b = 0; b < k; ++b) dst[static_cast<size_t>(i * k + b)] = NaN;
                            continue;
                        }
                        const double clipped = std::min(std::max(raw, 0.0), 1.0);
                        double total = 0.0;
                        for (int b = 0; b < k; ++b) {
                            const double center = static_cast<double>(b) / static_cast<double>(k);
                            const double dist = std::abs(clipped - center);
                            const double circ = std::min(dist, 1.0 - dist);
                            const double val = std::exp(-0.5 * (circ / sigma) * (circ / sigma));
                            dst[static_cast<size_t>(i * k + b)] = val;
                            total += val;
                        }
                        for (int b = 0; b < k; ++b) dst[static_cast<size_t>(i * k + b)] = total <= 1e-18 ? 1.0 / k : dst[static_cast<size_t>(i * k + b)] / total;
                    }
                    break;
                }
                case OpCode::RbfBasis:
                    fill_rbf_basis(state, spec, dst_v);
                    break;
                case OpCode::FutureRbfBasisSum:
                    fill_future_rbf_basis_sum(state, spec, dst_v);
                    break;
                case OpCode::Col: {
                    const auto& x = child(state, spec, 0);
                    const int col = spec.int_param;
                    for (int i = 0; i < n; ++i) dst[i] = at(x, n, i, col);
                    break;
                }
                case OpCode::Cumsum: {
                    auto& s = value_state(state, spec);
                    const auto& x = child(state, spec, 0);
                    for (int i = 0; i < dst_v.size(n); ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        if (finite(v)) { s.value[i] += v; s.initialized[i] = 1; dst[i] = s.value[i]; }
                        else dst[i] = NaN;
                    }
                    break;
                }
                case OpCode::Ewm: {
                    auto& s = value_state(state, spec);
                    const auto& x = child(state, spec, 0);
                    const double alpha = 2.0 / (spec.param + 1.0);
                    const double old_wt_factor = 1.0 - alpha;
                    const int min_periods = spec.int_param / 4 - 1;
                    const bool ignore_na = (spec.int_param & 1) != 0;
                    const bool adjust = (spec.int_param & 2) != 0;
                    for (int i = 0; i < dst_v.size(n); ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        const bool is_observation = finite(v);
                        double old_wt = s.weight[i];
                        if (s.initialized[i] && (is_observation || !ignore_na)) old_wt *= old_wt_factor;
                        if (is_observation) {
                            if (s.initialized[i]) {
                                double new_wt = adjust ? 1.0 : alpha;
                                if (!adjust && std::abs(alpha - 0.5) <= 1e-12) new_wt = 1.0 - old_wt;
                                if (s.value[i] != v) s.value[i] = (old_wt * s.value[i] + new_wt * v) / (old_wt + new_wt);
                                old_wt = adjust ? old_wt + new_wt : 1.0;
                            } else {
                                s.value[i] = v;
                                s.initialized[i] = 1;
                                old_wt = 1.0;
                            }
                            s.streak[i] += 1;
                        }
                        s.weight[i] = old_wt;
                        const bool enough = min_periods < 0 || s.streak[i] >= min_periods;
                        dst[i] = (s.initialized[i] && enough) ? s.value[i] : NaN;
                    }
                    break;
                }
                case OpCode::RollMean: {
                    auto& s = rolling_mean_state(state, spec);
                    const auto& x = child(state, spec, 0);
                    const int row_size = dst_v.size(n);
                    const int min_periods = static_cast<int>(std::llround(spec.param));
                    for (int i = 0; i < row_size; ++i) {
                        const double old = s.buffer(s.pos, i);
                        const bool old_valid = finite(old);
                        const double v = x.data[static_cast<size_t>(i)];
                        const bool valid = finite(v);
                        s.total[i] += (valid ? v : 0.0) - (old_valid ? old : 0.0);
                        s.valid_count[i] += (valid ? 1 : 0) - (old_valid ? 1 : 0);
                        s.buffer(s.pos, i) = v;
                        dst[i] = (s.count + 1 >= min_periods && s.valid_count[i] >= min_periods)
                            ? s.total[i] / static_cast<double>(s.valid_count[i])
                            : NaN;
                    }
                    s.pos = (s.pos + 1) % s.lookback;
                    s.count = std::min(s.count + 1, s.lookback);
                    break;
                }
                case OpCode::FFill: {
                    auto& s = value_state(state, spec);
                    const auto& x = child(state, spec, 0);
                    const int limit = spec.int_param;
                    for (int i = 0; i < dst_v.size(n); ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        if (finite(v)) { s.value[i] = v; s.seen[i] = 1; s.streak[i] = 0; dst[i] = v; }
                        else { s.streak[i] += 1; const bool allowed = s.seen[i] && (limit < 0 || s.streak[i] <= limit); dst[i] = allowed ? s.value[i] : NaN; }
                    }
                    break;
                }
                case OpCode::Shift: {
                    auto& s = shift_state(state, spec);
                    const auto& x = child(state, spec, 0);
                    const auto& lag = child(state, spec, 1);
                    const int width = dst_v.width;
                    for (int i = 0; i < n; ++i) {
                        const double raw_lag = at(lag, n, i, 0);
                        int lag_i = finite(raw_lag) ? static_cast<int>(std::llround(raw_lag)) : -1;
                        lag_i = std::min(std::max(lag_i, 0), s.cap - 1);
                        for (int c = 0; c < width; ++c) {
                            if (!finite(raw_lag)) dst[static_cast<size_t>(i * width + c)] = NaN;
                            else if (lag_i == 0) dst[static_cast<size_t>(i * width + c)] = at(x, n, i, c);
                            else if (s.count >= lag_i) { int rp = (s.pos - lag_i) % s.cap; if (rp < 0) rp += s.cap; dst[static_cast<size_t>(i * width + c)] = s.buffer(rp, i * width + c); }
                            else dst[static_cast<size_t>(i * width + c)] = NaN;
                        }
                    }
                    std::copy(x.data.begin(), x.data.begin() + n * width, s.buffer.row(s.pos).data());
                    s.pos = (s.pos + 1) % s.cap;
                    s.count = std::min(s.count + 1, s.cap);
                    break;
                }
                case OpCode::InstrumentBasisMean:
                    eval_instrument_basis_mean(state, spec, dst_v);
                    break;
                case OpCode::Ridge:
                    eval_ridge(state, spec, dst_v);
                    break;
                case OpCode::GetBeta: {
                    const auto& child_node = nodes_[static_cast<size_t>(spec.children[0])];
                    if (child_node.opcode == OpCode::Ridge) {
                        const auto& s = state.ridge_states_[static_cast<size_t>(child_node.state_index)];
                        std::copy(s.beta.data(), s.beta.data() + s.beta.size(), dst);
                    } else if (child_node.opcode == OpCode::InstrumentBasisMean) {
                        const auto& s = state.instrument_basis_mean_states_[static_cast<size_t>(child_node.state_index)];
                        std::copy(s.beta.data(), s.beta.data() + s.beta.size(), dst);
                    } else {
                        throw std::invalid_argument("C++ jax_flat get_beta expects Ridge or InstrumentBasisMean child");
                    }
                    break;
                }
                case OpCode::GetPreds: {
                    const auto& r = child(state, spec, 0);
                    std::copy(r.data.begin(), r.data.end(), dst);
                    break;
                }
                case OpCode::Group:
                    eval_group(state, spec, dst_v);
                    break;
            }
        }
        const auto& root = state.values_[static_cast<size_t>(output_id_)];
        std::copy(root.data.begin(), root.data.begin() + root.size(n), out_ptr);
    }



    static double eval_scalar_opcode(OpCode opcode, double a, double b, double c) {
        switch (opcode) {
            case OpCode::Add: return a + b;
            case OpCode::Sub: return a - b;
            case OpCode::Mul: return a * b;
            case OpCode::Div: return b == 0.0 ? NaN : a / b;
            case OpCode::Mod: return b == 0.0 ? NaN : a - std::floor(a / b) * b;
            case OpCode::Pow: return std::pow(a, b);
            case OpCode::FloorDiv: return b == 0.0 ? NaN : std::floor(a / b);
            case OpCode::Eq: return a == b ? 1.0 : 0.0;
            case OpCode::Ne: return a != b ? 1.0 : 0.0;
            case OpCode::Lt: return a < b ? 1.0 : 0.0;
            case OpCode::Gt: return a > b ? 1.0 : 0.0;
            case OpCode::Le: return a <= b ? 1.0 : 0.0;
            case OpCode::Ge: return a >= b ? 1.0 : 0.0;
            case OpCode::And: return (a != 0.0 && b != 0.0) ? 1.0 : 0.0;
            case OpCode::Or: return (a != 0.0 || b != 0.0) ? 1.0 : 0.0;
            case OpCode::Xor: return ((a != 0.0) != (b != 0.0)) ? 1.0 : 0.0;
            case OpCode::FillNa: return std::isnan(a) ? b : a;
            case OpCode::Where: return a != 0.0 ? b : c;
            case OpCode::Abs: return std::abs(a);
            case OpCode::Ln: return std::log(a);
            case OpCode::Ceil: return std::ceil(a);
            case OpCode::Floor: return std::floor(a);
            case OpCode::Round: return std::round(a);
            case OpCode::Exp: return std::exp(a);
            case OpCode::Sign: return (a > 0.0) - (a < 0.0);
            case OpCode::Arctan: return std::atan(a);
            case OpCode::IsNan: return std::isnan(a) ? 1.0 : 0.0;
            case OpCode::Purify: return finite(a) ? a : NaN;
            case OpCode::Fraction: return a - std::floor(a);
            case OpCode::NormInv: return norm_inv(a);
            case OpCode::Cache: return a;
            case OpCode::Clip: return std::min(std::max(a, b), c);
            default: return NaN;
        }
    }

    double eval_group_inner_node(GroupState& s, const NodeSpec& node, int n, int row, int slot_i, std::vector<double>& values) const {
        const auto child_value = [&](size_t i) { return values[static_cast<size_t>(node.children.at(i))]; };
        const size_t off = static_cast<size_t>(slot_i * n + row);
        switch (node.opcode) {
            case OpCode::Input:
                return values[0];
            case OpCode::Literal:
                return node.literal;
            case OpCode::Cumsum: {
                const double v = child_value(0);
                auto& state_v = s.inner_values.at(static_cast<size_t>(node.state_index));
                auto& init = s.inner_initialized.at(static_cast<size_t>(node.state_index));
                if (finite(v)) {
                    state_v[off] += v;
                    init[off] = 1;
                    return state_v[off];
                }
                return NaN;
            }
            case OpCode::Ewm: {
                const double v = child_value(0);
                auto& state_v = s.inner_values.at(static_cast<size_t>(node.state_index));
                auto& weight = s.inner_weights.at(static_cast<size_t>(node.state_index));
                auto& init = s.inner_initialized.at(static_cast<size_t>(node.state_index));
                auto& count = s.inner_streak.at(static_cast<size_t>(node.state_index));
                const double alpha = 2.0 / (node.param + 1.0);
                const double old_wt_factor = 1.0 - alpha;
                const int min_periods = node.int_param / 4 - 1;
                const bool ignore_na = (node.int_param & 1) != 0;
                const bool adjust = (node.int_param & 2) != 0;
                const bool is_observation = finite(v);
                double old_wt = weight[off];
                if (init[off] && (is_observation || !ignore_na)) old_wt *= old_wt_factor;
                if (is_observation) {
                    if (init[off]) {
                        double new_wt = adjust ? 1.0 : alpha;
                        if (!adjust && std::abs(alpha - 0.5) <= 1e-12) new_wt = 1.0 - old_wt;
                        if (state_v[off] != v) state_v[off] = (old_wt * state_v[off] + new_wt * v) / (old_wt + new_wt);
                        old_wt = adjust ? old_wt + new_wt : 1.0;
                    } else {
                        state_v[off] = v;
                        init[off] = 1;
                        old_wt = 1.0;
                    }
                    count[off] += 1;
                }
                weight[off] = old_wt;
                return (init[off] && (min_periods < 0 || count[off] >= min_periods)) ? state_v[off] : NaN;
            }
            case OpCode::FFill: {
                const double v = child_value(0);
                auto& state_v = s.inner_values.at(static_cast<size_t>(node.state_index));
                auto& seen = s.inner_seen.at(static_cast<size_t>(node.state_index));
                auto& streak = s.inner_streak.at(static_cast<size_t>(node.state_index));
                if (finite(v)) {
                    state_v[off] = v;
                    seen[off] = 1;
                    streak[off] = 0;
                    return v;
                }
                streak[off] += 1;
                const bool allowed = seen[off] && (node.int_param < 0 || streak[off] <= node.int_param);
                return allowed ? state_v[off] : NaN;
            }
            default: {
                const double a = node.children.empty() ? NaN : child_value(0);
                const double b = node.children.size() < 2 ? NaN : child_value(1);
                const double c = node.children.size() < 3 ? NaN : child_value(2);
                return eval_scalar_opcode(node.opcode, a, b, c);
            }
        }
    }

    void eval_group_row(
        State& state,
        const NodeSpec& spec,
        GroupState& s,
        const NodeValue& lhs,
        int group_i,
        int row,
        std::vector<double>& inner_values_tmp,
        NodeValue& dst_v
    ) const {
        const int n = state.n_instruments_;
        const int slot_i = find_or_insert_group_slot(state, spec, row, group_i);
        inner_values_tmp.assign(spec.inner_nodes.size(), NaN);
        if (!inner_values_tmp.empty()) inner_values_tmp[0] = at(lhs, n, row, 0);
        for (size_t node_i = 0; node_i < spec.inner_nodes.size(); ++node_i) {
            inner_values_tmp[node_i] = eval_group_inner_node(s, spec.inner_nodes[node_i], n, row, slot_i, inner_values_tmp);
        }
        dst_v.data[static_cast<size_t>(row)] = inner_values_tmp.at(static_cast<size_t>(spec.inner_output_id));
    }

    void eval_group(State& state, const NodeSpec& spec, NodeValue& dst_v) const {
        GroupState& s = group_state(state, spec);
        const int lhs_child = s.n_keys;
        const auto& lhs = state.values_[static_cast<size_t>(spec.children[static_cast<size_t>(lhs_child)])];
        std::fill(dst_v.data.begin(), dst_v.data.end(), NaN);
        std::vector<double> inner_values_tmp(spec.inner_nodes.size(), NaN);
        for (int group_i = 0; group_i < s.n_groups; ++group_i) {
            for (int pos = s.group_offsets[static_cast<size_t>(group_i)]; pos < s.group_offsets[static_cast<size_t>(group_i + 1)]; ++pos) {
                eval_group_row(state, spec, s, lhs, group_i, s.group_indices[static_cast<size_t>(pos)], inner_values_tmp, dst_v);
            }
        }
    }

    bool row_matches_key(State& state, const NodeSpec& spec, int row, const Vec& key) const {
        const int n = state.n_instruments_;
        for (int k = 0; k < static_cast<int>(key.size()); ++k) {
            const double incoming = at(state.values_[static_cast<size_t>(spec.children[static_cast<size_t>(k)])], n, row, 0);
            if (!same_double_key(key[k], incoming)) return false;
        }
        return true;
    }

    bool row_matches_slot(State& state, const NodeSpec& spec, int row, const GroupState& s, int slot_i) const {
        const int n = state.n_instruments_;
        for (int k = 0; k < s.n_keys; ++k) {
            const double stored = s.keys(slot_i, k);
            const double incoming = at(state.values_[static_cast<size_t>(spec.children[static_cast<size_t>(k)])], n, row, 0);
            if (!same_double_key(stored, incoming)) return false;
        }
        return true;
    }

    void cache_group_slot(State& state, const NodeSpec& spec, int row, int group_i, GroupState& s, int slot_i) const {
        const int n = state.n_instruments_;
        for (int k = 0; k < s.n_keys; ++k) {
            s.cached_group_key[k] = at(state.values_[static_cast<size_t>(spec.children[static_cast<size_t>(k)])], n, row, 0);
        }
        s.cached_group_slot = slot_i;
        s.cached_group_universe = group_i;
        s.cached_group_valid = 1;
    }

    int find_or_insert_group_slot(State& state, const NodeSpec& spec, int row, int group_i) const {
        GroupState& s = group_state(state, spec);
        if (
            s.cached_group_valid && s.cached_group_slot >= 0 &&
            s.cached_group_universe == group_i &&
            row_matches_key(state, spec, row, s.cached_group_key)
        ) {
            return s.cached_group_slot;
        }
        const int begin = group_i * s.capacity;
        const int end = begin + s.capacity;
        for (int slot_i = begin; slot_i < end; ++slot_i) {
            if (!s.occupied[slot_i]) continue;
            if (row_matches_slot(state, spec, row, s, slot_i)) {
                cache_group_slot(state, spec, row, group_i, s, slot_i);
                return slot_i;
            }
        }
        for (int slot_i = begin; slot_i < end; ++slot_i) {
            if (s.occupied[slot_i]) continue;
            s.occupied[slot_i] = 1;
            const int n = state.n_instruments_;
            for (int k = 0; k < s.n_keys; ++k) {
                s.keys(slot_i, k) = at(state.values_[static_cast<size_t>(spec.children[static_cast<size_t>(k)])], n, row, 0);
            }
            cache_group_slot(state, spec, row, group_i, s, slot_i);
            return slot_i;
        }
        throw std::runtime_error("C++ jax_flat groupby capacity exceeded");
    }

    void eval_ridge(State& state, const NodeSpec& spec, NodeValue& dst_v) const {
        const int n = state.n_instruments_;
        RidgeState& s = ridge_state(state, spec);
        const int k = s.k;
        const bool has_weights = spec.children.size() == spec.feature_widths.size() + 4;
        const int y_child = static_cast<int>(spec.feature_widths.size());
        const int w_child = y_child + 1;
        const int hl_child = y_child + (has_weights ? 2 : 1);
        const int lam_child = hl_child + 1;
        const auto& yv = state.values_[static_cast<size_t>(spec.children[y_child])];
        const auto& wv = has_weights ? state.values_[static_cast<size_t>(spec.children[w_child])] : yv;
        const auto& hlv = state.values_[static_cast<size_t>(spec.children[hl_child])];
        const auto& lamv = state.values_[static_cast<size_t>(spec.children[lam_child])];

        RowMatrix xmat = RowMatrix::Constant(n, k, NaN);
        for (int row = 0; row < n; ++row) {
            int off = 0;
            for (size_t f = 0; f < spec.feature_widths.size(); ++f) {
                const auto& feat = state.values_[static_cast<size_t>(spec.children[f])];
                for (int c = 0; c < spec.feature_widths[f]; ++c) xmat(row, off++) = at(feat, n, row, c);
            }
        }

        std::vector<uint8_t> row_valid(static_cast<size_t>(n), 0);
        for (int row = 0; row < n; ++row) {
            bool valid_row = finite(at(yv, n, row, 0));
            double pred = 0.0;
            for (int a = 0; a < k; ++a) {
                const double x = xmat(row, a);
                valid_row &= finite(x);
                pred += (finite(x) ? x : 0.0) * s.beta[a];
            }
            row_valid[static_cast<size_t>(row)] = valid_row ? 1 : 0;
            s.preds[row] = valid_row ? pred : NaN;
        }

        RowMatrix xx_new = RowMatrix::Zero(k, k);
        Vec xy_new = Vec::Zero(k);
        ByteVec xx_valid = ByteVec::Zero(k * k);
        ByteVec xy_valid = ByteVec::Zero(k);
        for (int row = 0; row < n; ++row) {
            const double y = at(yv, n, row, 0);
            const double w_raw = has_weights ? at(wv, n, row, 0) : 1.0;
            const bool valid_w = finite(w_raw);
            const double w = valid_w ? w_raw : 0.0;
            const bool valid_y = finite(y);
            for (int a = 0; a < k; ++a) {
                const double xa = xmat(row, a);
                const bool valid_xa = finite(xa);
                if (valid_xa && valid_y && valid_w) {
                    xy_new[a] += xa * w * y;
                    xy_valid[a] = 1;
                }
                for (int b = 0; b < k; ++b) {
                    const double xb = xmat(row, b);
                    if (valid_xa && finite(xb) && valid_w) {
                        const size_t idx = static_cast<size_t>(a * k + b);
                        xx_new(a, b) += xa * w * xb;
                        xx_valid[idx] = 1;
                    }
                }
            }
        }

        const double hl = at(hlv, n, 0, 0);
        const bool instant = !finite(hl) || hl <= 0.0;
        for (int a = 0; a < k; ++a) {
            if (instant) {
                s.xy[a] = xy_valid[a] ? xy_new[a] : 0.0;
                s.has_xy[a] = xy_valid[a];
                if (xy_valid[a]) s.last_xy[a] = s.t;
            } else if (xy_valid[a]) {
                update_ew_stat(s.xy[a], s.has_xy[a], s.last_xy[a], xy_new[a], s.t, hl);
            }
            for (int b = 0; b < k; ++b) {
                const size_t idx = static_cast<size_t>(a * k + b);
                if (instant) {
                    s.xx(a, b) = xx_valid[idx] ? xx_new(a, b) : 0.0;
                    s.has_xx[idx] = xx_valid[idx];
                    if (xx_valid[idx]) s.last_xx[idx] = s.t;
                } else if (xx_valid[idx]) {
                    update_ew_stat(s.xx(a, b), s.has_xx[idx], s.last_xx[idx], xx_new(a, b), s.t, hl);
                }
            }
        }
        for (int a = 0; a < k; ++a) {
            for (int b = a + 1; b < k; ++b) {
                const double avg = 0.5 * (s.xx(a, b) + s.xx(b, a));
                s.xx(a, b) = avg;
                s.xx(b, a) = avg;
            }
        }
        const double lam_raw = at(lamv, n, 0, 0);
        const double lam = std::max(finite(lam_raw) ? lam_raw : 0.0, 0.0);
        Vec fallback = instant ? Vec::Zero(k) : s.beta;
        s.beta = (spec.int_param != 0) ? solve_nonnegative_ridge(s.xx, s.xy, lam, fallback) : solve_ridge(s.xx, s.xy, lam, fallback);
        if (instant) {
            for (int row = 0; row < n; ++row) {
                double pred = 0.0;
                for (int a = 0; a < k; ++a) {
                    const double x = xmat(row, a);
                    pred += (finite(x) ? x : 0.0) * s.beta[a];
                }
                s.preds[row] = row_valid[static_cast<size_t>(row)] ? pred : NaN;
            }
        }
        ++s.t;
        std::copy(s.preds.data(), s.preds.data() + s.preds.size(), dst_v.data.begin());
    }


    static Vec solve_nonnegative_ridge(const RowMatrix& xx, const Vec& xy, double lam, const Vec& fallback) {
        RowMatrix lhs = xx;
        lhs.diagonal().array() += lam * xx.diagonal().array();
        if (!lhs.allFinite() || !xy.allFinite()) return fallback.cwiseMax(0.0);
        Vec beta = Vec::Zero(static_cast<int>(xy.size()));
        nnqp_eigen::active_set_impl(lhs.data(), xy.data(), beta.data(), static_cast<int>(xy.size()), std::max(64, 4 * static_cast<int>(xy.size())));
        return beta.allFinite() ? beta.cwiseMax(0.0) : fallback.cwiseMax(0.0);
    }

    static void update_ew_stat(double& current, uint8_t& has, int64_t& last, double fresh, int64_t t, double hl) {
        const double rho = (!finite(hl) || hl <= 0.0) ? 0.0 : std::exp(std::log(0.5) / hl);
        const double alpha = std::min(std::max(1.0 - rho, 0.0), 1.0);
        const double a = std::pow(alpha, static_cast<double>(t - last));
        current = has ? current * (1.0 - a) + fresh * a : fresh;
        has = 1;
        last = t;
    }

    static Vec solve_ridge(const RowMatrix& xx, const Vec& xy, double lam, const Vec& fallback) {
        RowMatrix lhs = xx;
        lhs.diagonal().array() += lam * xx.diagonal().array();
        if (!lhs.allFinite() || !xy.allFinite()) return fallback;

        Eigen::ColPivHouseholderQR<RowMatrix> solver(lhs);
        Vec beta = solver.solve(xy);
        if (beta.allFinite()) return beta;

        Eigen::JacobiSVD<RowMatrix> svd(lhs, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const auto& singular = svd.singularValues();
        if (singular.size() == 0) return fallback;
        const double scale = std::max(1.0, singular.array().abs().maxCoeff());
        const double tol = std::numeric_limits<double>::epsilon() * static_cast<double>(std::max(lhs.rows(), lhs.cols())) * scale;
        Vec inv = singular.unaryExpr([tol](double value) { return std::abs(value) > tol ? 1.0 / value : 0.0; });
        beta = svd.matrixV() * inv.asDiagonal() * svd.matrixU().transpose() * xy;
        return beta.allFinite() ? beta : fallback;
    }


};

void configure_group_universe(GroupState& group, const NodeSpec& spec, int n_instruments, int groups, int capacity) {
    group.n_groups = groups;
    group.capacity = capacity;
    if (spec.feature_widths.empty()) {
        group.group_offsets = {0, n_instruments};
        group.group_indices.resize(static_cast<size_t>(n_instruments));
        std::iota(group.group_indices.begin(), group.group_indices.end(), 0);
        return;
    }

    group.group_offsets.clear();
    group.group_indices.clear();
    group.group_offsets.push_back(0);
    size_t pos = 1;
    for (int group_i = 0; group_i < groups; ++group_i) {
        if (pos >= spec.feature_widths.size()) throw std::invalid_argument("invalid C++ jax_flat group universe encoding");
        const int len = spec.feature_widths[pos++];
        for (int j = 0; j < len; ++j) {
            if (pos >= spec.feature_widths.size()) throw std::invalid_argument("invalid C++ jax_flat group universe index encoding");
            const int col = spec.feature_widths[pos++];
            if (col < 0 || col >= n_instruments) throw std::invalid_argument("C++ jax_flat group universe column out of range");
            group.group_indices.push_back(col);
        }
        group.group_offsets.push_back(static_cast<int>(group.group_indices.size()));
    }
}

State::State(const Runtime* runtime, int n_instruments) : n_instruments_(n_instruments) {
    if (n_instruments <= 0) throw std::invalid_argument("n_instruments must be positive");
    values_.resize(runtime->nodes_.size());
    for (size_t i = 0; i < runtime->nodes_.size(); ++i) {
        const NodeSpec& spec = runtime->nodes_[i];
        NodeValue& value = values_[i];
        value.width = spec.width == 0 ? n_instruments : std::max(spec.width, 1);
        value.fixed_rows = (spec.opcode == OpCode::Einsum && spec.str_param.ends_with("->jk") && !spec.feature_widths.empty()) ? spec.feature_widths[0] : 1;
        const bool instrument_beta = spec.opcode == OpCode::GetBeta
            && !spec.children.empty()
            && runtime->nodes_[static_cast<size_t>(spec.children[0])].opcode == OpCode::InstrumentBasisMean;
        bool fixed_einsum_rows = spec.opcode == OpCode::Einsum && (spec.str_param.ends_with("->") || spec.str_param.ends_with("->jk"));
        if (spec.opcode == OpCode::Einsum) {
            const size_t arrow = spec.str_param.find("->");
            const std::string out_text = arrow == std::string::npos ? std::string() : spec.str_param.substr(arrow + 2);
            const auto out_labels = parse_einsum_term(out_text);
            if (out_labels.size() == 1) {
                bool instrument_rows = false;
                const auto in_terms = split_einsum_csv(spec.str_param.substr(0, arrow));
                for (size_t child_i = 0; child_i < in_terms.size() && child_i < spec.children.size(); ++child_i) {
                    const auto labels = parse_einsum_term(in_terms[child_i]);
                    if (!labels.empty() && labels[0] == out_labels[0]) {
                        const NodeValue& child_value = values_[static_cast<size_t>(spec.children[child_i])];
                        if (!(child_value.rows_kind == 1 && child_value.width > 1 && labels.size() == 1)) instrument_rows = true;
                    }
                }
                fixed_einsum_rows = !instrument_rows;
            }
        }
        value.rows_kind = ((spec.opcode == OpCode::GetBeta && !instrument_beta) || spec.opcode == OpCode::Mean || fixed_einsum_rows) ? 1 : 0;
        value.data.assign(static_cast<size_t>(value.size(n_instruments)), NaN);
    }
    output_.assign(static_cast<size_t>(values_[static_cast<size_t>(runtime->output_id_)].size(n_instruments)), NaN);
    row_ptrs_.assign(count_inputs(runtime->nodes_), nullptr);
    for (const NodeSpec& spec : runtime->nodes_) {
        if (spec.state_index < 0) continue;
        const int node_size = values_[static_cast<size_t>(&spec - runtime->nodes_.data())].size(n_instruments);

        switch (spec.opcode) {
            case OpCode::RollMean:
                rolling_mean_states_.emplace_back(spec.int_param, node_size);
                break;
            case OpCode::Shift:
                shift_states_.emplace_back(spec.int_param + 1, node_size);
                break;
            case OpCode::Ridge: {
                const int feature_count = std::accumulate(spec.feature_widths.begin(), spec.feature_widths.end(), 0);
                ridge_states_.emplace_back(feature_count, n_instruments);
                break;
            }
            case OpCode::InstrumentBasisMean: {
                const int feature_width = spec.feature_widths.empty() ? spec.width : spec.feature_widths.at(0);
                instrument_basis_mean_states_.emplace_back(feature_width, n_instruments);
                break;
            }
            case OpCode::Group: {
                const int groups = spec.feature_widths.empty() ? 1 : spec.feature_widths.at(0);
                const int capacity = static_cast<int>(spec.param);
                const int total_slots = capacity * groups;
                const int inner_states = count_inner_states(spec.inner_nodes);
                group_states_.emplace_back(spec.int_param, total_slots, inner_states, n_instruments);
                configure_group_universe(group_states_.back(), spec, n_instruments, groups, capacity);
                break;
            }
            default:
                value_states_.emplace_back(node_size);
                break;
        }
    }
}

size_t count_inputs(const std::vector<NodeSpec>& nodes) {
    int max_input = -1;
    for (const auto& node : nodes) if (node.opcode == OpCode::Input) max_input = std::max(max_input, node.input_index);
    return static_cast<size_t>(max_input + 1);
}

NodeSpec parse_node(py::handle item) {
    py::tuple t = py::cast<py::tuple>(item);
    if (t.size() != 9 && t.size() != 11 && t.size() != 12) throw std::invalid_argument("C++ jax_flat node specs must have nine, eleven, or twelve fields");
    NodeSpec spec;
    spec.opcode = parse_opcode(py::cast<std::string>(t[0]));
    spec.children = py::cast<std::vector<int>>(t[1]);
    spec.input_index = py::cast<int>(t[2]);
    spec.state_index = py::cast<int>(t[3]);
    spec.literal = py::cast<double>(t[4]);
    spec.param = py::cast<double>(t[5]);
    spec.int_param = py::cast<int>(t[6]);
    spec.width = py::cast<int>(t[7]);
    spec.feature_widths = py::cast<std::vector<int>>(t[8]);
    if (t.size() >= 11) {
        py::iterable inner = py::cast<py::iterable>(t[9]);
        for (py::handle child : inner) spec.inner_nodes.push_back(parse_node(child));
        spec.inner_output_id = py::cast<int>(t[10]);
    }
    if (t.size() >= 12) {
        spec.str_param = py::cast<std::string>(t[11]);
    }
    return spec;
}

Runtime make_runtime(py::iterable specs, int output_id, int n_states) {
    std::vector<NodeSpec> nodes;
    for (py::handle item : specs) nodes.push_back(parse_node(item));
    return Runtime(std::move(nodes), output_id, n_states);
}
}  // namespace
