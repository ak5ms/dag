#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
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
    And,
    Or,
    Xor,
    FillNa,
    Where,
    Cat,
    Bspline,
    Col,
    Cumsum,
    Ewm,
    FFill,
    Shift,
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
    if (name == "and") return OpCode::And;
    if (name == "or") return OpCode::Or;
    if (name == "xor") return OpCode::Xor;
    if (name == "fillna") return OpCode::FillNa;
    if (name == "where") return OpCode::Where;
    if (name == "cat") return OpCode::Cat;
    if (name == "bspline") return OpCode::Bspline;
    if (name == "col") return OpCode::Col;
    if (name == "cumsum") return OpCode::Cumsum;
    if (name == "ewm") return OpCode::Ewm;
    if (name == "ffill") return OpCode::FFill;
    if (name == "shift") return OpCode::Shift;
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
    ByteVec initialized;
    I64Vec streak;
    ByteVec seen;

    explicit ValueState(int size = 0)
        : value(Vec::Zero(size)),
          initialized(ByteVec::Zero(size)),
          streak(I64Vec::Zero(size)),
          seen(ByteVec::Zero(size)) {}
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
        inner_initialized.reserve(static_cast<size_t>(inner_state_count));
        inner_streak.reserve(static_cast<size_t>(inner_state_count));
        inner_seen.reserve(static_cast<size_t>(inner_state_count));
        for (int i = 0; i < inner_state_count; ++i) {
            inner_values.emplace_back(Vec::Zero(slot_count * instruments));
            inner_initialized.emplace_back(ByteVec::Zero(slot_count * instruments));
            inner_streak.emplace_back(I64Vec::Zero(slot_count * instruments));
            inner_seen.emplace_back(ByteVec::Zero(slot_count * instruments));
        }
    }
};


inline bool finite(double x) { return std::isfinite(x); }
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
    std::vector<RidgeState> ridge_states_;
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
    static RidgeState& ridge_state(State& state, const NodeSpec& spec) { return state.ridge_states_.at(static_cast<size_t>(spec.state_index)); }
    static GroupState& group_state(State& state, const NodeSpec& spec) { return state.group_states_.at(static_cast<size_t>(spec.state_index)); }

    void assign_native_state_indices() {
        std::array<int, 4> next{0, 0, 0, 0};
        for (NodeSpec& spec : nodes_) {
            if (spec.state_index < 0) continue;
            spec.state_index = next[static_cast<size_t>(state_bucket(spec.opcode))]++;
        }
    }

    static int state_bucket(OpCode opcode) {
        switch (opcode) {
            case OpCode::Shift: return 1;
            case OpCode::Ridge: return 2;
            case OpCode::Group: return 3;
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
                case OpCode::And:
                case OpCode::Or:
                case OpCode::Xor:
                case OpCode::FillNa: {
                    const auto& l = child(state, spec, 0);
                    const auto& r = child(state, spec, 1);
                    const int width = dst_v.width;
                    for (int i = 0; i < n; ++i) {
                        for (int c = 0; c < width; ++c) {
                            const double a = at(l, n, i, c);
                            const double b = at(r, n, i, c);
                            double out = NaN;
                            if (spec.opcode == OpCode::Add) out = a + b;
                            else if (spec.opcode == OpCode::Sub) out = a - b;
                            else if (spec.opcode == OpCode::Mul) out = a * b;
                            else if (spec.opcode == OpCode::Div) out = b == 0.0 ? NaN : a / b;
                            else if (spec.opcode == OpCode::Mod) out = b == 0.0 ? NaN : a - std::floor(a / b) * b;
                            else if (spec.opcode == OpCode::Pow) out = std::pow(a, b);
                            else if (spec.opcode == OpCode::FloorDiv) out = b == 0.0 ? NaN : std::floor(a / b);
                            else if (spec.opcode == OpCode::FillNa) out = std::isnan(a) ? b : a;
                            else if (std::isnan(a) || std::isnan(b)) out = NaN;
                            else if (spec.opcode == OpCode::Eq) out = a == b ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Ne) out = a != b ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Lt) out = a < b ? 1.0 : 0.0;
                            else if (spec.opcode == OpCode::Gt) out = a > b ? 1.0 : 0.0;
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
                case OpCode::Fraction: {
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
                        else dst[i] = v - std::floor(v);
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
                    const std::string& sub = spec.str_param;
                    if (sub == "i,i->i" || sub == "i,i,i->i") {
                        Eigen::Map<Vec> out(dst, n);
                        out.setOnes();
                        for (int child_id : spec.children) {
                            const auto& child_v = state.values_[static_cast<size_t>(child_id)];
                            Eigen::Map<const Vec> child_vec(child_v.data.data(), n);
                            out.array() *= child_vec.array();
                        }
                    } else if (sub == "i,ij->i") {
                        const auto& x = child(state, spec, 0);
                        const auto& m = child(state, spec, 1);
                        Eigen::Map<const Vec> x_vec(x.data.data(), n);
                        Eigen::Map<const RowMatrix> matrix(m.data.data(), n, m.width);
                        Eigen::Map<Vec> out(dst, n);
                        out.array() = x_vec.array() * matrix.rowwise().sum().array();
                    } else if (sub == "ij,ij->ij") {
                        const auto& a = child(state, spec, 0);
                        const auto& b = child(state, spec, 1);
                        Eigen::Map<const RowMatrix> left(a.data.data(), n, a.width);
                        Eigen::Map<const RowMatrix> right(b.data.data(), n, b.width);
                        Eigen::Map<RowMatrix> out(dst, n, dst_v.width);
                        out.array() = left.array() * right.array();
                    } else if (sub == "ij,ik->jk") {
                        const auto& a = child(state, spec, 0);
                        const auto& b = child(state, spec, 1);
                        Eigen::Map<const RowMatrix> left(a.data.data(), n, a.width);
                        Eigen::Map<const RowMatrix> right(b.data.data(), n, b.width);
                        Eigen::Map<RowMatrix> out(dst, a.width, b.width);
                        out.noalias() = left.transpose() * right;
                    } else if (sub == "ij,ij->") {
                        const auto& a = child(state, spec, 0);
                        const auto& b = child(state, spec, 1);
                        Eigen::Map<const RowMatrix> left(a.data.data(), n, a.width);
                        Eigen::Map<const RowMatrix> right(b.data.data(), n, b.width);
                        dst[0] = (left.array() * right.array()).sum();
                    } else {
                        throw std::invalid_argument("unsupported C++ jax_flat einsum pattern: " + sub);
                    }
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
                    const double denom = std::max<size_t>(compact.size(), 1);
                    for (int i = 0; i < n; ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        dst[i] = finite(v) ? static_cast<double>(std::upper_bound(compact.begin(), compact.end(), v) - compact.begin()) / denom : NaN;
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
                    for (int i = 0; i < dst_v.size(n); ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        if (finite(v)) { s.value[i] = s.initialized[i] ? alpha * v + (1.0 - alpha) * s.value[i] : v; s.initialized[i] = 1; }
                        dst[i] = s.initialized[i] ? s.value[i] : NaN;
                    }
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
                case OpCode::Ridge:
                    eval_ridge(state, spec, dst_v);
                    break;
                case OpCode::GetBeta: {
                    const auto& s = state.ridge_states_[static_cast<size_t>(nodes_[static_cast<size_t>(spec.children[0])].state_index)];
                    std::copy(s.beta.data(), s.beta.data() + s.beta.size(), dst);
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
                auto& init = s.inner_initialized.at(static_cast<size_t>(node.state_index));
                const double alpha = 2.0 / (node.param + 1.0);
                if (finite(v)) {
                    state_v[off] = init[off] ? alpha * v + (1.0 - alpha) * state_v[off] : v;
                    init[off] = 1;
                }
                return init[off] ? state_v[off] : NaN;
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

        // Emit predictions from the previous beta before the sufficient statistics update.
        for (int row = 0; row < n; ++row) {
            bool row_valid = finite(at(yv, n, row, 0));
            double pred = 0.0;
            for (int a = 0; a < k; ++a) {
                const double x = xmat(row, a);
                row_valid &= finite(x);
                pred += (finite(x) ? x : 0.0) * s.beta[a];
            }
            s.preds[row] = row_valid ? pred : NaN;
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
        for (int a = 0; a < k; ++a) {
            if (xy_valid[a]) update_ew_stat(s.xy[a], s.has_xy[a], s.last_xy[a], xy_new[a], s.t, hl);
            for (int b = 0; b < k; ++b) {
                const size_t idx = static_cast<size_t>(a * k + b);
                if (xx_valid[idx]) update_ew_stat(s.xx(a, b), s.has_xx[idx], s.last_xx[idx], xx_new(a, b), s.t, hl);
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
        solve_ridge(s, lam);
        ++s.t;
        std::copy(s.preds.data(), s.preds.data() + s.preds.size(), dst_v.data.begin());
    }

    static void update_ew_stat(double& current, uint8_t& has, int64_t& last, double fresh, int64_t t, double hl) {
        const double rho = (!finite(hl) || hl <= 0.0) ? 0.0 : std::exp(std::log(0.5) / hl);
        const double alpha = std::min(std::max(1.0 - rho, 0.0), 1.0);
        const double a = std::pow(alpha, static_cast<double>(t - last));
        current = has ? current * (1.0 - a) + fresh * a : fresh;
        has = 1;
        last = t;
    }

    static void solve_ridge(RidgeState& s, double lam) {
        const int k = s.k;
        RowMatrix lhs = s.xx;
        lhs.diagonal().array() += lam * s.xx.diagonal().array();
        if (!lhs.allFinite() || !s.xy.allFinite()) return;

        Eigen::ColPivHouseholderQR<RowMatrix> solver(lhs);
        solver.setThreshold(1e-12);
        if (solver.rank() < k) return;

        Vec beta = solver.solve(s.xy);
        if (beta.allFinite()) s.beta = std::move(beta);
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
        value.rows_kind = (spec.opcode == OpCode::GetBeta || spec.opcode == OpCode::Mean || (spec.opcode == OpCode::Einsum && (spec.str_param.ends_with("->") || spec.str_param.ends_with("->jk")))) ? 1 : 0;
        value.data.assign(static_cast<size_t>(value.size(n_instruments)), NaN);
    }
    output_.assign(static_cast<size_t>(values_[static_cast<size_t>(runtime->output_id_)].size(n_instruments)), NaN);
    row_ptrs_.assign(count_inputs(runtime->nodes_), nullptr);
    for (const NodeSpec& spec : runtime->nodes_) {
        if (spec.state_index < 0) continue;
        const int node_size = values_[static_cast<size_t>(&spec - runtime->nodes_.data())].size(n_instruments);

        switch (spec.opcode) {
            case OpCode::Shift:
                shift_states_.emplace_back(spec.int_param + 1, node_size);
                break;
            case OpCode::Ridge: {
                const int feature_count = std::accumulate(spec.feature_widths.begin(), spec.feature_widths.end(), 0);
                ridge_states_.emplace_back(feature_count, n_instruments);
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

