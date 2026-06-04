#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
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
    GroupCumsum,
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
    if (name == "group_cumsum") return OpCode::GroupCumsum;
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
};

struct NodeValue {
    int rows_kind = 0;  // 0 => instrument-aligned rows; 1 => fixed rows = width (beta/scalar-vector)
    int width = 1;
    std::vector<double> data;

    int rows(int n) const { return rows_kind == 0 ? n : width; }
    int size(int n) const { return rows(n) * (rows_kind == 0 ? width : 1); }
};

struct StateSlot {
    std::vector<double> value;
    std::vector<uint8_t> initialized;
    std::vector<int64_t> streak;
    std::vector<uint8_t> seen;
    std::vector<double> buffer;
    int pos = 0;
    int count = 0;
    int cap = 0;

    // Ridge state.
    std::vector<double> xx;
    std::vector<double> xy;
    std::vector<uint8_t> has_xx;
    std::vector<uint8_t> has_xy;
    std::vector<int64_t> last_xx;
    std::vector<int64_t> last_xy;
    std::vector<double> beta;
    std::vector<double> preds;
    int64_t t = 0;
    int k = 0;

    // Grouped cumsum state.
    int n_keys = 0;
    int capacity = 0;
    std::vector<double> keys;
    std::vector<uint8_t> occupied;
    std::vector<double> grouped_values;
};

inline bool finite(double x) { return std::isfinite(x); }
inline bool same_double_key(double a, double b) { return (std::isnan(a) && std::isnan(b)) || a == b; }

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
    std::vector<StateSlot> slots_;
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

    static StateSlot& slot(State& state, const NodeSpec& spec) { return state.slots_.at(static_cast<size_t>(spec.state_index)); }
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
                    auto& s = slot(state, spec);
                    const auto& x = child(state, spec, 0);
                    for (int i = 0; i < dst_v.size(n); ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        if (finite(v)) { s.value[static_cast<size_t>(i)] += v; s.initialized[static_cast<size_t>(i)] = 1; dst[i] = s.value[static_cast<size_t>(i)]; }
                        else dst[i] = NaN;
                    }
                    break;
                }
                case OpCode::Ewm: {
                    auto& s = slot(state, spec);
                    const auto& x = child(state, spec, 0);
                    const double alpha = 2.0 / (spec.param + 1.0);
                    for (int i = 0; i < dst_v.size(n); ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        if (finite(v)) { s.value[static_cast<size_t>(i)] = s.initialized[static_cast<size_t>(i)] ? alpha * v + (1.0 - alpha) * s.value[static_cast<size_t>(i)] : v; s.initialized[static_cast<size_t>(i)] = 1; }
                        dst[i] = s.initialized[static_cast<size_t>(i)] ? s.value[static_cast<size_t>(i)] : NaN;
                    }
                    break;
                }
                case OpCode::FFill: {
                    auto& s = slot(state, spec);
                    const auto& x = child(state, spec, 0);
                    const int limit = spec.int_param;
                    for (int i = 0; i < dst_v.size(n); ++i) {
                        const double v = x.data[static_cast<size_t>(i)];
                        if (finite(v)) { s.value[static_cast<size_t>(i)] = v; s.seen[static_cast<size_t>(i)] = 1; s.streak[static_cast<size_t>(i)] = 0; dst[i] = v; }
                        else { s.streak[static_cast<size_t>(i)] += 1; const bool allowed = s.seen[static_cast<size_t>(i)] && (limit < 0 || s.streak[static_cast<size_t>(i)] <= limit); dst[i] = allowed ? s.value[static_cast<size_t>(i)] : NaN; }
                    }
                    break;
                }
                case OpCode::Shift: {
                    auto& s = slot(state, spec);
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
                            else if (s.count >= lag_i) { int rp = (s.pos - lag_i) % s.cap; if (rp < 0) rp += s.cap; dst[static_cast<size_t>(i * width + c)] = s.buffer[static_cast<size_t>((rp * n + i) * width + c)]; }
                            else dst[static_cast<size_t>(i * width + c)] = NaN;
                        }
                    }
                    std::copy(x.data.begin(), x.data.begin() + n * width, s.buffer.begin() + static_cast<size_t>(s.pos * n * width));
                    s.pos = (s.pos + 1) % s.cap;
                    s.count = std::min(s.count + 1, s.cap);
                    break;
                }
                case OpCode::Ridge:
                    eval_ridge(state, spec, dst_v);
                    break;
                case OpCode::GetBeta: {
                    const auto& r = child(state, spec, 0);
                    const auto& s = state.slots_[static_cast<size_t>(nodes_[static_cast<size_t>(spec.children[0])].state_index)];
                    std::copy(s.beta.begin(), s.beta.end(), dst);
                    break;
                }
                case OpCode::GetPreds: {
                    const auto& r = child(state, spec, 0);
                    std::copy(r.data.begin(), r.data.end(), dst);
                    break;
                }
                case OpCode::GroupCumsum:
                    eval_group_cumsum(state, spec, dst_v);
                    break;
            }
        }
        const auto& root = state.values_[static_cast<size_t>(output_id_)];
        std::copy(root.data.begin(), root.data.begin() + root.size(n), out_ptr);
    }

    void eval_ridge(State& state, const NodeSpec& spec, NodeValue& dst_v) const {
        const int n = state.n_instruments_;
        StateSlot& s = slot(state, spec);
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

        std::vector<double> xmat(static_cast<size_t>(n * k), NaN);
        for (int row = 0; row < n; ++row) {
            int off = 0;
            for (size_t f = 0; f < spec.feature_widths.size(); ++f) {
                const auto& feat = state.values_[static_cast<size_t>(spec.children[f])];
                for (int c = 0; c < spec.feature_widths[f]; ++c) xmat[static_cast<size_t>(row * k + off++)] = at(feat, n, row, c);
            }
        }

        // Emit predictions from the previous beta before the sufficient statistics update.
        for (int row = 0; row < n; ++row) {
            bool row_valid = finite(at(yv, n, row, 0));
            double pred = 0.0;
            for (int a = 0; a < k; ++a) {
                const double x = xmat[static_cast<size_t>(row * k + a)];
                row_valid &= finite(x);
                pred += (finite(x) ? x : 0.0) * s.beta[static_cast<size_t>(a)];
            }
            s.preds[static_cast<size_t>(row)] = row_valid ? pred : NaN;
        }

        std::vector<double> xx_new(static_cast<size_t>(k * k), 0.0);
        std::vector<double> xy_new(static_cast<size_t>(k), 0.0);
        std::vector<uint8_t> xx_valid(static_cast<size_t>(k * k), 0);
        std::vector<uint8_t> xy_valid(static_cast<size_t>(k), 0);
        for (int row = 0; row < n; ++row) {
            const double y = at(yv, n, row, 0);
            const double w_raw = has_weights ? at(wv, n, row, 0) : 1.0;
            const bool valid_w = finite(w_raw);
            const double w = valid_w ? w_raw : 0.0;
            const bool valid_y = finite(y);
            for (int a = 0; a < k; ++a) {
                const double xa = xmat[static_cast<size_t>(row * k + a)];
                const bool valid_xa = finite(xa);
                if (valid_xa && valid_y && valid_w) {
                    xy_new[static_cast<size_t>(a)] += xa * w * y;
                    xy_valid[static_cast<size_t>(a)] = 1;
                }
                for (int b = 0; b < k; ++b) {
                    const double xb = xmat[static_cast<size_t>(row * k + b)];
                    if (valid_xa && finite(xb) && valid_w) {
                        const size_t idx = static_cast<size_t>(a * k + b);
                        xx_new[idx] += xa * w * xb;
                        xx_valid[idx] = 1;
                    }
                }
            }
        }

        const double hl = at(hlv, n, 0, 0);
        for (int a = 0; a < k; ++a) {
            if (xy_valid[static_cast<size_t>(a)]) update_ew_stat(s.xy[static_cast<size_t>(a)], s.has_xy[static_cast<size_t>(a)], s.last_xy[static_cast<size_t>(a)], xy_new[static_cast<size_t>(a)], s.t, hl);
            for (int b = 0; b < k; ++b) {
                const size_t idx = static_cast<size_t>(a * k + b);
                if (xx_valid[idx]) update_ew_stat(s.xx[idx], s.has_xx[idx], s.last_xx[idx], xx_new[idx], s.t, hl);
            }
        }
        for (int a = 0; a < k; ++a) {
            for (int b = a + 1; b < k; ++b) {
                const size_t ab = static_cast<size_t>(a * k + b);
                const size_t ba = static_cast<size_t>(b * k + a);
                const double avg = 0.5 * (s.xx[ab] + s.xx[ba]);
                s.xx[ab] = avg;
                s.xx[ba] = avg;
            }
        }
        const double lam_raw = at(lamv, n, 0, 0);
        const double lam = std::max(finite(lam_raw) ? lam_raw : 0.0, 0.0);
        solve_ridge(s, lam);
        ++s.t;
        std::copy(s.preds.begin(), s.preds.end(), dst_v.data.begin());
    }

    static void update_ew_stat(double& current, uint8_t& has, int64_t& last, double fresh, int64_t t, double hl) {
        const double rho = (!finite(hl) || hl <= 0.0) ? 0.0 : std::exp(std::log(0.5) / hl);
        const double alpha = std::min(std::max(1.0 - rho, 0.0), 1.0);
        const double a = std::pow(alpha, static_cast<double>(t - last));
        current = has ? current * (1.0 - a) + fresh * a : fresh;
        has = 1;
        last = t;
    }

    static void solve_ridge(StateSlot& s, double lam) {
        const int k = s.k;
        std::vector<double> a = s.xx;
        std::vector<double> b = s.xy;
        for (int i = 0; i < k; ++i) a[static_cast<size_t>(i * k + i)] += lam * s.xx[static_cast<size_t>(i * k + i)];
        std::vector<double> x(static_cast<size_t>(k), 0.0);
        for (int col = 0; col < k; ++col) {
            int pivot = col;
            double best = std::abs(a[static_cast<size_t>(col * k + col)]);
            for (int r = col + 1; r < k; ++r) if (std::abs(a[static_cast<size_t>(r * k + col)]) > best) { best = std::abs(a[static_cast<size_t>(r * k + col)]); pivot = r; }
            if (best <= 1e-18 || !finite(best)) return;
            if (pivot != col) {
                for (int c = col; c < k; ++c) std::swap(a[static_cast<size_t>(col * k + c)], a[static_cast<size_t>(pivot * k + c)]);
                std::swap(b[static_cast<size_t>(col)], b[static_cast<size_t>(pivot)]);
            }
            const double diag = a[static_cast<size_t>(col * k + col)];
            for (int c = col; c < k; ++c) a[static_cast<size_t>(col * k + c)] /= diag;
            b[static_cast<size_t>(col)] /= diag;
            for (int r = 0; r < k; ++r) {
                if (r == col) continue;
                const double factor = a[static_cast<size_t>(r * k + col)];
                for (int c = col; c < k; ++c) a[static_cast<size_t>(r * k + c)] -= factor * a[static_cast<size_t>(col * k + c)];
                b[static_cast<size_t>(r)] -= factor * b[static_cast<size_t>(col)];
            }
        }
        if (std::all_of(b.begin(), b.end(), [](double v) { return finite(v); })) s.beta = b;
    }

    void eval_group_cumsum(State& state, const NodeSpec& spec, NodeValue& dst_v) const {
        const int n = state.n_instruments_;
        StateSlot& s = slot(state, spec);
        const int n_keys = s.n_keys;
        const auto& lhs = state.values_[static_cast<size_t>(spec.children[static_cast<size_t>(n_keys)])];
        std::fill(dst_v.data.begin(), dst_v.data.end(), NaN);
        for (int row = 0; row < n; ++row) {
            int slot_i = find_or_insert_group_slot(state, spec, row);
            const double v = at(lhs, n, row, 0);
            if (finite(v)) s.grouped_values[static_cast<size_t>(slot_i * n + row)] += v;
            dst_v.data[static_cast<size_t>(row)] = finite(v) ? s.grouped_values[static_cast<size_t>(slot_i * n + row)] : NaN;
        }
    }

    int find_or_insert_group_slot(State& state, const NodeSpec& spec, int row) const {
        const int n = state.n_instruments_;
        StateSlot& s = slot(state, spec);
        for (int slot_i = 0; slot_i < s.capacity; ++slot_i) {
            if (!s.occupied[static_cast<size_t>(slot_i)]) continue;
            bool same = true;
            for (int k = 0; k < s.n_keys; ++k) same &= same_double_key(s.keys[static_cast<size_t>(slot_i * s.n_keys + k)], at(state.values_[static_cast<size_t>(spec.children[static_cast<size_t>(k)])], n, row, 0));
            if (same) return slot_i;
        }
        for (int slot_i = 0; slot_i < s.capacity; ++slot_i) {
            if (s.occupied[static_cast<size_t>(slot_i)]) continue;
            s.occupied[static_cast<size_t>(slot_i)] = 1;
            for (int k = 0; k < s.n_keys; ++k) s.keys[static_cast<size_t>(slot_i * s.n_keys + k)] = at(state.values_[static_cast<size_t>(spec.children[static_cast<size_t>(k)])], n, row, 0);
            return slot_i;
        }
        throw std::runtime_error("C++ jax_flat groupby capacity exceeded");
    }
};

State::State(const Runtime* runtime, int n_instruments) : n_instruments_(n_instruments) {
    if (n_instruments <= 0) throw std::invalid_argument("n_instruments must be positive");
    values_.resize(runtime->nodes_.size());
    for (size_t i = 0; i < runtime->nodes_.size(); ++i) {
        const NodeSpec& spec = runtime->nodes_[i];
        NodeValue& value = values_[i];
        value.width = std::max(spec.width, 1);
        value.rows_kind = (spec.opcode == OpCode::GetBeta || spec.opcode == OpCode::Mean) ? 1 : 0;
        value.data.assign(static_cast<size_t>(value.size(n_instruments)), NaN);
    }
    output_.assign(static_cast<size_t>(values_[static_cast<size_t>(runtime->output_id_)].size(n_instruments)), NaN);
    row_ptrs_.assign(count_inputs(runtime->nodes_), nullptr);
    slots_.resize(static_cast<size_t>(runtime->n_states_));
    for (const NodeSpec& spec : runtime->nodes_) {
        if (spec.state_index < 0) continue;
        StateSlot& s = slots_[static_cast<size_t>(spec.state_index)];
        const int size = values_[static_cast<size_t>(&spec - runtime->nodes_.data())].size(n_instruments);
        if (spec.opcode == OpCode::Shift) {
            s.cap = spec.int_param + 1;
            s.buffer.assign(static_cast<size_t>(s.cap * size), NaN);
        } else if (spec.opcode == OpCode::Ridge) {
            s.k = std::accumulate(spec.feature_widths.begin(), spec.feature_widths.end(), 0);
            s.value.assign(static_cast<size_t>(s.k), 0.0);
            s.xx.assign(static_cast<size_t>(s.k * s.k), 0.0);
            s.xy.assign(static_cast<size_t>(s.k), 0.0);
            s.has_xx.assign(static_cast<size_t>(s.k * s.k), 0);
            s.has_xy.assign(static_cast<size_t>(s.k), 0);
            s.last_xx.assign(static_cast<size_t>(s.k * s.k), 0);
            s.last_xy.assign(static_cast<size_t>(s.k), 0);
            s.beta.assign(static_cast<size_t>(s.k), 0.0);
            s.preds.assign(static_cast<size_t>(n_instruments), NaN);
        } else if (spec.opcode == OpCode::GroupCumsum) {
            s.n_keys = spec.int_param;
            s.capacity = static_cast<int>(spec.param);
            s.keys.assign(static_cast<size_t>(s.capacity * std::max(s.n_keys, 1)), NaN);
            s.occupied.assign(static_cast<size_t>(s.capacity), 0);
            s.grouped_values.assign(static_cast<size_t>(s.capacity * n_instruments), 0.0);
        } else {
            s.value.assign(static_cast<size_t>(size), 0.0);
            s.initialized.assign(static_cast<size_t>(size), 0);
            s.streak.assign(static_cast<size_t>(size), 0);
            s.seen.assign(static_cast<size_t>(size), 0);
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
    if (t.size() != 9) throw std::invalid_argument("C++ jax_flat node specs must have nine fields");
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
    return spec;
}

Runtime make_runtime(py::iterable specs, int output_id, int n_states) {
    std::vector<NodeSpec> nodes;
    for (py::handle item : specs) nodes.push_back(parse_node(item));
    return Runtime(std::move(nodes), output_id, n_states);
}
}  // namespace

