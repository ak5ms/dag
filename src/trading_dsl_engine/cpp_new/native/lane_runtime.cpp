#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unsupported/Eigen/SpecialFunctions>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace py = pybind11;

namespace {
struct RankItem { double value; std::size_t instrument; };
struct State {
    std::size_t instruments;
    std::size_t lanes;
    std::vector<double> value;
    std::vector<double> weight;
    std::vector<std::int64_t> count;
    std::vector<std::uint8_t> initialized;
    std::vector<double> row_scratch;
    std::vector<RankItem> rank_items;
    std::vector<double> full_rank_scores;

    State(std::size_t n, std::size_t l)
        : instruments(n), lanes(l), value(n * l), weight(n * l), count(n * l), initialized(n * l),
          row_scratch(n * l), rank_items(n), full_rank_scores(n) {
        for (std::size_t rank = 1; rank <= n; ++rank)
            full_rank_scores[rank - 1] = Eigen::numext::ndtri(static_cast<double>(rank) / (n + 1.0));
    }
};

class EwmLaneRuntime {
public:
    explicit EwmLaneRuntime(std::vector<double> spans, bool rank_output = false)
        : spans_(std::move(spans)), rank_output_(rank_output) {
        alpha_.reserve(spans_.size());
        decay_.reserve(spans_.size());
        for (double span : spans_) {
            const double alpha = 2.0 / (span + 1.0);
            alpha_.push_back(alpha);
            decay_.push_back(1.0 - alpha);
        }
    }

    std::shared_ptr<State> init_state(std::size_t instruments) const {
        return std::make_shared<State>(instruments, spans_.size());
    }
    bool rank_output() const noexcept { return rank_output_; }

    void row(State& state, const double* input, double* output) const noexcept {
        if (!rank_output_) {
            row_instrument_major(state, input, output);
            return;
        }
        row_instrument_major(state, input, state.row_scratch.data());
        for (std::size_t lane = 0; lane < state.lanes; ++lane)
            rank_lane(state, lane, output);
    }

    static void rank_lane(State& state, std::size_t lane, double* output) noexcept {
        const std::size_t n = state.instruments;
        std::size_t count = 0;
        for (std::size_t instrument = 0; instrument < n; ++instrument) {
            const double value = state.row_scratch[instrument * state.lanes + lane];
            if (std::isfinite(value)) state.rank_items[count++] = {value, instrument};
            else output[instrument * state.lanes + lane] = NAN;
        }
        std::sort(state.rank_items.begin(), state.rank_items.begin() + count,
                  [](const RankItem& left, const RankItem& right) {
                      return left.value < right.value || (left.value == right.value && left.instrument < right.instrument);
                  });
        for (std::size_t begin = 0; begin < count;) {
            std::size_t upper = begin + 1;
            while (upper < count && state.rank_items[upper].value == state.rank_items[begin].value) ++upper;
            const double score = count == n ? state.full_rank_scores[upper - 1]
                : Eigen::numext::ndtri(static_cast<double>(upper) / (count + 1.0));
            for (std::size_t position = begin; position < upper; ++position)
                output[state.rank_items[position].instrument * state.lanes + lane] = score;
            begin = upper;
        }
    }

    void row_lane_major(State& state, const double* input, double* output, bool lane_major_output) const noexcept {
        const std::size_t n = state.instruments;
        const std::size_t lanes = state.lanes;
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const double alpha = alpha_[lane];
            const double decay = decay_[lane];
            const std::size_t base = lane * n;
            for (std::size_t instrument = 0; instrument < n; ++instrument) {
                const std::size_t state_index = base + instrument;
                const std::size_t output_index = lane_major_output ? state_index : instrument * lanes + lane;
                const double observation = input[instrument];
                if (!std::isfinite(observation)) {
                    output[output_index] = state.initialized[state_index] ? state.value[state_index] : NAN;
                    continue;
                }
                double old_weight = state.weight[state_index];
                if (state.initialized[state_index]) {
                    old_weight *= decay;
                    double new_weight = alpha;
                    if (std::abs(alpha - 0.5) <= 1e-12) new_weight = 1.0 - old_weight;
                    if (state.value[state_index] != observation) {
                        state.value[state_index] =
                            (old_weight * state.value[state_index] + new_weight * observation) /
                            (old_weight + new_weight);
                    }
                } else {
                    state.value[state_index] = observation;
                    state.initialized[state_index] = 1;
                }
                state.weight[state_index] = 1.0;
                ++state.count[state_index];
                output[output_index] = state.value[state_index];
            }
        }
    }

    void row_instrument_major(State& state, const double* input, double* output) const noexcept {
        const std::size_t n = state.instruments;
        const std::size_t lanes = state.lanes;
        for (std::size_t instrument = 0; instrument < n; ++instrument) {
            const double observation = input[instrument];
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                const std::size_t index = lane * n + instrument;
                const std::size_t output_index = instrument * lanes + lane;
                if (!std::isfinite(observation)) {
                    output[output_index] = state.initialized[index] ? state.value[index] : NAN;
                    continue;
                }
                double old_weight = state.weight[index];
                if (state.initialized[index]) {
                    old_weight *= decay_[lane];
                    double new_weight = alpha_[lane];
                    if (std::abs(alpha_[lane] - 0.5) <= 1e-12) new_weight = 1.0 - old_weight;
                    if (state.value[index] != observation)
                        state.value[index] = (old_weight * state.value[index] + new_weight * observation) /
                                             (old_weight + new_weight);
                } else {
                    state.value[index] = observation;
                    state.initialized[index] = 1;
                }
                state.weight[index] = 1.0;
                ++state.count[index];
                output[output_index] = state.value[index];
            }
        }
    }

    void row_materialized(State& state, const double* input, double* output) const noexcept {
        row_lane_major(state, input, state.row_scratch.data(), true);
        for (std::size_t instrument = 0; instrument < state.instruments; ++instrument)
            for (std::size_t lane = 0; lane < state.lanes; ++lane)
                output[instrument * state.lanes + lane] = state.row_scratch[lane * state.instruments + instrument];
    }

    void row_store_only(const State& state, const double* input, double* output) const noexcept {
        for (std::size_t instrument = 0; instrument < state.instruments; ++instrument)
            for (std::size_t lane = 0; lane < state.lanes; ++lane)
                output[instrument * state.lanes + lane] = input[instrument];
    }

    void tick_into(const std::shared_ptr<State>& state, py::array_t<double, py::array::c_style> output,
                   py::array_t<double, py::array::c_style | py::array::forcecast> input) const {
        validate_row(*state, input, output);
        row(*state, input.data(), output.mutable_data());
    }

    py::array_t<double> run_batch(const std::shared_ptr<State>& state,
                                  py::array_t<double, py::array::c_style | py::array::forcecast> input) const {
        if (input.ndim() != 2 || static_cast<std::size_t>(input.shape(1)) != state->instruments)
            throw std::invalid_argument("input must have shape (rows, n_instruments)");
        py::array_t<double> output({input.shape(0), input.shape(1), static_cast<py::ssize_t>(spans_.size())});
        const std::size_t input_stride = state->instruments;
        const std::size_t output_stride = state->instruments * spans_.size();
        const double* input_data = input.data();
        double* output_data = output.mutable_data();
        py::gil_scoped_release release;
        for (py::ssize_t timestep = 0; timestep < input.shape(0); ++timestep)
            row(*state, input_data + timestep * input_stride, output_data + timestep * output_stride);
        return output;
    }

    void run_batch_into(const std::shared_ptr<State>& state,
                        py::array_t<double, py::array::c_style> output,
                        py::array_t<double, py::array::c_style | py::array::forcecast> input) const {
        if (input.ndim() != 2 || static_cast<std::size_t>(input.shape(1)) != state->instruments)
            throw std::invalid_argument("input must have shape (rows, n_instruments)");
        if (output.ndim() != 3 || output.shape(0) != input.shape(0) || output.shape(1) != input.shape(1) ||
            static_cast<std::size_t>(output.shape(2)) != state->lanes)
            throw std::invalid_argument("output must have shape (rows, n_instruments, lanes)");
        const std::size_t input_stride = state->instruments;
        const std::size_t output_stride = state->instruments * spans_.size();
        const double* input_data = input.data();
        double* output_data = output.mutable_data();
        py::gil_scoped_release release;
        for (py::ssize_t timestep = 0; timestep < input.shape(0); ++timestep)
            row(*state, input_data + timestep * input_stride, output_data + timestep * output_stride);
    }

    void run_batch_ablation(const std::shared_ptr<State>& state,
                            py::array_t<double, py::array::c_style> output,
                            py::array_t<double, py::array::c_style | py::array::forcecast> input,
                            const std::string& variant) const {
        if (input.ndim() != 2 || output.ndim() != 3 || output.shape(0) != input.shape(0) ||
            output.shape(1) != input.shape(1) || static_cast<std::size_t>(output.shape(2)) != state->lanes)
            throw std::invalid_argument("ablation arrays have incompatible shapes");
        enum class Variant { LaneMajor, InstrumentMajor, Materialized, StoreOnly };
        const Variant selected = variant == "lane-major" ? Variant::LaneMajor :
            variant == "instrument-major" ? Variant::InstrumentMajor :
            variant == "materialized" ? Variant::Materialized :
            variant == "store-only" ? Variant::StoreOnly : throw std::invalid_argument("unknown ablation variant");
        const std::size_t input_stride = state->instruments;
        const std::size_t output_stride = state->instruments * state->lanes;
        const double* input_data = input.data();
        double* output_data = output.mutable_data();
        py::gil_scoped_release release;
        for (py::ssize_t timestep = 0; timestep < input.shape(0); ++timestep) {
            const double* in = input_data + timestep * input_stride;
            double* out = output_data + timestep * output_stride;
            switch (selected) {
                case Variant::LaneMajor: row_lane_major(*state, in, out, false); break;
                case Variant::InstrumentMajor: row_instrument_major(*state, in, out); break;
                case Variant::Materialized: row_materialized(*state, in, out); break;
                case Variant::StoreOnly: row_store_only(*state, in, out); break;
            }
        }
    }

private:
    void validate_row(const State& state, const py::array& input, const py::array& output) const {
        if (input.ndim() != 1 || static_cast<std::size_t>(input.shape(0)) != state.instruments)
            throw std::invalid_argument("input row has wrong shape");
        if (output.ndim() != 2 || static_cast<std::size_t>(output.shape(0)) != state.instruments ||
            static_cast<std::size_t>(output.shape(1)) != state.lanes)
            throw std::invalid_argument("output row has wrong shape");
    }

    std::vector<double> spans_;
    std::vector<double> alpha_;
    std::vector<double> decay_;
    bool rank_output_;
};
}  // namespace

PYBIND11_MODULE(_cpp_new_lanes, module) {
    py::class_<State, std::shared_ptr<State>>(module, "State");
    py::class_<EwmLaneRuntime>(module, "EwmLaneRuntime")
        .def(py::init<std::vector<double>, bool>(), py::arg("spans"), py::arg("rank_output") = false)
        .def("init_state", &EwmLaneRuntime::init_state)
        .def_property_readonly("rank_output", &EwmLaneRuntime::rank_output)
        .def("tick_into", &EwmLaneRuntime::tick_into)
        .def("run_batch", &EwmLaneRuntime::run_batch)
        .def("run_batch_into", &EwmLaneRuntime::run_batch_into)
        .def("run_batch_ablation", &EwmLaneRuntime::run_batch_ablation);
}
