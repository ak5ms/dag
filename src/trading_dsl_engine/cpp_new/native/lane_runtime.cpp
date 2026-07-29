#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace py = pybind11;

namespace {
struct State {
    std::size_t instruments;
    std::size_t lanes;
    std::vector<double> value;
    std::vector<double> weight;
    std::vector<std::int64_t> count;
    std::vector<std::uint8_t> initialized;

    State(std::size_t n, std::size_t l)
        : instruments(n), lanes(l), value(n * l), weight(n * l), count(n * l), initialized(n * l) {}
};

class EwmLaneRuntime {
public:
    explicit EwmLaneRuntime(std::vector<double> spans) : spans_(std::move(spans)) {
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

    void row(State& state, const double* input, double* output) const noexcept {
        const std::size_t n = state.instruments;
        const std::size_t lanes = state.lanes;
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const double alpha = alpha_[lane];
            const double decay = decay_[lane];
            const std::size_t base = lane * n;
            for (std::size_t instrument = 0; instrument < n; ++instrument) {
                const std::size_t state_index = base + instrument;
                const std::size_t output_index = instrument * lanes + lane;
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
};
}  // namespace

PYBIND11_MODULE(_cpp_new_lanes, module) {
    py::class_<State, std::shared_ptr<State>>(module, "State");
    py::class_<EwmLaneRuntime>(module, "EwmLaneRuntime")
        .def(py::init<std::vector<double>>())
        .def("init_state", &EwmLaneRuntime::init_state)
        .def("tick_into", &EwmLaneRuntime::tick_into)
        .def("run_batch", &EwmLaneRuntime::run_batch)
        .def("run_batch_into", &EwmLaneRuntime::run_batch_into);
}
