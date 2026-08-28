#include "ops.cpp"

PYBIND11_MODULE(_cpp_flat, m) {
    py::class_<State>(m, "CppFlatState").def_property_readonly("n_instruments", &State::n_instruments);
    py::class_<Runtime>(m, "CppFlatRuntimeCore")
        .def("init_state", &Runtime::init_state, py::arg("n_instruments"))
        .def("tick", &Runtime::tick, py::arg("state"))
        .def("tick_into", &Runtime::tick_into, py::arg("state"), py::arg("out"))
        .def("run_batch", &Runtime::run_batch, py::arg("state"))
        .def("run_batch_into", &Runtime::run_batch_into, py::arg("state"), py::arg("out"));
    m.def("make_runtime", &make_runtime, py::arg("node_specs"), py::arg("output_id"), py::arg("n_states"), py::arg("workers"));
}
