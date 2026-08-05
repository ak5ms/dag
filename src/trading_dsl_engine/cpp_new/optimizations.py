"""Optimization pass order, exposed for diagnostics and downstream extensions."""
PASS_ORDER = ("shape_type_inference", "constant_folding", "stateless_cse", "deterministic_stateful_cse", "dead_code_elimination", "alias_projection_elimination", "kernel_fusion", "parameter_lane_lifting", "state_scratch_layout", "scratch_lifetime_coloring", "parallel_schedule", "cpp_emission")
