"""Compile-time-only cpp_new operator descriptors."""
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class OperatorDescriptor:
    name: str
    arity: tuple[int, int]
    state_family: str | None
    scratch_family: str | None
    pure: bool
    deterministic_state: bool
    fusion_barrier: bool
    direct_root: bool
    parallel: str
    reference: str
    lane_lift: bool = False
    lane_parameters: tuple[str, ...] = ()
    lane_root: bool = False


_DESCRIPTORS = {
    "input": OperatorDescriptor("input", (0, 0), None, None, True, False, False, True, "instruments", "jax_flat.InputOp"),
    "literal": OperatorDescriptor("literal", (0, 0), None, None, True, False, False, True, "serial", "jax_flat.LiteralOp"),
    "ewm": OperatorDescriptor("ewm", (1, 3), "EwmState", None, False, True, False, True, "instruments", "jax_flat.EwmOp", True, ("param",)),
    "xs_rank": OperatorDescriptor("xs_rank", (1, 1), None, "XsRankScratch", True, False, True, True, "rank_lanes", "jax_flat xs_rank", True),
    "ridge": OperatorDescriptor("ridge", (4, 64), "RidgeState", "RidgeScratch", False, True, True, False, "ridge_reduction", "jax_flat.RidgeOp", True, ("int_param",)),
    "get_beta": OperatorDescriptor("get_beta", (1, 1), None, None, True, False, False, True, "serial", "RidgeValue.beta"),
    "cat": OperatorDescriptor("cat", (1, 64), None, None, True, False, True, True, "feature_lanes", "jax_flat cat", lane_root=True),
}
for _name in ("add", "sub", "mul", "div"):
    _DESCRIPTORS[_name] = OperatorDescriptor(_name, (2, 2), None, None, True, False, False, True, "instruments", f"jax.numpy.{_name}", True)
DESCRIPTORS = MappingProxyType(_DESCRIPTORS)


def descriptor(name: str) -> OperatorDescriptor:
    try:
        return DESCRIPTORS[name]
    except KeyError as exc:
        raise NotImplementedError(f"cpp_new has no registered lowerer for {name!r}") from exc


def lane_family_key(node) -> tuple | None:
    """Return topology/static-parameter identity for descriptor-driven lifting."""
    spec = descriptor(node.opcode)
    if not spec.lane_lift:
        return None
    parameters = dict(node.parameters)
    ignored = {*spec.lane_parameters, "scratch_offset"}
    invariants = tuple((name, value) for name, value in node.parameters if name not in ignored)
    return node.opcode, node.children, invariants
