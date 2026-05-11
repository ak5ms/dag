from __future__ import annotations

from typing import Callable

import numpy as np
from numba import boolean, float64, int64, literal_unroll, njit, types
from numba.experimental import jitclass
from numba.typed import List

from trading_dsl_engine.registry import REGISTRY, CompiledNode, OpSpec, TypeInfo


VECTOR = TypeInfo("vector")
MATRIX = TypeInfo("matrix")
SCALAR = TypeInfo("scalar")
OBJECT = TypeInfo("object")


_INPUT_NODE_CACHE: dict[int, CompiledNode] = {}
_LITERAL_NODE_CACHE: dict[float, CompiledNode] = {}


def _make_input_node(input_index: int) -> CompiledNode:
    cached = _INPUT_NODE_CACHE.get(input_index)
    if cached is not None:
        return cached

    spec = [
        ("input_index", int64),
        ("initialized", boolean),
        ("out", float64[:, :]),
    ]

    @jitclass(spec)
    class InputOp:
        def __init__(self, input_index: int):
            self.input_index = input_index
            self.initialized = False
            self.out = np.empty((1, 1), dtype=np.float64)

        def on_data(self, frame2d):
            row = frame2d[self.input_index]
            if (not self.initialized) or self.out.shape[0] != row.shape[0]:
                self.out = np.empty((row.shape[0], 1), dtype=np.float64)
                self.initialized = True
            self.out[:, 0] = row

        def emit(self):
            return self.out

    node = CompiledNode(VECTOR, InputOp.class_type.instance_type, lambda: InputOp(input_index))
    _INPUT_NODE_CACHE[input_index] = node
    return node


def _make_literal_node(value: float) -> CompiledNode:
    cached = _LITERAL_NODE_CACHE.get(value)
    if cached is not None:
        return cached

    spec = [
        ("value", float64),
        ("out", float64[:, :]),
    ]

    @jitclass(spec)
    class LiteralOp:
        def __init__(self, value: float):
            self.value = value
            self.out = np.empty((1, 1), dtype=np.float64)
            self.out[0, 0] = value

        def on_data(self, frame2d):
            return

        def emit(self):
            return self.out

    node = CompiledNode(SCALAR, LiteralOp.class_type.instance_type, lambda: LiteralOp(value))
    _LITERAL_NODE_CACHE[value] = node
    return node


def make_nary_op(
    name: str,
    arity: int,
    kernel: Callable[[np.ndarray], float],
    *,
    axis: int | None = None,
) -> None:
    if axis not in (None, -1, 0, 1):
        raise ValueError(f"{name} axis must be one of None, -1, 0, or 1")
    if axis is not None and arity != 1:
        raise ValueError(f"{name} axis reducers currently support exactly one arg")

    kernel_jit = njit(inline="always")(kernel)
    reduce_axis = -2 if axis is None else axis

    def validator(types: list[TypeInfo]) -> TypeInfo:
        if len(types) != arity:
            raise ValueError(f"{name} expects exactly {arity} args")
        if reduce_axis == -1:
            if types[0].kind == "object":
                raise ValueError(f"{name} arg must emit scalar/vector/matrix, not object")
            return SCALAR
        if reduce_axis == 0:
            if types[0].kind == "object":
                raise ValueError(f"{name} arg must emit scalar/vector/matrix, not object")
            return SCALAR if types[0].kind in ("scalar", "vector") else MATRIX
        if reduce_axis == 1:
            if types[0].kind == "object":
                raise ValueError(f"{name} arg must emit scalar/vector/matrix, not object")
            return SCALAR if types[0].kind == "scalar" else VECTOR
        kinds = {t.kind for t in types}
        if kinds <= {"scalar"}:
            return SCALAR
        if kinds <= {"scalar", "vector"}:
            return VECTOR
        if kinds <= {"scalar", "matrix"}:
            return MATRIX
        if kinds == {"vector"}:
            return VECTOR
        if kinds == {"matrix"}:
            return MATRIX
        raise ValueError(f"{name} received incompatible arg kinds: {sorted(kinds)}")

    node_cache: dict[tuple[object, ...], CompiledNode] = {}

    def builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
        child_types = tuple(child.instance_type for child in children)
        cached = node_cache.get(child_types)
        if cached is not None:
            return cached

        spec = [
            ("children", types.Tuple(child_types)),
            ("initialized", boolean),
            ("out", float64[:, :]),
            ("scratch", float64[:]),
        ]

        @jitclass(spec)
        class NaryOp:
            def __init__(self, children):
                self.children = children
                self.initialized = False
                self.out = np.empty((1, 1), dtype=np.float64)
                self.scratch = np.empty(arity, dtype=np.float64)

            def _ensure_out(self, rows: int64, cols: int64):
                if not self.initialized or self.out.shape[0] != rows or self.out.shape[1] != cols:
                    self.out = np.empty((rows, cols), dtype=np.float64)
                    self.initialized = True

            def _ensure_scratch(self, size: int64):
                if self.scratch.shape[0] != size:
                    self.scratch = np.empty(size, dtype=np.float64)

            def on_data(self, frame2d):
                for child in literal_unroll(self.children):
                    child.on_data(frame2d)
                rows = 1
                cols = 1
                for child in literal_unroll(self.children):
                    v = child.emit()
                    if v.shape[0] != 1:
                        rows = v.shape[0]
                    if v.shape[1] != 1:
                        cols = v.shape[1]

                if reduce_axis == -1:
                    src = self.children[0].emit()
                    n = src.shape[0] * src.shape[1]
                    self._ensure_out(1, 1)
                    self._ensure_scratch(n)
                    idx = 0
                    for i in range(src.shape[0]):
                        for j in range(src.shape[1]):
                            self.scratch[idx] = src[i, j]
                            idx += 1
                    self.out[0, 0] = kernel_jit(self.scratch)
                    return

                if reduce_axis == 0:
                    src = self.children[0].emit()
                    self._ensure_out(1, src.shape[1])
                    self._ensure_scratch(src.shape[0])
                    for j in range(src.shape[1]):
                        for i in range(src.shape[0]):
                            self.scratch[i] = src[i, j]
                        self.out[0, j] = kernel_jit(self.scratch)
                    return

                if reduce_axis == 1:
                    src = self.children[0].emit()
                    self._ensure_out(src.shape[0], 1)
                    self._ensure_scratch(src.shape[1])
                    for i in range(src.shape[0]):
                        for j in range(src.shape[1]):
                            self.scratch[j] = src[i, j]
                        self.out[i, 0] = kernel_jit(self.scratch)
                    return

                self._ensure_out(rows, cols)
                self._ensure_scratch(arity)
                for i in range(rows):
                    for j in range(cols):
                        idx = 0
                        for child in literal_unroll(self.children):
                            arg = child.emit()
                            ai = i if arg.shape[0] > 1 else 0
                            aj = j if arg.shape[1] > 1 else 0
                            val = arg[ai, aj]
                            self.scratch[idx] = val
                            idx += 1
                        self.out[i, j] = kernel_jit(self.scratch)

            def emit(self):
                return self.out

        out_type = validator([child.type_info for child in children])
        child_ctors = tuple(child.ctor for child in children)

        def _ctor():
            nodes = tuple(fn() for fn in child_ctors)
            return NaryOp(nodes)

        node = CompiledNode(out_type, NaryOp.class_type.instance_type, _ctor)
        node_cache[child_types] = node
        return node

    REGISTRY.register(OpSpec(name=name, validator=validator, builder=builder))


def _cumsum_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) != 1 or types[0].kind != "vector":
        raise ValueError("cumsum expects one vector arg")
    return VECTOR


def _cumsum_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    spec = [("src", src.instance_type), ("initialized", boolean), ("state", float64[:, :]), ("out", float64[:, :])]

    @jitclass(spec)
    class CumSumOp:
        def __init__(self, src):
            self.src = src
            self.initialized = False
            self.state = np.empty((1, 1), dtype=np.float64)
            self.out = np.empty((1, 1), dtype=np.float64)

        def on_data(self, frame2d):
            self.src.on_data(frame2d)
            x = self.src.emit()
            n = x.shape[0]
            if (not self.initialized) or self.out.shape[0] != n:
                self.state = np.zeros((n, 1), dtype=np.float64)
                self.out = np.empty((n, 1), dtype=np.float64)
                self.initialized = True
            for i in range(n):
                xv = x[i, 0]
                if not np.isnan(xv):
                    self.state[i, 0] += xv
                self.out[i, 0] = self.state[i, 0]

        def emit(self):
            return self.out

    return CompiledNode(VECTOR, CumSumOp.class_type.instance_type, lambda: CumSumOp(src.ctor()))


def _shift_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) not in (2, 3):
        raise ValueError("shift expects args: vector, nlag[, max_size]")
    if types[0].kind != "vector":
        raise ValueError("shift first arg must be vector")
    if types[1].kind != "scalar":
        raise ValueError("shift nlag arg must be scalar")
    if len(types) == 3 and types[2].kind != "scalar":
        raise ValueError("shift max_size arg must be scalar")
    return VECTOR


def _shift_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    nlag = children[1]
    max_size_literal = literals[2] if len(children) == 3 else literals[1]
    if np.isnan(max_size_literal):
        raise ValueError("shift max_size must be a static numeric literal")
    max_size = int(round(max_size_literal))
    if max_size < 0:
        raise ValueError("shift max_size must be >= 0")
    spec = [
        ("src", src.instance_type),
        ("nlag", nlag.instance_type),
        ("initialized", boolean),
        ("max_size", int64),
        ("ring", float64[:, :]),
        ("head", int64),
        ("size", int64),
        ("out", float64[:, :]),
    ]

    @jitclass(spec)
    class ShiftOp:
        def __init__(self, src, nlag, max_size):
            self.src = src
            self.nlag = nlag
            self.initialized = False
            self.max_size = max_size
            self.ring = np.empty((1, 1), dtype=np.float64)
            self.head = 0
            self.size = 0
            self.out = np.empty((1, 1), dtype=np.float64)

        def on_data(self, frame2d):
            self.src.on_data(frame2d)
            self.nlag.on_data(frame2d)
            x = self.src.emit()
            lag_value = self.nlag.emit()[0, 0]
            n = x.shape[0]
            ring_len = self.max_size + 1
            if (not self.initialized) or self.out.shape[0] != n:
                self.ring = np.empty((n, ring_len), dtype=np.float64)
                self.out = np.empty((n, 1), dtype=np.float64)
                self.head = 0
                self.size = 0
                self.ring[:, :] = np.nan
                self.initialized = True
            self.ring[:, self.head] = x[:, 0]
            if np.isnan(lag_value):
                self.out[:, 0] = np.nan
            else:
                lag = int(round(lag_value))
                if lag < 0 or lag > self.max_size:
                    raise ValueError("shift nlag must be between 0 and max_size")
                if self.size >= lag:
                    idx = self.head - lag
                    if idx < 0:
                        idx += ring_len
                    self.out[:, 0] = self.ring[:, idx]
                else:
                    self.out[:, 0] = np.nan
            self.head += 1
            if self.head == ring_len:
                self.head = 0
            if self.size < ring_len:
                self.size += 1

        def emit(self):
            return self.out

    return CompiledNode(VECTOR, ShiftOp.class_type.instance_type, lambda: ShiftOp(src.ctor(), nlag.ctor(), max_size))


def _ewm_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) not in (2, 3):
        raise ValueError("ewm expects 2 or 3 args")
    if types[0].kind != "vector":
        raise ValueError("ewm first arg must be vector")
    if types[1].kind != "scalar":
        raise ValueError("ewm second arg must be scalar span")
    return VECTOR


def _ewm_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    span = children[1]
    alpha = 2.0 / (literals[1] + 1.0)

    spec = [
        ("src", src.instance_type),
        ("initialized", boolean),
        ("has_state", boolean),
        ("alpha", float64),
        ("state", float64[:, :]),
        ("out", float64[:, :]),
    ]

    @jitclass(spec)
    class EWMOp:
        def __init__(self, src, alpha):
            self.src = src
            self.initialized = False
            self.has_state = False
            self.alpha = alpha
            self.state = np.empty((1, 1), dtype=np.float64)
            self.out = np.empty((1, 1), dtype=np.float64)

        def on_data(self, frame2d):
            self.src.on_data(frame2d)
            x = self.src.emit()
            rows, cols = x.shape
            if not self.initialized:
                self.state = np.empty((rows, cols), dtype=np.float64)
                self.out = np.empty((rows, cols), dtype=np.float64)
                self.initialized = True
            if not self.has_state:
                self.state[:, :] = x
                self.has_state = True
            else:
                a = self.alpha
                b = 1.0 - a
                for i in range(rows):
                    for j in range(cols):
                        xv = x[i, j]
                        sv = self.state[i, j]
                        if np.isnan(xv):
                            self.state[i, j] = sv
                        elif np.isnan(sv):
                            self.state[i, j] = xv
                        else:
                            self.state[i, j] = a * xv + b * sv
            self.out[:, :] = self.state

        def emit(self):
            return self.out

    return CompiledNode(VECTOR, EWMOp.class_type.instance_type, lambda: EWMOp(src.ctor(), alpha))


def _xs_rank_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) != 1:
        raise ValueError("xs_rank expects one arg")
    if types[0].kind != "vector":
        raise ValueError("xs_rank arg must be vector")
    return VECTOR


def _xs_rank_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    spec = [
        ("src", src.instance_type),
        ("initialized", boolean),
        ("out", float64[:, :]),
    ]

    @jitclass(spec)
    class XsRankOp:
        def __init__(self, src):
            self.src = src
            self.initialized = False
            self.out = np.empty((1, 1), dtype=np.float64)

        def on_data(self, frame2d):
            self.src.on_data(frame2d)
            x = self.src.emit()
            n = x.shape[0]
            if not self.initialized:
                self.out = np.empty((n, 1), dtype=np.float64)
                self.initialized = True

            vals = np.empty(n, dtype=np.float64)
            valid = np.empty(n, dtype=np.float64)
            m = 0
            for i in range(n):
                vals[i] = x[i, 0]
                if np.isnan(vals[i]):
                    self.out[i, 0] = np.nan
                else:
                    valid[m] = vals[i]
                    m += 1
            if m == 0:
                return
            idx = np.argsort(valid[:m])

            pos = 0
            while pos < m:
                start = pos
                v = valid[idx[pos]]
                pos += 1
                while pos < m and valid[idx[pos]] == v:
                    pos += 1
                rank = pos / m
                target_count = 0
                for i in range(n):
                    if not np.isnan(vals[i]) and vals[i] == v:
                        self.out[i, 0] = rank
                        target_count += 1
                        if target_count == pos - start:
                            break

        def emit(self):
            return self.out

    return CompiledNode(VECTOR, XsRankOp.class_type.instance_type, lambda: XsRankOp(src.ctor()))


def _outer_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) != 1 or types[0].kind != "vector":
        raise ValueError("outer expects one vector arg")
    return MATRIX


def _outer_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    spec = [
        ("src", src.instance_type),
        ("initialized", boolean),
        ("out", float64[:, :]),
    ]

    @jitclass(spec)
    class OuterOp:
        def __init__(self, src):
            self.src = src
            self.initialized = False
            self.out = np.empty((1, 1), dtype=np.float64)

        def on_data(self, frame2d):
            self.src.on_data(frame2d)
            x = self.src.emit()
            n = x.shape[0]
            if not self.initialized:
                self.out = np.empty((n, n), dtype=np.float64)
                self.initialized = True
            for i in range(n):
                for j in range(n):
                    self.out[i, j] = x[i, 0] * x[j, 0]

        def emit(self):
            return self.out

    return CompiledNode(MATRIX, OuterOp.class_type.instance_type, lambda: OuterOp(src.ctor()))


def _bspline_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) != 2:
        raise ValueError("bspline expects 2 args: x, n_basis")
    if types[0].kind != "vector":
        raise ValueError("bspline first arg must be vector")
    if types[1].kind != "scalar":
        raise ValueError("bspline n_basis arg must be scalar")
    return MATRIX


@njit(inline="always")
def _periodic_basis_eval(centers: np.ndarray, sigma: float, x: float, out_row: np.ndarray):
    total = 0.0
    inv_sigma2 = 1.0 / (sigma * sigma)
    for i in range(centers.shape[0]):
        d = abs(x - centers[i])
        if 1.0 - d < d:
            d = 1.0 - d
        v = np.exp(-0.5 * d * d * inv_sigma2)
        out_row[i] = v
        total += v
    if total <= 1e-18 or np.isnan(total):
        val = 1.0 / centers.shape[0]
        for i in range(centers.shape[0]):
            out_row[i] = val
        return
    inv_total = 1.0 / total
    for i in range(centers.shape[0]):
        out_row[i] *= inv_total


def _bspline_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    n_basis = int(round(literals[1]))
    if n_basis <= 0:
        raise ValueError("bspline n_basis must be >= 1")
    spec = [
        ("src", src.instance_type),
        ("initialized", boolean),
        ("out", float64[:, :]),
        ("centers", float64[:]),
        ("sigma", float64),
        ("n_basis", int64),
    ]

    @jitclass(spec)
    class BSplineOp:
        def __init__(self, src, n_basis):
            self.src = src
            self.initialized = False
            self.out = np.empty((1, 1), dtype=np.float64)
            self.n_basis = n_basis
            self.sigma = 1.0 / n_basis
            self.centers = np.empty(n_basis, dtype=np.float64)
            for i in range(n_basis):
                self.centers[i] = i / n_basis

        def on_data(self, frame2d):
            self.src.on_data(frame2d)
            x = self.src.emit()
            n = x.shape[0]
            if (not self.initialized) or self.out.shape[0] != n:
                self.out = np.empty((n, self.n_basis), dtype=np.float64)
                self.initialized = True
            for i in range(n):
                xv = x[i, 0]
                if np.isnan(xv):
                    self.out[i, :] = np.nan
                    continue
                if xv < 0.0:
                    xv = 0.0
                elif xv > 1.0:
                    xv = 1.0
                row = self.out[i]
                _periodic_basis_eval(self.centers, self.sigma, xv, row)

        def emit(self):
            return self.out

    return CompiledNode(
        MATRIX,
        BSplineOp.class_type.instance_type,
        lambda: BSplineOp(src.ctor(), n_basis),
    )


def _col_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) != 2:
        raise ValueError("col expects 2 args: matrix, index")
    if types[0].kind != "matrix":
        raise ValueError("col first arg must be matrix")
    if types[1].kind != "scalar":
        raise ValueError("col second arg must be scalar index")
    return VECTOR


def _col_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    idx = int(round(literals[1]))
    if idx < 0:
        raise ValueError("col index must be >= 0")
    spec = [
        ("src", src.instance_type),
        ("initialized", boolean),
        ("idx", int64),
        ("out", float64[:, :]),
    ]

    @jitclass(spec)
    class ColOp:
        def __init__(self, src, idx):
            self.src = src
            self.initialized = False
            self.idx = idx
            self.out = np.empty((1, 1), dtype=np.float64)

        def on_data(self, frame2d):
            self.src.on_data(frame2d)
            x = self.src.emit()
            n = x.shape[0]
            if self.idx >= x.shape[1]:
                raise ValueError("col index out of bounds for matrix width")
            if (not self.initialized) or self.out.shape[0] != n:
                self.out = np.empty((n, 1), dtype=np.float64)
                self.initialized = True
            self.out[:, 0] = x[:, self.idx]

        def emit(self):
            return self.out

    return CompiledNode(VECTOR, ColOp.class_type.instance_type, lambda: ColOp(src.ctor(), idx))


def _rolling_quantile_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) != 3:
        raise ValueError("rolling_quantile expects 3 args: x, window, q")
    if types[0].kind != "vector":
        raise ValueError("rolling_quantile first arg must be vector")
    if types[1].kind != "scalar" or types[2].kind != "scalar":
        raise ValueError("rolling_quantile window and q args must be scalars")
    return VECTOR


@njit(inline="always")
def _nan_quantile_linear(buf: np.ndarray, n: int64, q: float) -> float:
    if n <= 0:
        return np.nan
    vals = np.empty(n, dtype=np.float64)
    m = 0
    for i in range(n):
        v = buf[i]
        if not np.isnan(v):
            vals[m] = v
            m += 1
    if m == 0:
        return np.nan
    vals = np.sort(vals[:m])
    if m == 1:
        return vals[0]
    pos = q * (m - 1.0)
    lo = int(np.floor(pos))
    hi = int(np.ceil(pos))
    if lo == hi:
        return vals[lo]
    w = pos - lo
    return vals[lo] * (1.0 - w) + vals[hi] * w


def _rolling_quantile_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    window = int(round(literals[1]))
    q = literals[2]
    if window <= 0:
        raise ValueError("rolling_quantile window must be >= 1")
    if np.isnan(q) or q < 0.0 or q > 1.0:
        raise ValueError("rolling_quantile q must be between 0 and 1")
    spec = [
        ("src", src.instance_type),
        ("initialized", boolean),
        ("window", int64),
        ("q", float64),
        ("head", int64),
        ("size", int64),
        ("ring", float64[:, :]),
        ("scratch", float64[:]),
        ("out", float64[:, :]),
    ]

    @jitclass(spec)
    class RollingQuantileOp:
        def __init__(self, src, window, q):
            self.src = src
            self.initialized = False
            self.window = window
            self.q = q
            self.head = 0
            self.size = 0
            self.ring = np.empty((1, 1), dtype=np.float64)
            self.scratch = np.empty(window, dtype=np.float64)
            self.out = np.empty((1, 1), dtype=np.float64)

        def on_data(self, frame2d):
            self.src.on_data(frame2d)
            x = self.src.emit()
            n = x.shape[0]
            if (not self.initialized) or self.out.shape[0] != n:
                self.ring = np.empty((n, self.window), dtype=np.float64)
                self.out = np.empty((n, 1), dtype=np.float64)
                self.head = 0
                self.size = 0
                self.ring[:, :] = np.nan
                self.initialized = True
            self.ring[:, self.head] = x[:, 0]
            self.head += 1
            if self.head == self.window:
                self.head = 0
            if self.size < self.window:
                self.size += 1
            for i in range(n):
                for j in range(self.size):
                    idx = self.head - self.size + j
                    if idx < 0:
                        idx += self.window
                    self.scratch[j] = self.ring[i, idx]
                self.out[i, 0] = _nan_quantile_linear(self.scratch, self.size, self.q)

        def emit(self):
            return self.out

    return CompiledNode(
        VECTOR,
        RollingQuantileOp.class_type.instance_type,
        lambda: RollingQuantileOp(src.ctor(), window, q),
    )


def _typed_universe_groups(groups: tuple[tuple[int, ...], ...]):
    typed_groups = List()
    for group in groups:
        arr = np.empty(len(group), dtype=np.int64)
        for i, idx in enumerate(group):
            arr[i] = idx
        typed_groups.append(arr)
    return typed_groups


def _make_universe_groupby_node(op_node: CompiledNode, groups: tuple[tuple[int, ...], ...]) -> CompiledNode:
    if len(groups) == 0:
        raise ValueError("groupby universe expects at least one group")
    if op_node.type_info.kind == "object":
        raise ValueError("groupby universe op arg must emit scalar/vector/matrix, not object")
    out_type = VECTOR if op_node.type_info.kind == "scalar" else op_node.type_info
    groups_list_type = types.ListType(types.Array(int64, 1, "C"))
    op_list_type = types.ListType(op_node.instance_type)

    spec = [
        ("group_indices", groups_list_type),
        ("ops", op_list_type),
        ("initialized", boolean),
        ("out", float64[:, :]),
    ]

    @jitclass(spec)
    class UniverseGroupByOp:
        def __init__(self, group_indices, ops):
            self.group_indices = group_indices
            self.ops = ops
            self.initialized = False
            self.out = np.empty((1, 1), dtype=np.float64)

        def _ensure_out(self, n_cols: int64, width: int64):
            if (not self.initialized) or self.out.shape[0] != n_cols or self.out.shape[1] != width:
                self.out = np.empty((n_cols, width), dtype=np.float64)
                self.initialized = True

        def on_data(self, frame2d):
            n_inputs = frame2d.shape[0]
            n_cols = frame2d.shape[1]
            width = 1
            for g in range(len(self.group_indices)):
                idxs = self.group_indices[g]
                for j in range(idxs.shape[0]):
                    if idxs[j] >= n_cols:
                        raise ValueError("universe column index out of bounds for input width")
                group_frame = np.empty((n_inputs, idxs.shape[0]), dtype=np.float64)
                for r in range(n_inputs):
                    for c in range(idxs.shape[0]):
                        group_frame[r, c] = frame2d[r, idxs[c]]
                self.ops[g].on_data(group_frame)
                y = self.ops[g].emit()
                if y.shape[1] > width:
                    width = y.shape[1]

            self._ensure_out(n_cols, width)
            self.out[:, :] = np.nan
            for g in range(len(self.group_indices)):
                idxs = self.group_indices[g]
                y = self.ops[g].emit()
                for member_pos in range(idxs.shape[0]):
                    dest = idxs[member_pos]
                    src_row = member_pos if y.shape[0] == idxs.shape[0] else 0
                    for w in range(width):
                        src_col = w if y.shape[1] > 1 else 0
                        self.out[dest, w] = y[src_row, src_col]

        def emit(self):
            return self.out

    op_ctors = [op_node.ctor for _ in groups]

    def _ctor():
        ops = List.empty_list(op_node.instance_type)
        for make_op in op_ctors:
            ops.append(make_op())
        return UniverseGroupByOp(_typed_universe_groups(groups), ops)

    return CompiledNode(out_type, UniverseGroupByOp.class_type.instance_type, _ctor)


def _groupby_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) != 2:
        raise ValueError("groupby expects 2 args: key, op")
    if types[0].kind != "vector":
        raise ValueError("groupby key arg must be vector")
    if types[1].kind == "object":
        raise ValueError("groupby op arg must emit scalar/vector/matrix, not object")
    return types[1]


def _groupby_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    key_node = children[0]
    op_node = children[1]
    groups_list_type = types.ListType(op_node.instance_type)
    max_groups = 256

    spec = [
        ("key_node", key_node.instance_type),
        ("groups", groups_list_type),
        ("group_keys", int64[:]),
        ("n_groups", int64),
        ("active_group_idx", int64),
    ]

    @jitclass(spec)
    class GroupByOp:
        def __init__(self, key_node, groups):
            self.key_node = key_node
            self.groups = groups
            self.group_keys = np.empty(max_groups, dtype=np.int64)
            for i in range(max_groups):
                self.group_keys[i] = -1
            self.n_groups = 0
            self.active_group_idx = -1

        def _find_group(self, key: int64):
            for i in range(self.n_groups):
                if self.group_keys[i] == key:
                    return i
            return -1

        def _append_group(self, key: int64):
            if self.n_groups >= self.group_keys.shape[0]:
                raise ValueError("groupby exceeded max groups (256)")
            self.group_keys[self.n_groups] = key
            self.n_groups += 1
            return self.n_groups - 1

        def on_data(self, frame2d):
            self.key_node.on_data(frame2d)
            k = self.key_node.emit()
            first = k[0, 0]
            if np.isnan(first):
                raise ValueError("groupby key cannot be NaN")
            key = int(first)
            for i in range(1, k.shape[0]):
                if np.isnan(k[i, 0]) or int(k[i, 0]) != key:
                    raise ValueError("groupby currently requires a single shared key across instruments per tick")

            idx = self._find_group(key)
            if idx < 0:
                idx = self._append_group(key)
            self.active_group_idx = idx
            self.groups[idx].on_data(frame2d)

        def emit(self):
            if self.active_group_idx < 0:
                raise ValueError("groupby has no active group")
            return self.groups[self.active_group_idx].emit()

    return CompiledNode(
        children[1].type_info,
        GroupByOp.class_type.instance_type,
        lambda: GroupByOp(key_node.ctor(), _build_group_slots(op_node.ctor, max_groups, op_node.instance_type)),
    )


def _build_group_slots(group_ctor, n_slots: int, instance_type):
    groups = List.empty_list(instance_type)
    for _ in range(n_slots):
        groups.append(group_ctor())
    return groups


_ridge_state_spec = [
    ("initialized", boolean),
    ("n_instruments", int64),
    ("n_features", int64),
    ("t", int64),
    ("xx", float64[:, :]),
    ("xy", float64[:]),
    ("last_xx", int64[:, :]),
    ("last_xy", int64[:]),
    ("has_xx", boolean[:, :]),
    ("has_xy", boolean[:]),
    ("beta", float64[:]),
    ("preds", float64[:, :]),
]


@jitclass(_ridge_state_spec)
class _RidgeState:
    def __init__(self):
        self.initialized = False
        self.n_instruments = 0
        self.n_features = 0
        self.t = 0
        self.xx = np.empty((1, 1), dtype=np.float64)
        self.xy = np.empty(1, dtype=np.float64)
        self.last_xx = np.empty((1, 1), dtype=np.int64)
        self.last_xy = np.empty(1, dtype=np.int64)
        self.has_xx = np.empty((1, 1), dtype=np.bool_)
        self.has_xy = np.empty(1, dtype=np.bool_)
        self.beta = np.empty(1, dtype=np.float64)
        self.preds = np.empty((1, 1), dtype=np.float64)


@njit(inline="always")
def _dot_vec(a, b):
    acc = 0.0
    for i in range(a.shape[0]):
        acc += a[i] * b[i]
    return acc


@njit
def _pairwise_weighted_moments(x, y, w):
    """Build pairwise-NaN-aware X'WX and X'Wy snapshots.

    This mirrors the intended NumPy semantics without requiring BLAS at runtime:
    valid_x = np.isfinite(x); x0 = np.where(valid_x, x, 0.0)
    xx_new[j, k] = np.sum(x0[:, j] * x0[:, k] * w0)
    with a separate validity mask for each statistic.
    """
    n = x.shape[0]
    k = x.shape[1]
    xx_new = np.zeros((k, k), dtype=np.float64)
    xy_new = np.zeros(k, dtype=np.float64)
    xx_valid = np.zeros((k, k), dtype=np.bool_)
    xy_valid = np.zeros(k, dtype=np.bool_)

    valid_x = np.isfinite(x)
    valid_y = np.isfinite(y)
    valid_w = np.isfinite(w)
    x0 = np.where(valid_x, x, 0.0)
    y0 = np.where(valid_y, y, 0.0)
    w0 = np.where(valid_w, w, 0.0)

    for j in range(k):
        xy_acc = 0.0
        xy_seen = False
        for i in range(n):
            if valid_x[i, j] and valid_y[i] and valid_w[i]:
                xy_acc += x0[i, j] * y0[i] * w0[i]
                xy_seen = True
        xy_new[j] = xy_acc
        xy_valid[j] = xy_seen

        for m in range(j, k):
            xx_acc = 0.0
            xx_seen = False
            for i in range(n):
                if valid_x[i, j] and valid_x[i, m] and valid_w[i]:
                    xx_acc += x0[i, j] * x0[i, m] * w0[i]
                    xx_seen = True
            xx_new[j, m] = xx_acc
            xx_new[m, j] = xx_acc
            xx_valid[j, m] = xx_seen
            xx_valid[m, j] = xx_seen

    return xx_new, xy_new, xx_valid, xy_valid


@njit
def _pairwise_weighted_moments_matrix(x, y, w):
    """Build pairwise-NaN-aware X'WX and X'Wy snapshots for a dense W."""
    n = x.shape[0]
    k = x.shape[1]
    xx_new = np.zeros((k, k), dtype=np.float64)
    xy_new = np.zeros(k, dtype=np.float64)
    xx_valid = np.zeros((k, k), dtype=np.bool_)
    xy_valid = np.zeros(k, dtype=np.bool_)

    valid_x = np.isfinite(x)
    valid_y = np.isfinite(y)
    valid_w = np.isfinite(w)
    x0 = np.where(valid_x, x, 0.0)
    y0 = np.where(valid_y, y, 0.0)
    w0 = np.where(valid_w, w, 0.0)

    for j in range(k):
        xy_acc = 0.0
        xy_seen = False
        for r in range(n):
            if valid_x[r, j]:
                for c in range(n):
                    if valid_w[r, c] and valid_y[c]:
                        xy_acc += x0[r, j] * w0[r, c] * y0[c]
                        xy_seen = True
        xy_new[j] = xy_acc
        xy_valid[j] = xy_seen

        for m in range(k):
            xx_acc = 0.0
            xx_seen = False
            for r in range(n):
                if valid_x[r, j]:
                    for c in range(n):
                        if valid_w[r, c] and valid_x[c, m]:
                            xx_acc += x0[r, j] * w0[r, c] * x0[c, m]
                            xx_seen = True
            xx_new[j, m] = xx_acc
            xx_valid[j, m] = xx_seen

    return xx_new, xy_new, xx_valid, xy_valid


@njit
def _solve_linear_system(a, b, out):
    """Small dense Gaussian solve used to avoid a SciPy/BLAS runtime dependency."""
    n = b.shape[0]
    aug = np.empty((n, n + 1), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            aug[i, j] = a[i, j]
        aug[i, n] = b[i]

    eps = 1e-12
    for col in range(n):
        pivot = col
        pivot_abs = abs(aug[col, col])
        for row in range(col + 1, n):
            candidate = abs(aug[row, col])
            if candidate > pivot_abs:
                pivot = row
                pivot_abs = candidate
        if pivot_abs <= eps or np.isnan(pivot_abs):
            return False
        if pivot != col:
            for j in range(col, n + 1):
                tmp = aug[col, j]
                aug[col, j] = aug[pivot, j]
                aug[pivot, j] = tmp

        pivot_value = aug[col, col]
        for row in range(col + 1, n):
            factor = aug[row, col] / pivot_value
            aug[row, col] = 0.0
            for j in range(col + 1, n + 1):
                aug[row, j] -= factor * aug[col, j]

    for i in range(n - 1, -1, -1):
        rhs = aug[i, n]
        for j in range(i + 1, n):
            rhs -= aug[i, j] * out[j]
        diag = aug[i, i]
        if abs(diag) <= eps or np.isnan(diag):
            return False
        out[i] = rhs / diag
    return True


@njit
def _solve_scaled_ridge(xx, xy, ridge, beta):
    k = xy.shape[0]
    system = np.empty((k, k), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            system[i, j] = xx[i, j]
        system[i, i] += ridge * xx[i, i]
    candidate = np.empty(k, dtype=np.float64)
    ok = _solve_linear_system(system, xy, candidate)
    if ok:
        for i in range(k):
            beta[i] = candidate[i]
    return ok


def _ridge_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) < 4:
        raise ValueError("Ridge expects x..., y, hl, lambda or x..., y, weights, hl, lambda")
    has_explicit_weights = len(types) >= 5
    feature_types = types[:-4] if has_explicit_weights else types[:-3]
    if len(feature_types) == 0:
        raise ValueError("Ridge expects at least one feature arg")
    for t in feature_types:
        if t.kind not in ("vector", "matrix"):
            raise ValueError("Ridge feature args must be vector or matrix")
    y_type = types[-4] if has_explicit_weights else types[-3]
    if y_type.kind != "vector":
        raise ValueError("Ridge y must be vector")
    if has_explicit_weights and types[-3].kind not in ("scalar", "vector", "matrix"):
        raise ValueError("Ridge weights must be scalar, vector, or matrix")
    if types[-2].kind != "scalar" or types[-1].kind != "scalar":
        raise ValueError("Ridge hl and lambda must be scalar")
    return OBJECT


def _ridge_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    has_explicit_weights = len(children) >= 5
    if has_explicit_weights:
        feature_nodes = children[:-4]
        y_node = children[-4]
        w_node = children[-3]
    else:
        feature_nodes = children[:-3]
        y_node = children[-3]
        w_node = _make_literal_node(1.0)
    hl_node = children[-2]
    lam_node = children[-1]
    x_nodes_type = types.Tuple(tuple(node.instance_type for node in feature_nodes))
    spec = [
        ("x_nodes", x_nodes_type),
        ("y_node", y_node.instance_type),
        ("w_node", w_node.instance_type),
        ("hl_node", hl_node.instance_type),
        ("lam_node", lam_node.instance_type),
        ("state", _RidgeState.class_type.instance_type),
    ]

    @jitclass(spec)
    class RidgeOp:
        def __init__(self, x_nodes, y_node, w_node, hl_node, lam_node, state):
            self.x_nodes = x_nodes
            self.y_node = y_node
            self.w_node = w_node
            self.hl_node = hl_node
            self.lam_node = lam_node
            self.state = state

        def _reset_state(self, n, k):
            self.state.initialized = True
            self.state.n_instruments = n
            self.state.n_features = k
            self.state.t = 0
            self.state.xx = np.zeros((k, k), dtype=np.float64)
            self.state.xy = np.zeros(k, dtype=np.float64)
            self.state.last_xx = np.zeros((k, k), dtype=np.int64)
            self.state.last_xy = np.zeros(k, dtype=np.int64)
            self.state.has_xx = np.zeros((k, k), dtype=np.bool_)
            self.state.has_xy = np.zeros(k, dtype=np.bool_)
            self.state.beta = np.zeros(k, dtype=np.float64)
            self.state.preds = np.empty((n, 1), dtype=np.float64)

        def on_data(self, frame2d):
            self.y_node.on_data(frame2d)
            self.w_node.on_data(frame2d)
            self.hl_node.on_data(frame2d)
            self.lam_node.on_data(frame2d)
            for node in literal_unroll(self.x_nodes):
                node.on_data(frame2d)

            y2d = self.y_node.emit()
            w2d = self.w_node.emit()
            hl = self.hl_node.emit()[0, 0]
            lam = self.lam_node.emit()[0, 0]
            n = y2d.shape[0]
            total_k = 0
            for node in literal_unroll(self.x_nodes):
                total_k += node.emit().shape[1]

            if (not self.state.initialized) or self.state.n_instruments != n or self.state.n_features != total_k:
                self._reset_state(n, total_k)

            k = self.state.n_features
            xmat = np.empty((n, k), dtype=np.float64)
            idx = 0
            for node in literal_unroll(self.x_nodes):
                feat = node.emit()
                width = feat.shape[1]
                xmat[:, idx : idx + width] = feat
                idx += width

            for i in range(n):
                if np.isnan(y2d[i, 0]) or np.any(np.isnan(xmat[i, :])):
                    self.state.preds[i, 0] = np.nan
                else:
                    self.state.preds[i, 0] = _dot_vec(self.state.beta, xmat[i, :])

            w_cols = w2d.shape[1]
            scalar_weight = w2d.shape[0] == 1 and w_cols == 1
            if (not scalar_weight) and (w2d.shape[0] != n or (w_cols != 1 and w_cols != n)):
                raise ValueError("Ridge weights shape must be scalar, (n, 1), or (n, n)")

            if np.isnan(hl) or hl <= 0.0:
                rho = 0.0
            else:
                rho = np.exp(np.log(0.5) / hl)
            alpha = 1.0 - rho
            if alpha < 0.0:
                alpha = 0.0
            if alpha > 1.0:
                alpha = 1.0
            if np.isnan(lam) or lam < 0.0:
                lam = 0.0

            y = y2d[:, 0]
            if scalar_weight:
                wvec = np.empty(n, dtype=np.float64)
                wvec[:] = w2d[0, 0]
                xx_new, xy_new, xx_valid, xy_valid = _pairwise_weighted_moments(xmat, y, wvec)
            elif w_cols == 1:
                xx_new, xy_new, xx_valid, xy_valid = _pairwise_weighted_moments(xmat, y, w2d[:, 0])
            else:
                xx_new, xy_new, xx_valid, xy_valid = _pairwise_weighted_moments_matrix(xmat, y, w2d)
            now = self.state.t

            for j in range(k):
                if xy_valid[j]:
                    if self.state.has_xy[j]:
                        dt = now - self.state.last_xy[j]
                        a = alpha**dt
                        self.state.xy[j] = self.state.xy[j] * (1.0 - a) + xy_new[j] * a
                    else:
                        self.state.xy[j] = xy_new[j]
                        self.state.has_xy[j] = True
                    self.state.last_xy[j] = now

                for m in range(k):
                    if xx_valid[j, m]:
                        if self.state.has_xx[j, m]:
                            dt = now - self.state.last_xx[j, m]
                            a = alpha**dt
                            self.state.xx[j, m] = self.state.xx[j, m] * (1.0 - a) + xx_new[j, m] * a
                        else:
                            self.state.xx[j, m] = xx_new[j, m]
                            self.state.has_xx[j, m] = True
                        self.state.last_xx[j, m] = now

            for j in range(k):
                for m in range(j + 1, k):
                    self.state.xx[j, m] = 0.5 * (self.state.xx[j, m] + self.state.xx[m, j])
                    self.state.xx[m, j] = self.state.xx[j, m]
                    last = self.state.last_xx[j, m]
                    if self.state.last_xx[m, j] > last:
                        last = self.state.last_xx[m, j]
                    self.state.last_xx[j, m] = last
                    self.state.last_xx[m, j] = last
                    has = self.state.has_xx[j, m] or self.state.has_xx[m, j]
                    self.state.has_xx[j, m] = has
                    self.state.has_xx[m, j] = has

            _solve_scaled_ridge(self.state.xx, self.state.xy, lam, self.state.beta)
            self.state.t += 1

        def emit(self):
            return self.state

    feature_ctors = [node.ctor for node in feature_nodes]
    y_ctor = y_node.ctor
    w_ctor = w_node.ctor
    hl_ctor = hl_node.ctor
    lam_ctor = lam_node.ctor

    def _ctor():
        x_nodes = tuple(fn() for fn in feature_ctors)
        return RidgeOp(x_nodes, y_ctor(), w_ctor(), hl_ctor(), lam_ctor(), _RidgeState())

    return CompiledNode(OBJECT, RidgeOp.class_type.instance_type, _ctor)


def _get_beta_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) != 1 or types[0].kind != "object":
        raise ValueError("get_beta expects one object arg")
    return VECTOR


def _get_beta_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    spec = [("src", src.instance_type), ("out", float64[:, :])]

    @jitclass(spec)
    class GetBetaOp:
        def __init__(self, src):
            self.src = src
            self.out = np.empty((1, 1), dtype=np.float64)

        def on_data(self, frame2d):
            self.src.on_data(frame2d)
            state = self.src.emit()
            k = state.beta.shape[0]
            if self.out.shape[0] != k:
                self.out = np.empty((k, 1), dtype=np.float64)
            self.out[:, 0] = state.beta

        def emit(self):
            return self.out

    return CompiledNode(VECTOR, GetBetaOp.class_type.instance_type, lambda: GetBetaOp(src.ctor()))


def _get_preds_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) != 1 or types[0].kind != "object":
        raise ValueError("get_preds expects one object arg")
    return VECTOR


def _get_preds_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    src = children[0]
    spec = [("src", src.instance_type)]

    @jitclass(spec)
    class GetPredsOp:
        def __init__(self, src):
            self.src = src

        def on_data(self, frame2d):
            self.src.on_data(frame2d)

        def emit(self):
            state = self.src.emit()
            return state.preds

    return CompiledNode(VECTOR, GetPredsOp.class_type.instance_type, lambda: GetPredsOp(src.ctor()))


def register_builtin_ops() -> None:
    if getattr(register_builtin_ops, "_done", False):
        return

    make_nary_op("div", 2, lambda args: np.nan if args[1] == 0.0 else args[0] / args[1])
    make_nary_op("add", 2, lambda args: args[0] + args[1])
    make_nary_op("sub", 2, lambda args: args[0] - args[1])
    make_nary_op("mod", 2, lambda args: args[0] % args[1])
    make_nary_op("mul", 2, lambda args: args[0] * args[1])
    make_nary_op("eq", 2, lambda args: np.nan if (np.isnan(args[0]) or np.isnan(args[1])) else (1.0 if args[0] == args[1] else 0.0))
    make_nary_op("ne", 2, lambda args: np.nan if (np.isnan(args[0]) or np.isnan(args[1])) else (1.0 if args[0] != args[1] else 0.0))
    make_nary_op("and", 2, lambda args: np.nan if (np.isnan(args[0]) or np.isnan(args[1])) else (1.0 if (args[0] != 0.0 and args[1] != 0.0) else 0.0))
    make_nary_op("or", 2, lambda args: np.nan if (np.isnan(args[0]) or np.isnan(args[1])) else (1.0 if (args[0] != 0.0 or args[1] != 0.0) else 0.0))
    make_nary_op("and_", 2, lambda args: np.nan if (np.isnan(args[0]) or np.isnan(args[1])) else (1.0 if (args[0] != 0.0 and args[1] != 0.0) else 0.0))
    make_nary_op("or_", 2, lambda args: np.nan if (np.isnan(args[0]) or np.isnan(args[1])) else (1.0 if (args[0] != 0.0 or args[1] != 0.0) else 0.0))
    make_nary_op("xor", 2, lambda args: np.nan if (np.isnan(args[0]) or np.isnan(args[1])) else (1.0 if ((args[0] != 0.0) != (args[1] != 0.0)) else 0.0))
    make_nary_op("where", 3, lambda args: args[1] if args[0] != 0.0 else args[2])
    make_nary_op("abs", 1, lambda args: np.abs(args[0]))
    make_nary_op("isnan", 1, lambda args: 1.0 if np.isnan(args[0]) else 0.0)
    make_nary_op("fillna", 2, lambda args: args[0] if np.isnan(args[1]) else args[0])
    make_nary_op("ln", 1, lambda args: np.log(args[0]))
    REGISTRY.register(OpSpec(name="ewm", validator=_ewm_validator, builder=_ewm_builder))
    REGISTRY.register(OpSpec(name="cumsum", validator=_cumsum_validator, builder=_cumsum_builder))
    REGISTRY.register(OpSpec(name="shift", validator=_shift_validator, builder=_shift_builder))
    REGISTRY.register(OpSpec(name="xs_rank", validator=_xs_rank_validator, builder=_xs_rank_builder))
    REGISTRY.register(OpSpec(name="outer", validator=_outer_validator, builder=_outer_builder))
    REGISTRY.register(OpSpec(name="bspline", validator=_bspline_validator, builder=_bspline_builder))
    REGISTRY.register(OpSpec(name="col", validator=_col_validator, builder=_col_builder))
    REGISTRY.register(OpSpec(name="rolling_quantile", validator=_rolling_quantile_validator, builder=_rolling_quantile_builder))
    make_nary_op("mean", 1, lambda args: np.nanmean(args), axis=-1)
    REGISTRY.register(OpSpec(name="groupby", validator=_groupby_validator, builder=_groupby_builder))
    REGISTRY.register(OpSpec(name="Ridge", validator=_ridge_validator, builder=_ridge_builder))
    REGISTRY.register(OpSpec(name="get_beta", validator=_get_beta_validator, builder=_get_beta_builder))
    REGISTRY.register(OpSpec(name="get_preds", validator=_get_preds_validator, builder=_get_preds_builder))
    register_builtin_ops._done = True


__all__ = [
    "register_builtin_ops",
    "_make_input_node",
    "_make_literal_node",
    "_make_universe_groupby_node",
    "VECTOR",
    "SCALAR",
    "MATRIX",
]
