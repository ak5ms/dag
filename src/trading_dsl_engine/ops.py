from __future__ import annotations

from typing import Callable

import numpy as np
from numba import boolean, float64, int64, njit
from numba.experimental import jitclass

from trading_dsl_engine.registry import REGISTRY, CompiledNode, OpSpec, TypeInfo


VECTOR = TypeInfo("vector")
MATRIX = TypeInfo("matrix")
SCALAR = TypeInfo("scalar")


def _make_input_node(input_index: int) -> CompiledNode:
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
            # frame2d shape: [n_inputs, n_instruments]
            row = frame2d[self.input_index]
            if not self.initialized:
                self.out = np.empty((row.shape[0], 1), dtype=np.float64)
                self.initialized = True
            for i in range(row.shape[0]):
                self.out[i, 0] = row[i]

        def emit(self):
            return self.out

    return CompiledNode(VECTOR, InputOp.class_type.instance_type, lambda: InputOp(input_index))


def _make_literal_node(value: float) -> CompiledNode:
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

    return CompiledNode(SCALAR, LiteralOp.class_type.instance_type, lambda: LiteralOp(value))


def make_binary_op(name: str, kernel: Callable[[float, float], float]) -> None:
    kernel_jit = njit(inline="always")(kernel)
    is_div = name == "div"
    def validator(types: list[TypeInfo]) -> TypeInfo:
        if len(types) != 2:
            raise ValueError(f"{name} expects exactly 2 args")
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

    def builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
        left, right = children
        spec = [
            ("left", left.instance_type),
            ("right", right.instance_type),
            ("initialized", boolean),
            ("out", float64[:, :]),
        ]

        @jitclass(spec)
        class BinaryOp:
            def __init__(self, left, right):
                self.left = left
                self.right = right
                self.initialized = False
                self.out = np.empty((1, 1), dtype=np.float64)

            def on_data(self, frame2d):
                self.left.on_data(frame2d)
                self.right.on_data(frame2d)
                a = self.left.emit()
                b = self.right.emit()
                rows = a.shape[0] if a.shape[0] != 1 else b.shape[0]
                cols = a.shape[1] if a.shape[1] != 1 else b.shape[1]
                if not self.initialized or self.out.shape[0] != rows or self.out.shape[1] != cols:
                    self.out = np.empty((rows, cols), dtype=np.float64)
                    self.initialized = True
                for i in range(rows):
                    ai = i if a.shape[0] > 1 else 0
                    bi = i if b.shape[0] > 1 else 0
                    for j in range(cols):
                        aj = j if a.shape[1] > 1 else 0
                        bj = j if b.shape[1] > 1 else 0
                        av = a[ai, aj]
                        bv = b[bi, bj]
                        if np.isnan(av) or np.isnan(bv):
                            self.out[i, j] = np.nan
                        elif is_div and bv == 0.0:
                            self.out[i, j] = np.nan
                        else:
                            self.out[i, j] = kernel_jit(av, bv)

            def emit(self):
                return self.out

        out_type = validator([left.type_info, right.type_info])
        return CompiledNode(out_type, BinaryOp.class_type.instance_type, lambda: BinaryOp(left.ctor(), right.ctor()))

    REGISTRY.register(OpSpec(name=name, validator=validator, builder=builder))


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
                for i in range(rows):
                    for j in range(cols):
                        self.state[i, j] = x[i, j]
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
            for i in range(rows):
                for j in range(cols):
                    self.out[i, j] = self.state[i, j]

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


def register_builtin_ops() -> None:
    if getattr(register_builtin_ops, "_done", False):
        return

    make_binary_op("div", lambda a, b: a / b)
    make_binary_op("add", lambda a, b: a + b)
    make_binary_op("sub", lambda a, b: a - b)
    REGISTRY.register(OpSpec(name="ewm", validator=_ewm_validator, builder=_ewm_builder))
    REGISTRY.register(OpSpec(name="xs_rank", validator=_xs_rank_validator, builder=_xs_rank_builder))
    REGISTRY.register(OpSpec(name="outer", validator=_outer_validator, builder=_outer_builder))
    register_builtin_ops._done = True


__all__ = ["register_builtin_ops", "_make_input_node", "_make_literal_node", "VECTOR", "SCALAR", "MATRIX"]
