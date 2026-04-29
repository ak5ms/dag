from __future__ import annotations

from typing import Callable

import numpy as np
from numba import boolean, float64, int64, literal_unroll, njit, types
from numba.experimental import jitclass

from trading_dsl_engine.registry import REGISTRY, CompiledNode, OpSpec, TypeInfo


VECTOR = TypeInfo("vector")
MATRIX = TypeInfo("matrix")
SCALAR = TypeInfo("scalar")
OBJECT = TypeInfo("object")


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
            row = frame2d[self.input_index]
            if (not self.initialized) or self.out.shape[0] != row.shape[0]:
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
                    for j in range(self.n_basis):
                        self.out[i, j] = np.nan
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
            for i in range(n):
                self.out[i, 0] = x[i, self.idx]

        def emit(self):
            return self.out

    return CompiledNode(VECTOR, ColOp.class_type.instance_type, lambda: ColOp(src.ctor(), idx))


_ridge_state_spec = [
    ("initialized", boolean),
    ("n_instruments", int64),
    ("n_features", int64),
    ("b", float64[:]),
    ("p", float64[:, :]),
    ("beta", float64[:]),
    ("preds", float64[:, :]),
    ("beta_out", float64[:, :]),
]


@jitclass(_ridge_state_spec)
class _RidgeState:
    def __init__(self):
        self.initialized = False
        self.n_instruments = 0
        self.n_features = 0
        self.b = np.empty(1, dtype=np.float64)
        self.p = np.empty((1, 1), dtype=np.float64)
        self.beta = np.empty(1, dtype=np.float64)
        self.preds = np.empty((1, 1), dtype=np.float64)
        self.beta_out = np.empty((1, 1), dtype=np.float64)


@njit(inline="always")
def _sm_update_matrix(p, u):
    k = p.shape[0]
    v = np.empty(k, dtype=np.float64)
    for i in range(k):
        acc = 0.0
        for j in range(k):
            acc += p[i, j] * u[j]
        v[i] = acc
    den = 1.0
    for i in range(k):
        den += u[i] * v[i]
    if den <= 1e-12 or np.isnan(den):
        return
    inv_den = 1.0 / den
    for i in range(k):
        for j in range(k):
            p[i, j] -= v[i] * v[j] * inv_den


@njit(inline="always")
def _dot_vec(a, b):
    acc = 0.0
    for i in range(a.shape[0]):
        acc += a[i] * b[i]
    return acc


def _ridge_validator(types: list[TypeInfo]) -> TypeInfo:
    if len(types) < 5:
        raise ValueError("Ridge expects at least 5 args: x..., y, weights, hl, lambda")
    for t in types[:-4]:
        if t.kind not in ("vector", "matrix"):
            raise ValueError("Ridge feature args must be vector or matrix")
    if types[-4].kind != "vector":
        raise ValueError("Ridge y must be vector")
    if types[-3].kind not in ("vector", "matrix"):
        raise ValueError("Ridge weights must be vector or matrix")
    if types[-2].kind != "scalar" or types[-1].kind != "scalar":
        raise ValueError("Ridge hl and lambda must be scalar")
    return OBJECT


def _ridge_builder(children: list[CompiledNode], literals: list[float]) -> CompiledNode:
    feature_nodes = children[:-4]
    y_node = children[-4]
    w_node = children[-3]
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
        def __init__(self, x_nodes, y_node, w_node, hl_node, lam_node):
            self.x_nodes = x_nodes
            self.y_node = y_node
            self.w_node = w_node
            self.hl_node = hl_node
            self.lam_node = lam_node
            self.state = _RidgeState()

        def on_data(self, frame2d):
            self.y_node.on_data(frame2d)
            self.w_node.on_data(frame2d)
            self.hl_node.on_data(frame2d)
            self.lam_node.on_data(frame2d)
            for node in literal_unroll(self.x_nodes):
                node.on_data(frame2d)
            y = self.y_node.emit()
            w = self.w_node.emit()
            hl = self.hl_node.emit()[0, 0]
            lam = self.lam_node.emit()[0, 0]
            n = y.shape[0]
            total_k = 0
            for node in literal_unroll(self.x_nodes):
                x = node.emit()
                total_k += x.shape[1]

            if (not self.state.initialized) or self.state.n_instruments != n or self.state.n_features != total_k:
                self.state.initialized = True
                self.state.n_instruments = n
                self.state.n_features = total_k
                self.state.b = np.zeros(total_k, dtype=np.float64)
                self.state.p = np.empty((total_k, total_k), dtype=np.float64)
                self.state.beta = np.zeros(total_k, dtype=np.float64)
                for i in range(total_k):
                    for j in range(total_k):
                        self.state.p[i, j] = 1e6 if i == j else 0.0
                self.state.preds = np.empty((n, 1), dtype=np.float64)
                self.state.beta_out = np.empty((total_k, 1), dtype=np.float64)

            k = self.state.n_features
            xmat = np.empty((k, n), dtype=np.float64)
            idx = 0
            for node in literal_unroll(self.x_nodes):
                feat = node.emit()
                feat_width = feat.shape[1]
                for j in range(feat_width):
                    for i in range(n):
                        xmat[idx, i] = feat[i, j]
                    idx += 1

            xvec = np.empty(k, dtype=np.float64)
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
            eps = 1e-12
            inv_rho = 1.0 / (rho if rho > eps else eps)
            # EW forgetting:
            # - b_t = rho * b_{t-1} + alpha * X'Wy
            # - cov_t = rho * cov_{t-1} + alpha * X'WX
            # state.p stores an inverse-like precision term, so scaling cov by rho
            # corresponds to scaling precision by 1/rho.
            for i in range(k):
                self.state.b[i] = rho * self.state.b[i]
                for j in range(k):
                    self.state.p[i, j] *= inv_rho

            w_cols = w.shape[1]
            if w.shape[0] != n or (w_cols != 1 and w_cols != n):
                raise ValueError("Ridge weights shape must be (n,1) or (n,n)")

            for i in range(n):
                for j in range(k):
                    xvec[j] = xmat[j, i]
                target = y[i, 0]
                has_nan = np.isnan(target)
                if not has_nan:
                    for j in range(k):
                        if np.isnan(xvec[j]):
                            has_nan = True
                            break
                if has_nan:
                    self.state.preds[i, 0] = np.nan
                    continue
                self.state.preds[i, 0] = _dot_vec(self.state.beta, xvec)

            u = np.empty(k, dtype=np.float64)
            if w_cols == 1:
                # Vector weights path: W = diag(w). Each instrument contributes one
                # rank-1 update using sqrt(alpha * w_i) * x_i, then optional
                # diagonal ridge stabilization via per-feature rank-1 updates.
                for i in range(n):
                    wi = w[i, 0]
                    target = y[i, 0]
                    # NaN/invalid handling:
                    # - ignore rows with invalid/non-positive weights
                    # - ignore rows with NaN target or NaN features
                    if np.isnan(wi) or wi <= 0.0 or np.isnan(target):
                        continue
                    has_nan = False
                    for j in range(k):
                        xvec[j] = xmat[j, i]
                        if np.isnan(xvec[j]):
                            has_nan = True
                    if has_nan:
                        continue
                    for j in range(k):
                        self.state.b[j] += alpha * wi * xvec[j] * target
                    scale = np.sqrt(alpha * wi)
                    for j in range(k):
                        u[j] = scale * xvec[j]
                    _sm_update_matrix(self.state.p, u)
                    if lam > 0.0:
                        for j in range(k):
                            if xvec[j] == 0.0:
                                continue
                            for m in range(k):
                                u[m] = 0.0
                            u[j] = np.sqrt(alpha * lam * wi) * abs(xvec[j])
                            _sm_update_matrix(self.state.p, u)
            else:
                # Matrix weights path:
                # b contribution uses alpha * X'Wy.
                for i in range(n):
                    wi_target = 0.0
                    valid = True
                    for j in range(n):
                        wij = w[i, j]
                        yj = y[j, 0]
                        if np.isnan(wij) or np.isnan(yj):
                            valid = False
                            break
                        wi_target += wij * yj
                    if (not valid) or np.isnan(y[i, 0]):
                        continue
                    for j in range(k):
                        xvec[j] = xmat[j, i]
                    has_nan = False
                    for j in range(k):
                        if np.isnan(xvec[j]):
                            has_nan = True
                    if has_nan:
                        continue
                    for j in range(k):
                        self.state.b[j] += alpha * xvec[j] * wi_target
                # cov contribution uses alpha * X'WX.
                # For each row r, build u = (W[r, :] @ X)^T and apply a normalized
                # rank-1 Sherman-Morrison-style update. The normalization keeps the
                # rank-1 term aligned with x_r' W x_r and avoids unstable negative
                # / NaN square roots.
                for r in range(n):
                    for j in range(k):
                        u[j] = 0.0
                    valid = True
                    for c in range(n):
                        wrc = w[r, c]
                        if np.isnan(wrc):
                            valid = False
                            break
                        xc = xmat[:, c]
                        for j in range(k):
                            if np.isnan(xc[j]):
                                valid = False
                                break
                            u[j] += wrc * xc[j]
                        if not valid:
                            break
                    if not valid:
                        continue
                    for j in range(k):
                        xvec[j] = xmat[j, r]
                    has_nan = False
                    for j in range(k):
                        if np.isnan(xvec[j]):
                            has_nan = True
                    if has_nan:
                        continue
                    proj = _dot_vec(xvec, u)
                    # If projected weight is non-positive/NaN, skip that update to
                    # preserve numeric stability and keep inverse updates valid.
                    if proj <= 0.0 or np.isnan(proj):
                        continue
                    scale = np.sqrt(alpha / proj)
                    for j in range(k):
                        u[j] = scale * u[j]
                    _sm_update_matrix(self.state.p, u)
                    if lam > 0.0:
                        # Apply ridge adjustment as diagonal rank-1 increments.
                        for j in range(k):
                            diag_contrib = alpha * lam * xvec[j] * u[j]
                            if diag_contrib <= 0.0 or np.isnan(diag_contrib):
                                continue
                            for m in range(k):
                                u[m] = 0.0
                            u[j] = np.sqrt(diag_contrib)
                            _sm_update_matrix(self.state.p, u)

            for i in range(k):
                acc = 0.0
                for j in range(k):
                    acc += self.state.p[i, j] * self.state.b[j]
                self.state.beta[i] = acc
                self.state.beta_out[i, 0] = acc

        def emit(self):
            return self.state

    feature_ctors = [node.ctor for node in feature_nodes]
    y_ctor = y_node.ctor
    w_ctor = w_node.ctor
    hl_ctor = hl_node.ctor
    lam_ctor = lam_node.ctor

    def _ctor():
        x_nodes = tuple(fn() for fn in feature_ctors)
        return RidgeOp(x_nodes, y_ctor(), w_ctor(), hl_ctor(), lam_ctor())

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
            k = state.beta_out.shape[0]
            if self.out.shape[0] != k:
                self.out = np.empty((k, 1), dtype=np.float64)
            for i in range(k):
                self.out[i, 0] = state.beta_out[i, 0]

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

    make_binary_op("div", lambda a, b: a / b)
    make_binary_op("add", lambda a, b: a + b)
    make_binary_op("sub", lambda a, b: a - b)
    REGISTRY.register(OpSpec(name="ewm", validator=_ewm_validator, builder=_ewm_builder))
    REGISTRY.register(OpSpec(name="xs_rank", validator=_xs_rank_validator, builder=_xs_rank_builder))
    REGISTRY.register(OpSpec(name="outer", validator=_outer_validator, builder=_outer_builder))
    REGISTRY.register(OpSpec(name="bspline", validator=_bspline_validator, builder=_bspline_builder))
    REGISTRY.register(OpSpec(name="col", validator=_col_validator, builder=_col_builder))
    REGISTRY.register(OpSpec(name="Ridge", validator=_ridge_validator, builder=_ridge_builder))
    REGISTRY.register(OpSpec(name="get_beta", validator=_get_beta_validator, builder=_get_beta_builder))
    REGISTRY.register(OpSpec(name="get_preds", validator=_get_preds_validator, builder=_get_preds_builder))
    register_builtin_ops._done = True


__all__ = ["register_builtin_ops", "_make_input_node", "_make_literal_node", "VECTOR", "SCALAR", "MATRIX"]
