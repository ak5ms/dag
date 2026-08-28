from __future__ import annotations

import ast
from pathlib import Path
import re
import shutil
import sys


ROOT = Path.cwd()
PAYLOAD = Path(__file__).resolve().parent
MARKER = "GP runtime ergonomics v14"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if new in text:
        return text
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def copy_payload() -> None:
    copies = {
        "result_mapping.py": "src/trading_dsl_engine/base/result_mapping.py",
        "gp_visualization.py": "src/flows/gp/visualization.py",
        "test_result_mapping.py": "tests/trading_dsl_engine/test_result_mapping.py",
        "test_expr_ergonomics.py": "tests/trading_dsl_engine/test_expr_ergonomics.py",
        "test_gp_visualization.py": "tests/flows/gp/test_visualization.py",
    }
    for source, destination in copies.items():
        target = ROOT / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PAYLOAD / source, target)


def patch_expr() -> None:
    path = ROOT / "src/trading_dsl_engine/base/parser.py"
    text = read(path)
    tree = ast.parse(text)
    expr_class = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Expr"),
        None,
    )
    if expr_class is None:
        raise RuntimeError("parser.py does not define Expr")
    existing = {
        node.name
        for node in expr_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    snippets = []
    if "pipe" not in existing:
        snippets.append(
            '''    def pipe(self, function, *args, **kwargs):
        """Apply ``function`` to this expression, following pandas pipe semantics."""
        if isinstance(function, tuple):
            if (
                len(function) != 2
                or not callable(function[0])
                or not isinstance(function[1], str)
            ):
                raise TypeError(
                    "Expr.pipe tuple form must be (callable, target_keyword)"
                )
            callable_, target = function
            if target in kwargs:
                raise ValueError(
                    f"{target!r} is both the pipe target and an explicit keyword"
                )
            kwargs[target] = self
            return callable_(*args, **kwargs)
        if not callable(function):
            raise TypeError("Expr.pipe expects a callable or (callable, keyword) tuple")
        return function(self, *args, **kwargs)
'''
        )
    if "__getattr__" not in existing:
        snippets.append(
            '''    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        from trading_dsl_engine.base import dsl as _dsl

        resolver = getattr(_dsl, "_resolve_dsl_method", None)
        function = resolver(name) if callable(resolver) else getattr(_dsl, name, None)
        if not callable(function):
            raise AttributeError(
                f"{type(self).__name__!s} has no attribute {name!r}"
            )

        def bound(*args, **kwargs):
            return function(self, *args, **kwargs)

        bound.__name__ = name
        bound.__doc__ = getattr(function, "__doc__", None)
        return bound
'''
        )
    if not snippets:
        return
    lines = text.splitlines(keepends=True)
    insert_line = expr_class.lineno
    if expr_class.body:
        first = expr_class.body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            insert_line = first.end_lineno or first.lineno
    insertion = "\n" + "\n".join(snippets) + "\n"
    lines.insert(insert_line, insertion)
    write(path, "".join(lines))


DSL_ADAPTER = r'''

# GP runtime ergonomics v14
import functools as _dsl_functools
import importlib as _dsl_importlib
import inspect as _dsl_inspect
import numbers as _dsl_numbers
import sys as _dsl_sys
from collections.abc import Mapping as _DslMapping

_DSL_METHOD_REGISTRY = globals().get("_DSL_METHOD_REGISTRY", {})


def _register_dsl_method(name, function):
    """Register a Python DSL helper for lazy ``Expr.<name>(...)`` binding."""
    if not isinstance(name, str) or not name:
        raise ValueError("DSL method names must be nonempty strings")
    if not callable(function) or isinstance(function, type):
        raise TypeError("DSL methods must be callable non-class objects")
    _DSL_METHOD_REGISTRY[name] = function
    return function


def _callable_from_registration(value, name):
    if callable(value) and not isinstance(value, type):
        return value
    for attribute in (
        "function", "func", "fn", "callable", "wrapper", "python_fn", "factory"
    ):
        candidate = getattr(value, attribute, None)
        if callable(candidate) and not isinstance(candidate, type):
            return candidate
    return None


def _registry_lookup(name):
    registered_name = False
    for module_name in (
        "trading_dsl_engine.base.dsl",
        "trading_dsl_engine.base.custom",
        "trading_dsl_engine.base.registry",
    ):
        try:
            module = _dsl_importlib.import_module(module_name)
        except Exception:
            continue
        for attribute_name, container in tuple(vars(module).items()):
            if not isinstance(container, _DslMapping):
                continue
            if name in container:
                registered_name = True
                candidate = _callable_from_registration(container[name], name)
                if candidate is not None:
                    return candidate, True
            lowered_attribute = attribute_name.lower()
            if not any(token in lowered_attribute for token in ("dsl", "function", "registry", "op")):
                continue
            for key, value in tuple(container.items()):
                candidate_name = key if isinstance(key, str) else getattr(value, "name", None)
                if candidate_name != name:
                    continue
                registered_name = True
                candidate = _callable_from_registration(value, name)
                if candidate is not None:
                    return candidate, True
    return None, registered_name


def _resolve_dsl_method(name):
    cached = _DSL_METHOD_REGISTRY.get(name)
    if callable(cached):
        return cached

    module = _dsl_sys.modules[__name__]
    candidate = globals().get(name)
    if candidate is None:
        try:
            candidate = getattr(module, name)
        except (AttributeError, KeyError):
            candidate = None
    if callable(candidate) and not isinstance(candidate, type):
        return _register_dsl_method(name, candidate)

    candidate, registered_name = _registry_lookup(name)
    if candidate is not None:
        return _register_dsl_method(name, candidate)

    signatures = globals().get("_DSL_OP_SIGNATURES", {})
    registered_name = registered_name or (
        isinstance(signatures, _DslMapping) and name in signatures
    )
    if registered_name:
        for loaded in tuple(_dsl_sys.modules.values()):
            if loaded is None:
                continue
            try:
                candidate = vars(loaded).get(name)
            except Exception:
                continue
            if callable(candidate) and not isinstance(candidate, type):
                return _register_dsl_method(name, candidate)
    raise AttributeError(f"unknown DSL method {name!r}")


def _remember_registration(call_args, call_kwargs, result):
    names = []
    explicit_name = call_kwargs.get("name")
    if isinstance(explicit_name, str):
        names.append(explicit_name)
    names.extend(arg for arg in call_args if isinstance(arg, str))
    callables = [
        value
        for value in (*call_args, *call_kwargs.values(), result)
        if callable(value) and not isinstance(value, type)
    ]
    if not callables:
        return
    function = callables[-1]
    names.extend(
        name
        for name in (
            getattr(function, "__name__", None),
            getattr(result, "name", None),
            getattr(result, "__name__", None),
        )
        if isinstance(name, str)
    )
    for name in names:
        if name and not name.startswith("_"):
            _register_dsl_method(name, function)


_original_register_dsl_function = globals().get("register_dsl_function")
if callable(_original_register_dsl_function) and not getattr(
    _original_register_dsl_function, "_expr_method_aware", False
):
    @_dsl_functools.wraps(_original_register_dsl_function)
    def register_dsl_function(*args, **kwargs):
        result = _original_register_dsl_function(*args, **kwargs)
        supplied_callable = any(
            callable(value) and not isinstance(value, type)
            for value in (*args, *kwargs.values())
        )
        if callable(result) and not supplied_callable:
            original_decorator = result

            @_dsl_functools.wraps(original_decorator)
            def decorator(function):
                registered = original_decorator(function)
                _remember_registration((function,), kwargs, registered)
                return registered

            return decorator
        _remember_registration(args, kwargs, result)
        return result

    register_dsl_function._expr_method_aware = True


def _scalar_like(reference, value):
    multiply = _resolve_dsl_method("mul")
    fill = _resolve_dsl_method("fillna")
    addition = _resolve_dsl_method("add")
    return addition(fill(multiply(reference, 0.0), 0.0), value)


def _is_scalar_literal(value):
    return isinstance(value, _dsl_numbers.Number)


def _install_scalar_vector_adapter(name, aliases=()):
    try:
        original = _resolve_dsl_method(name)
    except AttributeError:
        return
    if getattr(original, "_scalar_vector_adapter", False):
        return
    try:
        signature = _dsl_inspect.signature(original)
        parameter_names = tuple(signature.parameters)
    except (TypeError, ValueError):
        parameter_names = ()
    first_name = parameter_names[0] if parameter_names else None
    second_name = parameter_names[1] if len(parameter_names) > 1 else None

    @_dsl_functools.wraps(original)
    def adapted(*args, **kwargs):
        positional = list(args)
        reference = positional[0] if positional else (
            kwargs.get(first_name) if first_name is not None else None
        )
        value_found = len(positional) > 1
        value = positional[1] if value_found else None
        selected_keyword = None
        if not value_found:
            for keyword in tuple(
                name_ for name_ in (second_name, *aliases) if isinstance(name_, str)
            ):
                if keyword in kwargs:
                    selected_keyword = keyword
                    value = kwargs[keyword]
                    value_found = True
                    break
        if reference is not None and value_found and _is_scalar_literal(value):
            value = _scalar_like(reference, value)
            if len(positional) > 1:
                positional[1] = value
            elif second_name is not None:
                if selected_keyword is not None and selected_keyword != second_name:
                    kwargs.pop(selected_keyword)
                kwargs[second_name] = value
            elif selected_keyword is not None:
                kwargs[selected_keyword] = value
            else:
                positional.append(value)
        elif selected_keyword is not None and second_name is not None and selected_keyword != second_name:
            kwargs[second_name] = kwargs.pop(selected_keyword)
        return original(*positional, **kwargs)

    adapted._scalar_vector_adapter = True
    globals()[name] = adapted
    _DSL_METHOD_REGISTRY[name] = adapted


for _weighted_name in (
    "xs_weighted_mean", "xs_weighted_sum", "xs_weighted_std", "xs_weighted_var"
):
    _install_scalar_vector_adapter(
        _weighted_name, aliases=("w", "weight", "weights")
    )
for _vector_name in (
    "xs_vector_projection",
    "xs_regression_projection",
    "xs_vector_proj",
    "xs_vector_neut",
    "xs_regression_neut",
    "xs_cov",
    "xs_corr",
):
    _install_scalar_vector_adapter(_vector_name)
'''


def patch_dsl() -> None:
    path = ROOT / "src/trading_dsl_engine/base/dsl.py"
    text = read(path)
    if MARKER not in text:
        write(path, text.rstrip() + DSL_ADAPTER + "\n")


def patch_types() -> None:
    path = ROOT / "src/flows/gp/types.py"
    text = read(path)
    if "class IntegerParam" not in text:
        anchor = '''@dataclass(frozen=True)
class QuantileParam(StaticValue):
'''
        classes = '''@dataclass(frozen=True)
class IntegerParam(ScalarNumber):
    """Any finite compile-time integer, including zero and negatives."""

    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool):
            raise TypeError("IntegerParam cannot be bool")
        numeric = float(self.value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("IntegerParam must be a finite integer")
        object.__setattr__(self, "value", int(numeric))


@dataclass(frozen=True)
class NonNegativeInt(IntegerParam):
    value: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.value < 0:
            raise ValueError("NonNegativeInt must be >= 0")


@dataclass(frozen=True)
class NonNegativeFloat(ScalarNumber):
    value: float

    def __post_init__(self) -> None:
        value = float(self.value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("NonNegativeFloat must be finite and >= 0")
        object.__setattr__(self, "value", value)


'''
        text = replace_once(text, anchor, classes + anchor, label="types numeric classes")
    for name, anchor in (
        ("IntegerParam", '    "KthIgnoreSpec",\n'),
        ("NonNegativeFloat", '    "NumericRow",\n'),
        ("NonNegativeInt", '    "NonNegativeFloat",\n'),
    ):
        export = f'    "{name}",\n'
        if export not in text:
            text = replace_once(text, anchor, anchor + export, label=f"types export {name}")
    write(path, text)


def patch_pset() -> None:
    path = ROOT / "src/flows/gp/pset.py"
    text = read(path)
    if "IntegerParam," not in text:
        text = replace_once(
            text,
            "    KthIgnoreSpec,\n    NumericRow,\n",
            "    KthIgnoreSpec,\n    IntegerParam,\n    NonNegativeFloat,\n    NonNegativeInt,\n    NumericRow,\n",
            label="pset type imports",
        )
    if "integers: tuple[int, ...]" not in text:
        text = replace_once(
            text,
            "    grammar: GrammarPolicy = GrammarPolicy()\n    positive_ints:",
            "    grammar: GrammarPolicy = GrammarPolicy()\n"
            "    integers: tuple[int, ...] = (-30, -20, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 20, 30, 60, 120, 240, 720, 1440)\n"
            "    nonnegative_floats: tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)\n"
            "    positive_ints:",
            label="pset config fields",
        )
    text = re.sub(
        r"positive_ints: tuple\[int, \.\.\.\] = \([^\n]+\)",
        "positive_ints: tuple[int, ...] = (1, 2, 3, 4, 5, 10, 20, 30, 60, 120, 240, 720, 1440)",
        text,
        count=1,
    )
    text = re.sub(
        r"positive_floats: tuple\[float, \.\.\.\] = \([^\n]+\)",
        "positive_floats: tuple[float, ...] = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)",
        text,
        count=1,
    )
    text = re.sub(
        r"negative_floats: tuple\[float, \.\.\.\] = \([^\n]+\)",
        "negative_floats: tuple[float, ...] = (-10.0, -5.0, -3.0, -2.0, -1.0, -0.5, -0.25, -0.1, -0.05, -0.01, -0.001)",
        text,
        count=1,
    )
    if "integers must contain only finite integers" not in text:
        validation_anchor = '''        if not isinstance(self.grammar, GrammarPolicy):
            raise TypeError("grammar must be a GrammarPolicy")
'''
        validation = '''        if not self.integers or any(
            isinstance(value, bool)
            or not math.isfinite(float(value))
            or not float(value).is_integer()
            for value in self.integers
        ):
            raise ValueError("integers must contain only finite integers")
        if not self.nonnegative_floats or any(
            not math.isfinite(float(value)) or float(value) < 0.0
            for value in self.nonnegative_floats
        ):
            raise ValueError(
                "nonnegative_floats must contain only finite values >= 0"
            )
'''
        text = replace_once(
            text,
            validation_anchor,
            validation_anchor + validation,
            label="pset config validation",
        )
    if "integer_values = sorted" not in text:
        terminal_anchor = "    positive_int_values = sorted({int(v) for v in config.positive_ints})\n"
        terminal_code = '''    integer_values = sorted({int(value) for value in config.integers})
    for value in integer_values:
        _add_terminal(pset, IntegerParam(value), IntegerParam, f"integer_{_safe_name(value)}")
        if value >= 0:
            _add_terminal(
                pset,
                NonNegativeInt(value),
                NonNegativeInt,
                f"nonnegative_int_{value}",
            )
    nonnegative_float_values = sorted(
        {float(value) for value in config.nonnegative_floats}
    )
    for value in nonnegative_float_values:
        _add_terminal(
            pset,
            NonNegativeFloat(value),
            NonNegativeFloat,
            f"nonnegative_float_{_safe_name(f'{value:g}')}",
        )
    scalar_values = sorted(
        {float(value) for value in config.integers}
        | {float(value) for value in config.nonnegative_floats}
        | {float(value) for value in config.positive_floats}
        | {float(value) for value in config.negative_floats}
    )
    for value in scalar_values:
        _add_terminal(
            pset,
            ScalarNumber(value),
            ScalarNumber,
            f"scalar_number_{_safe_name(f'{value:g}')}",
        )
'''
        text = replace_once(
            text,
            terminal_anchor,
            terminal_code + terminal_anchor,
            label="pset static terminals",
        )
    weighted_anchor = '        _core(reg, "xs_weighted_mean", (row_type, DimensionlessRow), row_type, tag)\n'
    weighted_scalar = '        _public(reg, "xs_weighted_mean", (row_type, ScalarNumber), row_type, f"{tag}_scalar_weight")\n'
    if weighted_scalar not in text:
        text = replace_once(
            text,
            weighted_anchor,
            weighted_anchor + weighted_scalar,
            label="pset scalar weighted mean",
        )
    projection_anchor = '''        for name in ("xs_vector_projection", "xs_regression_projection"):
            _core(reg, name, (row_type, NumericRow), row_type, tag)
'''
    projection_new = projection_anchor + '''            _public(
                reg,
                name,
                (row_type, ScalarNumber),
                row_type,
                f"{tag}_scalar",
            )
'''
    if "f\"{tag}_scalar\"," not in text[text.find(projection_anchor): text.find(projection_anchor) + 600]:
        text = replace_once(
            text,
            projection_anchor,
            projection_new,
            label="pset scalar projection",
        )
    vector_anchor = '''        for name in ("xs_vector_proj", "xs_vector_neut"):
            _core(reg, name, (row_type, NumericRow), row_type, tag)
'''
    vector_new = vector_anchor + '''            _public(
                reg,
                name,
                (row_type, ScalarNumber),
                row_type,
                f"{tag}_scalar",
            )
'''
    if vector_anchor in text and "scalar vector convenience" not in text:
        text = replace_once(
            text,
            vector_anchor,
            vector_new + "        # scalar vector convenience\n",
            label="pset scalar vector ops",
        )
    write(path, text)


def patch_compile_formula_exports() -> None:
    marker = "nested formula mapping adapter v14"
    roots = [
        ROOT / "src/trading_dsl_engine/base",
        ROOT / "src/trading_dsl_engine/cpp_stream",
        ROOT / "src/trading_dsl_engine/jax_flat",
    ]
    candidates = {ROOT / "src/trading_dsl_engine/__init__.py"}
    for directory in roots:
        for path in directory.rglob("*.py"):
            text = read(path)
            if re.search(r"^def compile_formula\b", text, flags=re.MULTILINE) or (
                path.name == "__init__.py" and "compile_formula" in text
            ):
                candidates.add(path)
    adapter = '''

# nested formula mapping adapter v14
try:
    compile_formula
except NameError:
    pass
else:
    from trading_dsl_engine.base.result_mapping import support_formula_mappings as _support_formula_mappings

    compile_formula = _support_formula_mappings(compile_formula)
'''
    for path in sorted(candidates):
        text = read(path)
        if marker not in text and "compile_formula" in text:
            write(path, text.rstrip() + adapter + "\n")

    for path in (
        ROOT / "src/trading_dsl_engine/base/__init__.py",
        ROOT / "src/trading_dsl_engine/__init__.py",
    ):
        text = read(path)
        import_line = (
            "from trading_dsl_engine.base.result_mapping import "
            "FormulaResultMapping, ResultMapping\n"
        )
        if import_line not in text:
            write(path, text.rstrip() + "\n\n" + import_line)


def patch_gp_exports() -> None:
    path = ROOT / "src/flows/gp/__init__.py"
    text = read(path)
    marker = "GP visualization exports v14"
    if marker not in text:
        addition = '''

# GP visualization exports v14
from flows.gp.visualization import (
    GPGraphEdge,
    GPGraphExplorer,
    GPGraphNode,
    explore_gp,
    explore_pset,
    gp_graph_data,
    plot_gp_graph,
    plot_pset,
    visualize_pset,
)

_gp_visualization_exports = (
    "GPGraphEdge",
    "GPGraphExplorer",
    "GPGraphNode",
    "explore_gp",
    "explore_pset",
    "gp_graph_data",
    "plot_gp_graph",
    "plot_pset",
    "visualize_pset",
)
if "__all__" in globals():
    __all__ = type(__all__)((*__all__, *_gp_visualization_exports))
'''
        write(path, text.rstrip() + addition + "\n")


def patch_existing_tests() -> None:
    replacements = {
        "(1, 2, 3, 5, 10, 20, 60, 120, 240, 1440)": "(1, 2, 3, 4, 5, 10, 20, 30, 60, 120, 240, 720, 1440)",
        "(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)": "(0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)",
        "(-1.0, -0.5, -0.25, -0.1)": "(-10.0, -5.0, -3.0, -2.0, -1.0, -0.5, -0.25, -0.1, -0.05, -0.01, -0.001)",
    }
    for path in (ROOT / "tests/flows/gp").glob("*.py"):
        text = read(path)
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        if updated != text:
            write(path, updated)


def patch_docs() -> None:
    sections = {
        ROOT / "README.md": '''

### Nested outputs and expression ergonomics

`compile_formula` accepts insertion-ordered, arbitrarily nested mappings of
formulas while retaining the existing fused multi-output compilation. After a
run, `result.load()` returns a `ResultMapping` with the same shape; `.map(fn)`
transforms every leaf and `.flatten()` returns tuple-keyed paths.

```python
formulas = {"signal": {"fast": x.ewm(span=5), "slow": x.ewm(span=20)}, "rank": x.xs_rank()}
values = compile_formula(formulas, data).run().load()
flat = values.map(np.asarray).flatten()
```

Expressions support pandas-style `.pipe(...)`, and every built-in or
`register_dsl_function` helper is available through method chaining. Scalar
literals are broadcast for cross-sectional vector slots, so
`x.xs_weighted_mean(w=1)` is equivalent to explicit unit weights.
''',
        ROOT / "src/flows/gp/README.md": '''

## Interactive grammar explorer

`explore_pset(make_pset())` returns a Plotly-backed explorer for terminals,
types, and operators. The HTML view has a live search box and clicking a node
filters to its direct type relations. `show=True` opens a standalone browser
view; `write_html(...)` saves a portable report. The default grammar now also
contains broader exact integer, nonnegative integer/float, positive, negative,
and general scalar terminal grids.
''',
        ROOT / "AGENTS.md": '''

- Nested formula mappings are a Python result-boundary feature: flatten leaves
  through the existing fused multi-output compiler, then reconstruct only in
  `result.load()`; never add a second execution loop.
- Registered DSL helpers must remain lazily method-chainable from `Expr`, and
  cross-sectional vector parameters should accept scalar literals through a
  generic expression broadcast rather than backend-specific hot-path branches.
- GP grammar visualization is compile-time tooling and must not enter live or
  batch execution paths.
''',
    }
    for path, section in sections.items():
        text = read(path)
        heading = section.strip().splitlines()[0]
        if heading not in text:
            write(path, text.rstrip() + section + "\n")


def remove_temporary_workflows() -> None:
    workflow_dir = ROOT / ".github/workflows"
    for pattern in (
        "export-agent-gp-*.yml",
        "export-agent-gp-*.yaml",
        "finalize-gp-runtime-ergonomics*.yml",
        "finalize-gp-runtime-ergonomics*.yaml",
    ):
        for path in workflow_dir.glob(pattern):
            path.unlink()


def main() -> None:
    copy_payload()
    patch_expr()
    patch_dsl()
    patch_types()
    patch_pset()
    patch_compile_formula_exports()
    patch_gp_exports()
    patch_existing_tests()
    patch_docs()
    remove_temporary_workflows()
    for path in (
        ROOT / "src/trading_dsl_engine/base/parser.py",
        ROOT / "src/trading_dsl_engine/base/dsl.py",
        ROOT / "src/flows/gp/types.py",
        ROOT / "src/flows/gp/pset.py",
    ):
        ast.parse(read(path), filename=str(path))
    print("Applied GP/runtime ergonomics changes")


if __name__ == "__main__":
    main()
