from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ValueKind = Literal["scalar", "vector"]


@dataclass(frozen=True, slots=True)
class ValueType:
    kind: ValueKind
    width: int = 1
    dtype: str = "float64"


SCALAR = ValueType("scalar")
VECTOR = ValueType("vector")
