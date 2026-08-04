"""Key-hinted version of the production roll-return graph.

The mathematical expression is identical to :mod:`flows.pov`. The only change is
metadata attached to ``session_start0`` when it is used as the POV group key.

``row_scalar=True`` asserts that every instrument lane carries the same session
start on a given row. cpp_stream can therefore resolve the group once per row
instead of once per instrument. This assertion must only be used when the source
really is lane-invariant; a false assertion changes grouping semantics.

No ``num_keys`` hint is supplied because session-start timestamps form an
unbounded domain. Bounded hints are appropriate for keys such as weekday or
minute-of-hour, not for absolute timestamps.
"""

from types import SimpleNamespace

from flows.pov import PovFields, RollRets
from trading_dsl_engine.base.keys import key


PovFieldsWithKeys = SimpleNamespace(
    ts=PovFields.ts,
    session_start=key(
        PovFields.session_start,
        row_scalar=True,
        # Input timestamps in the current pipeline are float64 because NaN is a
        # valid missing marker. Remove or update this assertion for another dtype.
        dtype="float64",
    ),
    session_end=PovFields.session_end,
    volume=PovFields.volume,
    is_tradable=PovFields.is_tradable,
)


class RollRetsWithKeys(RollRets):
    """RollRets using row-scalar group routing for the POV session key."""

    def roll_rets(self, days_roll: int = 2, **kwargs):
        kwargs.setdefault("f", PovFieldsWithKeys)
        return super().roll_rets(days_roll=days_roll, **kwargs)


roll_rets_keys = RollRetsWithKeys().roll_rets()


__all__ = [
    "PovFieldsWithKeys",
    "RollRetsWithKeys",
    "roll_rets_keys",
]
