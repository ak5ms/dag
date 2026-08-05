"""Maximum-performance key hints for the production roll-return graph.

The formula is mathematically identical to :mod:`flows.riskmodel.roll_rets`.
Only the POV session key carries metadata:

* ``row_scalar=True``: one session-start value applies to every instrument lane;
* ``dtype="float64"``: verify the physical key dtype without converting it;
* ``monotonic=True``: session values form contiguous, non-returning runs, so
  cpp_stream may recycle one group-state slot whenever the session changes.

A false hint changes grouping semantics. In particular, do not use
``monotonic=True`` when an earlier key value can reappear later in the stream.
``num_keys`` is intentionally omitted because absolute timestamps are unbounded.
"""

from types import SimpleNamespace

from flows.pov import PovFields, RollRets
from trading_dsl_engine.base.keys import key


PovFieldsWithHints = SimpleNamespace(
    ts=PovFields.ts,
    session_start=key(
        PovFields.session_start,
        row_scalar=True,
        dtype="float64",
        monotonic=True,
    ),
    session_end=PovFields.session_end,
    volume=PovFields.volume,
    is_tradable=PovFields.is_tradable,
)


class RollRetsWithHints(RollRets):
    """RollRets with recyclable state for its monotonic POV session key."""

    def roll_rets(self, days_roll: int = 2, **kwargs):
        kwargs.setdefault("f", PovFieldsWithHints)
        return super().roll_rets(days_roll=days_roll, **kwargs)


roll_rets_hints = RollRetsWithHints().roll_rets()


__all__ = [
    "PovFieldsWithHints",
    "RollRetsWithHints",
    "roll_rets_hints",
]
