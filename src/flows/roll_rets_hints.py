"""Compatibility aliases for the old hinted roll-return entry point.

The production :func:`flows.pov.pov` now attaches the monotonic session ``Key``
directly to its groupby, so there is no separate hinted implementation anymore.
"""

from flows.pov import PovFields, RollRets


PovFieldsWithHints = PovFields
RollRetsWithHints = RollRets
roll_rets_hints = RollRets().roll_rets()


__all__ = [
    "PovFieldsWithHints",
    "RollRetsWithHints",
    "roll_rets_hints",
]
