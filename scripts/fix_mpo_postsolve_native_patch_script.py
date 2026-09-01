from pathlib import Path


path = Path(__file__).with_name("apply_mpo_postsolve_native_patch.py")
text = path.read_text()

helper_anchor = '''def patch_direct_clarabel() -> None:\n'''
helper = '''def replace_all(path: Path, old: str, new: str) -> None:\n    text = path.read_text()\n    if new in text and old not in text:\n        return\n    count = text.count(old)\n    if count <= 0:\n        raise RuntimeError(\n            f"{path}: expected at least one patch anchor: {old[:100]!r}"\n        )\n    path.write_text(text.replace(old, new))\n\n\n'''
if "def replace_all(" not in text:
    if text.count(helper_anchor) != 1:
        raise RuntimeError("missing helper insertion anchor")
    text = text.replace(helper_anchor, helper + helper_anchor, 1)

old_call = '''    replace_once(\n        path,\n        """        DualLayout,\n        FieldAlias,\n""",\n        """        ConstraintValueLayout,\n        DualLayout,\n        FieldAlias,\n""",\n    )\n'''
new_call = old_call.replace("replace_once", "replace_all", 1)
if old_call in text:
    text = text.replace(old_call, new_call, 1)
elif new_call not in text:
    raise RuntimeError("missing ambiguous import patch call")

path.write_text(text)
