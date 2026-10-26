"""Leakage-safe NE-AFE v2 research implementation."""

from importlib import import_module


_EXPORTS = {
    "ChronologicalFold": ("splits", "ChronologicalFold"),
    "GWBPConfig": ("gwbp", "GWBPConfig"),
    "GWBPSelector": ("gwbp", "GWBPSelector"),
    "PAFAConfig": ("pafa", "PAFAConfig"),
    "PAFAFeatureGenerator": ("pafa", "PAFAFeatureGenerator"),
    "VariableMetadata": ("metadata", "VariableMetadata"),
    "chronological_three_way_split": ("splits", "chronological_three_way_split"),
    "infer_metadata": ("metadata", "infer_metadata"),
    "make_walk_forward_folds": ("splits", "make_walk_forward_folds"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module_name}", __name__), attribute)
    globals()[name] = value
    return value
