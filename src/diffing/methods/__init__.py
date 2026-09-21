"""Diffing methods, loaded on demand so tokenizer diagnostics need no GPU stack."""

from importlib import import_module

_MODULES = {
    "KLDivergenceDiffingMethod": ".kl",
    "ActivationAnalysisDiffingMethod": ".activation_analysis",
    "CrosscoderDiffingMethod": ".crosscoder",
    "SAEDifferenceMethod": ".sae_difference",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name in _MODULES:
        value = getattr(import_module(_MODULES[name], __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(name)
