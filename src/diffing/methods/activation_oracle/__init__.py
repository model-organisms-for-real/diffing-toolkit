"""Activation oracle; prompt preparation can be imported without model loading."""

__all__ = ["ActivationOracleMethod"]


def __getattr__(name):
    if name == "ActivationOracleMethod":
        from .method import ActivationOracleMethod

        return ActivationOracleMethod
    raise AttributeError(name)
