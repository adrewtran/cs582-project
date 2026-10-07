"""Registry of dataset loaders. ``load("crm")`` returns a :class:`src.data.Dataset`."""

from __future__ import annotations

from collections.abc import Callable

from src.data import Dataset
from src.datasets import crm

LOADERS: dict[str, Callable[[], Dataset]] = {
    "crm": crm.build,
}


def load(name: str) -> Dataset:
    try:
        return LOADERS[name]()
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}") from None
