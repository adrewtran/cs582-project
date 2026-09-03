"""Registry of dataset loaders. ``load("leads")`` etc. return a :class:`src.data.Dataset`."""

from __future__ import annotations

from collections.abc import Callable

from src.data import Dataset
from src.datasets import bank, crm, leads, telco

LOADERS: dict[str, Callable[[], Dataset]] = {
    "leads": leads.build,
    "bank": bank.build,
    "telco": telco.build,
    "crm": crm.build,
}


def load(name: str) -> Dataset:
    try:
        return LOADERS[name]()
    except KeyError:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(LOADERS)}") from None
