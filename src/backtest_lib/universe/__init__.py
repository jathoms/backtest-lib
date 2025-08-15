from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeVar

SecurityName = str
Price = float
Volume = float


T_co = TypeVar("T_co", covariant=True)


Universe = tuple[SecurityName]


type UniverseMapping[T_co] = Mapping[SecurityName, T_co]


type UniverseVolume = UniverseMapping[Volume]


type UniverseMask = UniverseMapping[bool]


type UniverseClosePrices = UniverseMapping[Price]


@dataclass(frozen=True)
class UniversePrices:
    close: UniverseMapping[Price]
    open: UniverseMapping[Price] | None
    high: UniverseMapping[Price] | None
    low: UniverseMapping[Price] | None
