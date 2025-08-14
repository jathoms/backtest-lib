from collections.abc import Mapping
from typing import Protocol, TypeVar

SecurityName = str
Price = float
Volume = float


T_co = TypeVar("T_co", covariant=True)


Universe = list[str]


class UniverseMapping(Mapping[SecurityName, T_co], Protocol): ...


class UniversePrices(UniverseMapping[Price], Protocol): ...


class UniverseVolume(UniverseMapping[Volume], Protocol): ...


class UniverseMask(UniverseMapping[bool], Protocol): ...
