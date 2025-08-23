from collections.abc import Mapping
from abc import abstractmethod, ABC
from collections.abc import Sequence
from typing import (
    Generic,
    Self,
    TypeVar,
    overload,
    SupportsFloat,
    Protocol,
    runtime_checkable,
)


K = TypeVar("K")
Scalar = TypeVar("Scalar", bound=SupportsFloat, covariant=True)
M = TypeVar("M", bound="VectorMapping", covariant=True)


@runtime_checkable
class VectorMappingConstructor(Protocol[M]):
    @classmethod
    def from_vectors(cls, keys: Sequence[K], values: Sequence[Scalar]) -> M: ...


class VectorMapping(Mapping[K, Scalar], Generic[K, Scalar], ABC):
    @overload
    def __add__(self, other: Mapping[K, Scalar]) -> Self: ...
    @overload
    def __add__(self, other: Scalar) -> Self: ...
    @abstractmethod
    def __add__(self, other: Mapping[K, Scalar] | Scalar) -> Self: ...

    @overload
    def __radd__(self, other: Mapping[K, Scalar]) -> Self: ...
    @overload
    def __radd__(self, other: Scalar) -> Self: ...
    @abstractmethod
    def __radd__(self, other: Mapping[K, Scalar] | Scalar) -> Self: ...

    @overload
    def __sub__(self, other: Mapping[K, Scalar]) -> Self: ...
    @overload
    def __sub__(self, other: Scalar) -> Self: ...
    @abstractmethod
    def __sub__(self, other: Mapping[K, Scalar] | Scalar) -> Self: ...

    @overload
    def __rsub__(self, other: Mapping[K, Scalar]) -> Self: ...
    @overload
    def __rsub__(self, other: Scalar) -> Self: ...
    @abstractmethod
    def __rsub__(self, other: Mapping[K, Scalar] | Scalar) -> Self: ...

    @overload
    def __mul__(self, other: Mapping[K, Scalar]) -> Self: ...
    @overload
    def __mul__(self, other: Scalar) -> Self: ...
    @abstractmethod
    def __mul__(self, other: Mapping[K, Scalar] | Scalar) -> Self: ...

    @overload
    def __rmul__(self, other: Mapping[K, Scalar]) -> Self: ...
    @overload
    def __rmul__(self, other: Scalar) -> Self: ...
    @abstractmethod
    def __rmul__(self, other: Mapping[K, Scalar] | Scalar) -> Self: ...

    @overload
    def __truediv__(self, other: Mapping[K, Scalar]) -> Self: ...
    @overload
    def __truediv__(self, other: Scalar) -> Self: ...
    @abstractmethod
    def __truediv__(self, other: Mapping[K, Scalar] | Scalar) -> Self: ...

    @overload
    def __rtruediv__(self, other: Mapping[K, Scalar]) -> Self: ...
    @overload
    def __rtruediv__(self, other: Scalar) -> Self: ...
    @abstractmethod
    def __rtruediv__(self, other: Mapping[K, Scalar] | Scalar) -> Self: ...

    # Default implementations of sum() and mean(), override this
    # where possible with faster vector operations
    def sum(self) -> float:
        s = 0.0
        for v in self.values():
            s += float(v)
        return s

    def mean(self) -> float:
        n = len(self)
        if n == 0:
            raise ValueError("mean of empty VectorMapping")
        return self.sum() / n
