from __future__ import annotations

from abc import abstractmethod
from typing import (
    Protocol,
    Self,
    Sized,
    TypeVar,
)

_Other_scalar = TypeVar("_Other_scalar", int, float)


# Scalar invariant (binary operations must be well-defined between values)
Numeric = int | float
Scalar = TypeVar("Scalar", float, int)


class VectorOps(Sized, Protocol[Scalar]):
    # We allow `other` in this case to be a VectorOps of Any,
    # as we want to keep the flexibility of, say, adding a
    # float to a vector of ints.
    #
    # In actuality, these type signatures are not completely accurate,
    # as adding a float to a vector of ints will produce a vector of floats,
    # which is not `Self` (a vector of ints)
    @abstractmethod
    def __add__(self, other: VectorOps | _Other_scalar) -> Self: ...

    @abstractmethod
    def __radd__(self, other: VectorOps | _Other_scalar) -> Self: ...

    @abstractmethod
    def __sub__(self, other: VectorOps | _Other_scalar) -> Self: ...

    @abstractmethod
    def __rsub__(self, other: VectorOps | _Other_scalar) -> Self: ...

    @abstractmethod
    def __mul__(self, other: VectorOps | _Other_scalar) -> Self: ...

    @abstractmethod
    def __rmul__(self, other: VectorOps | _Other_scalar) -> Self: ...

    # Always widen to float on division
    @abstractmethod
    def __truediv__(self, other: VectorOps | _Other_scalar) -> VectorOps[float]: ...

    @abstractmethod
    def __rtruediv__(self, other: VectorOps | _Other_scalar) -> VectorOps[float]: ...

    @abstractmethod
    def sum(self) -> float: ...

    # Default implementations of mean(), override this
    # where possible with faster vector operations
    def mean(self) -> float:
        n = len(self)
        if n == 0:
            raise ValueError("mean of empty VectorMapping")
        return self.sum() / n
    
    def abs(self) -> Self: ...

    @abstractmethod
    def floor(self) -> VectorOps[int]: ...
