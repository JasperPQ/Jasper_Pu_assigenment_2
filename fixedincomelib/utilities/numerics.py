import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List


class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class ExtrapMethod(Enum):

    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value


class Interpolator1D(ABC):
    """Abstract interface for a 1-D interpolator."""

    def __init__(self,
                 axis1: np.ndarray,
                 values: np.ndarray,
                 interpolation_method: InterpMethod,
                 extrapolation_method: ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrapolation_method
        self.length_ = len(self.axis1_)

    @abstractmethod
    def interpolate(self, x: float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x: float, end_x: float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        pass

    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def values(self) -> np.ndarray:
        return self.values_

    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()

    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator1DPCP(Interpolator1D):
    """Piecewise-constant left-continuous interpolator with FLAT extrapolation.

    With axis1 = [1, 3, 5, 7], values = [3, 4, 5, 6]:
        f(0.5) = 3, f(1) = 3, f(1.5) = 4, f(3) = 4, f(5.5) = 6, f(8) = 6
    """

    def __init__(self, axis1: np.ndarray, values: np.ndarray,
                 extrapolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values,
                         InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS,
                         extrapolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        index = int(np.searchsorted(self.axis1_, x, side='left'))
        index = min(index, self.length_ - 1)
        return float(self.values_[index])

    def integrate(self, start_x: float, end_x: float) -> float:
        if start_x == end_x:
            return 0.0

        lower = min(start_x, end_x)
        upper = max(start_x, end_x)
        weights = np.zeros(self.length_, dtype=float)

        if self.length_ == 1:
            weights[0] = upper - lower
        else:
            # The value at each knot is the value on the interval ending at
            # that knot; the first and last values extend flat beyond the grid.
            weights[0] = max(0.0, min(upper, self.axis1_[0]) - lower)
            for index in range(1, self.length_ - 1):
                weights[index] = max(
                    0.0,
                    min(upper, self.axis1_[index])
                    - max(lower, self.axis1_[index - 1]),
                )
            weights[-1] = max(
                0.0, upper - max(lower, self.axis1_[-2])
            )

        integral = float(np.dot(self.values_, weights))
        return integral if end_x > start_x else -integral

    def gradient_wrt_ordinate(self, x: float) -> np.ndarray:
        gradient = np.zeros(self.length_, dtype=float)
        index = int(np.searchsorted(self.axis1_, x, side='left'))
        gradient[min(index, self.length_ - 1)] = 1.0
        return gradient

    def gradient_of_integrated_value_wrt_ordinate(self, start_x: float, end_x: float) -> np.ndarray:
        if start_x == end_x:
            return np.zeros(self.length_, dtype=float)

        lower = min(start_x, end_x)
        upper = max(start_x, end_x)
        gradient = np.zeros(self.length_, dtype=float)

        if self.length_ == 1:
            gradient[0] = upper - lower
        else:
            gradient[0] = max(0.0, min(upper, self.axis1_[0]) - lower)
            for index in range(1, self.length_ - 1):
                gradient[index] = max(
                    0.0,
                    min(upper, self.axis1_[index])
                    - max(lower, self.axis1_[index - 1]),
                )
            gradient[-1] = max(
                0.0, upper - max(lower, self.axis1_[-2])
            )

        return gradient if end_x > start_x else -gradient


class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1: np.ndarray | List,
                               values: np.ndarray | List,
                               interpolation_method: InterpMethod,
                               extrapolation_method: ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)

        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrapolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')