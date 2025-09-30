"""

# References
- https://www.johndcook.com/blog/standard_deviation/
"""

import math


@fieldwise_init
struct RunningStats[dtype: DType]:
    """Computes running mean, variance, standard deviation, min, and max using Welford's algorithm.
    """

    var count: Int
    var prev_mean: Float64
    var current_mean: Float64
    var prev_s: Float64
    var current_s: Float64
    var min_value: Scalar[dtype]
    var max_value: Scalar[dtype]

    fn __init__(out self):
        self.count = 0
        self.prev_mean = 0.0
        self.current_mean = 0.0
        self.prev_s = 0.0
        self.current_s = 0.0
        self.min_value = Scalar[dtype].MAX
        self.max_value = Scalar[dtype].MIN

    fn clear(mut self):
        self.count = 0
        self.prev_mean = 0.0
        self.current_mean = 0.0
        self.prev_s = 0.0
        self.current_s = 0.0
        self.min_value = 0.0
        self.max_value = 0.0

    fn push(mut self, value: Scalar[dtype]):
        if self.count == 0:
            self.min_value = value
            self.max_value = value
            self.prev_mean = Float64(value)
            self.current_mean = Float64(value)
            self.prev_s = 0.0
        else:
            if value < self.min_value:
                self.min_value = value
            if value > self.max_value:
                self.max_value = value

            self.current_mean = self.prev_mean + (
                Float64(value) - self.prev_mean
            ) / (self.count + 1)
            self.current_s = self.prev_s + (Float64(value) - self.prev_mean) * (
                Float64(value) - self.current_mean
            )

            self.prev_mean = self.current_mean
            self.prev_s = self.current_s

        self.count += 1

    fn num_values(self) -> Int:
        return self.count

    fn mean(self) -> Float64:
        return self.current_mean if self.count > 0 else 0.0

    fn variance(self) -> Float64:
        return self.current_s / (self.count - 1) if self.count > 1 else 0.0

    fn standard_deviation(self) -> Float64:
        return math.sqrt(self.variance())

    fn min(self) -> Scalar[dtype]:
        return self.min_value if self.count > 0 else Scalar[dtype].MAX

    fn max(self) -> Scalar[dtype]:
        return self.max_value if self.count > 0 else Scalar[dtype].MIN
