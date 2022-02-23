import numpy as np
import sys


class AutoLinearPolicy:
    def __init__(self, points, iterations, start_scale=None, stop_scale=None, verbose=False):
        from ._CMS_ImplNumba import _find_min_max_l2_impl

        assert type(iterations) == int
        assert len(points.shape) == 2

        if len(points) < 2:
            raise ValueError

        # Used to default to 1 / sqrt(-ln(1/4)) ~= 0.85
        start_scale = start_scale if start_scale is not None else 1.
        stop_scale = stop_scale if stop_scale is not None else start_scale

        # Sample points if too many would slow us down
        if len(points) > 1000:
            points = points[np.random.choice(np.arange(len(points)), size=1000, replace=False)]

        assert 0 < start_scale
        assert 0 < stop_scale

        l2_min, l2_max = _find_min_max_l2_impl(points.astype(np.float))

        low = np.sqrt(l2_min) * start_scale
        high = np.sqrt(l2_max) * stop_scale

        if verbose:
            print("Minimum and maximum squared L2s were: {}, {}".format(l2_min, l2_max), file=sys.stderr)
            print("Using auto linear window bounds {} -> {}".format(low, high), file=sys.stderr)

        self.iterations = iterations
        self.low = low
        self.high = high

    def __call__(self, iteration):
        return (iteration / (self.iterations - 1)) * (self.high - self.low) + self.low
