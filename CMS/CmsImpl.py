import numpy as np
from numba import jit
import sys


def l2_squared(left, right=None, cuda=True):
    if cuda:
        from ._CMS_ImplCuda import cuda_l2_squared
        return cuda_l2_squared(left, right)
    else:
        from ._CMS_ImplNumba import numba_l2_squared
        return numba_l2_squared(left, right)


@jit(nopython=True)
def _rbf_reduce_mul_impl(mode_mode_dist_sq, constraint_list, h_global, result, truncate, scale):
    for x, y in constraint_list:
        constraint_fit = np.sqrt(mode_mode_dist_sq[x, y]) * scale
        # Protect against underrunning a sensible fit if constraints are very close, 0 leads to arithmetic error later
        constraint_fit = max(constraint_fit, 0.001)
        constraint_fit = min(h_global, constraint_fit)

        x_ks = np.exp(-mode_mode_dist_sq[x] / (constraint_fit ** 2))
        y_ks = np.exp(-mode_mode_dist_sq[y] / (constraint_fit ** 2))
        x_ks[x_ks < truncate] = 0
        y_ks[y_ks < truncate] = 0

        buf = np.nonzero(y_ks)[0]

        for a in range(len(mode_mode_dist_sq)):
            if x_ks[a] == 0:
                continue
            for b in buf:
                result[a, b] = result[a, b] * (1 - x_ks[a] * y_ks[b])


@jit(nopython=True)
def _ball_reduce_mul_impl(mode_mode_dist_sq, constraint_list, h_global, result, scale):
    for x, y in constraint_list:
        constraint_fit = np.sqrt(mode_mode_dist_sq[x, y]) * scale
        # Protect against underrunning a sensible fit if constraints are very close, 0 leads to arithmetic error later
        constraint_fit = max(constraint_fit, 0.001)
        constraint_fit = min(h_global, constraint_fit)

        x_ks = mode_mode_dist_sq[x] <= constraint_fit ** 2
        y_ks = mode_mode_dist_sq[y] <= constraint_fit ** 2

        buf = np.nonzero(y_ks)[0]

        for a in range(len(mode_mode_dist_sq)):
            if x_ks[a] == 0:
                continue
            for b in buf:
                result[a, b] = result[a, b] * (1 - x_ks[a] * y_ks[b])


class CMS:
    def __init__(self, h, max_iterations=1000, blurring=True, kernel=.02, use_cuda=False, c_scale=.5, label_merge_k=.95,
                 label_merge_b=.1, stop_early=True, verbose=True, save_history=True):
        """
        Constrained Mean Shift Clustering

        :param h: If scalar, scalar bandwidth to be used for all iterations. If callable, function with signature
        f(int) -> float returning the scalar bandwidth to be used depending on the iteration number.
        :param max_iterations: Maximum number of iterations.
        :param blurring: If True use blurring mean shift, otherwise use non-blurring mean shift.
        :param kernel: If 'ball', use ball kernel. If scalar float in range [0, 1), use truncated Gaussian kernel
        with the value of kernel as truncation boundary.
        :param use_cuda: If True, use CUDA. Requires an installed CUDA-Toolkit.
        :param c_scale: Constraint scaling parameter. Determines influence of constraints.
        :param label_merge_k: Parameter for connectivity matrix for label extraction using connected components.
        Configures minimum closeness in terms of kernel for two modes to be marked as connected.
        :param label_merge_b: Parameter for connectivity matrix for label extraction using connected components.
        Configures worst constraint reduction below which two modes are never considered connected. Set to 0 to disable.
        :param stop_early: Whether to stop before reaching the maximum number of iterations if cluster centers become
        stationary. Works best with fixed bandwidth.
        :param verbose: If true is quite talkative.
        :param save_history: Whether to save history of modes, bandwidths, kernels and reduction weights. Disabling
        reduces memory consumption.
        """
        if kernel != 'ball' and not (0 < kernel < 1):
            raise ValueError("Invalid kernel: {}".format(kernel))

        if c_scale <= 0:
            raise ValueError("Invalid constraint scale: {}".format(c_scale))

        self.h = h
        self.max_iterations = max_iterations
        self.blurring = blurring
        self.kernel = kernel
        self.use_cuda = use_cuda
        self.c_scale = c_scale
        self.label_merge_k = label_merge_k
        self.label_merge_b = label_merge_b
        self.stop_early = stop_early
        self.verbose = verbose
        self.save_history = save_history

        self.mode_history_ = None
        self.block_history_ = None
        self.kernel_history_ = None
        self.bandwidth_history_ = None
        self.labels_ = None
        self.modes_ = None

    def calculate_labels(self, modes, curr_h, inter_weights):
        """
        Calculate labels for the given modes, bandwidth and reduction weights. Should usually be called with arguments
        of a matching history step of this CMS instance
        :param modes: Modes/cluster centers
        :param curr_h: Bandwidth used
        :param inter_weights: Reduction weights for the given modes and bandwidth
        :return: Label assignments
        """
        from scipy.sparse.csgraph import connected_components

        mode_mode_dist_sqs = l2_squared(modes, cuda=self.use_cuda)

        if self.kernel == 'ball':
            mode_mode_kernel_weights = (mode_mode_dist_sqs <= curr_h ** 2).astype(np.float32)
        else:
            mode_mode_kernel_weights = np.exp(-mode_mode_dist_sqs / (curr_h ** 2))
            mode_mode_kernel_weights[mode_mode_kernel_weights < self.kernel] = 0

        assert mode_mode_kernel_weights.shape == inter_weights.shape

        allow_merge = np.logical_and(mode_mode_kernel_weights > self.label_merge_k, inter_weights > self.label_merge_b)
        return connected_components(allow_merge)[1]

    def fit_transform(self, points, constraints):
        self.fit(points, constraints)
        return self.modes_

    def fit_predict(self, points, constraints):
        self.fit(points, constraints)
        return self.labels_

    def fit(self, points, constraints):
        """
        Fit on the given data/sampling points and cannot-link constraints
        :param points: An NxD array of points, which serves as both the initial sampling points and initial cluster
        centers
        :param constraints: A Cx2 array indexing sampling points that shall not be linked
        """
        from .Constraints import constraint_list_from_constraints

        constraint_list = constraint_list_from_constraints(constraints)

        points = points.astype(np.float32)
        modes = points

        if self.save_history:
            self.mode_history_ = [modes]
            self.block_history_ = []
            self.kernel_history_ = []
            self.bandwidth_history_ = []

        is_fixed_h = not callable(self.h)

        for epoch in range(self.max_iterations):
            curr_h = np.float32(self.h if is_fixed_h else self.h(epoch))
            if self.verbose:
                print('CMS: Iteration: {}, Bandwidth: {}'.format(epoch, curr_h), file=sys.stderr)

            assert not np.isnan(curr_h)

            if curr_h <= 0:
                raise ValueError("Require curr_h > 0, was {}".format(curr_h))

            mode_mode_dist_sqs = l2_squared(modes, cuda=self.use_cuda)

            if self.blurring:
                # The mean shift starts with modes at original points and samples kernels on the last modes
                sample_dist_sqs = mode_mode_dist_sqs
                sample_points = modes
            else:
                # Also known as basin mode, the mean shift always samples kernels from original data points
                sample_dist_sqs = l2_squared(modes, points, cuda=self.use_cuda)
                sample_points = points

            if self.kernel != 'ball':
                kernel_weights = np.exp(-sample_dist_sqs / (curr_h ** 2))
                kernel_weights[kernel_weights < self.kernel] = 0
            else:
                kernel_weights = (sample_dist_sqs <= curr_h ** 2).astype(np.float32)

            inter_weights = np.ones((len(modes),) * 2, dtype=np.float32)

            assert np.all(np.logical_and(constraint_list >= 0, constraint_list < len(modes)))

            if self.kernel == 'ball':
                _ball_reduce_mul_impl(mode_mode_dist_sqs, constraint_list, curr_h, inter_weights, self.c_scale)
            else:
                _rbf_reduce_mul_impl(mode_mode_dist_sqs, constraint_list, curr_h, inter_weights, self.kernel,
                                     self.c_scale)

            if self.blurring:
                # Only enforce inter-weight diagonal 1 for cluster mode
                np.fill_diagonal(inter_weights, 1)

                # Only check symetry in cluster sample mode, in other mode cluster centers may drift unsymetrically
                assert not np.isnan(inter_weights).any()

            assert np.logical_and(inter_weights >= 0, inter_weights <= 1).all()

            eff_weights = kernel_weights * inter_weights

            weights_sum = np.sum(eff_weights, axis=-1, keepdims=True)

            # If something completely loses attraction, reset to self
            weightless = (weights_sum <= 1e-10).nonzero()[0]
            if len(weightless) > 0:
                print("CMS: Warn: Iteration {}: Had {} cluster centers without weight".format(epoch, len(weightless)),
                      file=sys.stderr)

            modes_new = np.dot(eff_weights, sample_points) / weights_sum
            modes_new[weightless] = modes[weightless]

            break_next = False
            if self.stop_early:
                if np.allclose(modes_new, modes, rtol=1.e-4, atol=1.e-6):
                    if np.all(np.logical_or(inter_weights < self.label_merge_b, kernel_weights > self.label_merge_k)):
                        # Don't immediately break here, but add to history and then break. This is because we have already
                        # invested the time into the calculation anyways so we might actually return the first "over-settled"
                        # modes instead of the ones before. Because modes may have still slightly changed on the iteration
                        # where we detect settling, this can actually improve cluster assignment on some edge cases
                        break_next = True
                    elif is_fixed_h:
                        # If h is fixed, can stop immediately when no movement occurs, because next iteration will have
                        # same result
                        break_next = True

            if self.save_history:
                self.mode_history_.append(modes_new)
                self.block_history_.append(inter_weights)
                self.kernel_history_.append(kernel_weights)
                self.bandwidth_history_.append(curr_h)

            modes = modes_new

            if break_next:
                break

        self.modes_ = modes
        self.labels_ = self.calculate_labels(modes, curr_h, inter_weights)
