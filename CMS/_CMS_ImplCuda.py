from numba import cuda
import math
import numpy as np


@cuda.jit('void(float32[:, :], float32[:, :], float32[:, :])')
def __mode_l2_sqs_cuda_impl(left, right, result):
    i, j = cuda.grid(2)

    if i < len(left) and j < len(right):
        l2_sq = 0
        for d in range(left.shape[-1]):
            l2_sq += (left[i, d] - right[j, d]) ** 2
        result[i, j] = l2_sq


def cuda_l2_squared(left, right=None, tbp=8):
    """
    Calculate all squared L2 norms between the input vectors
    :param left: Input vectors of shape (n_left, n_dimensions)
    :param right: Input vectors of shape, defaults to left if None (n_right, n_dimensions)
    :param tbp: Threads per block
    :return: Norms between vectors of shape (n_left, n_right)
    """

    # TODO: If one felt like it, this would probably be much simpler in tensorflow/torch and equally fast
    assert len(left.shape) == 2

    if right is not None:
        assert len(right.shape) == 2
        assert left.shape[1] == right.shape[1]
        right_len = len(right)
    else:
        right_len = len(left)

    stream = cuda.stream()

    result_dev = cuda.device_array(shape=(len(left), right_len), dtype=np.float32, stream=stream)
    left_dev = cuda.to_device(left.astype(np.float32), stream=stream)

    if right is not None:
        right_dev = cuda.to_device(right, stream=stream)
    else:
        right_dev = None

    result = np.empty((len(left), right_len), dtype=np.float32)
    threadsperblock = (tbp, tbp)
    blockspergrid_x = math.ceil(len(left) / threadsperblock[0])
    blockspergrid_y = math.ceil(right_len / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    if right_dev is not None:
        __mode_l2_sqs_cuda_impl[blockspergrid, threadsperblock, stream](left_dev, right_dev, result_dev)
    else:
        __mode_l2_sqs_cuda_impl[blockspergrid, threadsperblock, stream](left_dev, left_dev, result_dev)

    result_dev.copy_to_host(result, stream=stream)
    stream.synchronize()
    return result
