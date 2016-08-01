import numpy as np
import tensorflow as tf
import itertools
import time
from scipy import misc
from Kroma import Kroma

flags = tf.flags
flags.DEFINE_string('marked_path', 'marked.bmp', 'path to marked image')
flags.DEFINE_string('original_path', 'original.bmp', 'path to original image (B/W or Color')
flags.DEFINE_string('colored_path', 'colored.bmp', 'path to save the colored image')
FLAGS = tf.flags.FLAGS


def _rgb_to_yuv(rgb):
    return np.dot(rgb, np.array([
        [0.299, -0.14713, 0.615],
        [0.587, -0.28886, -0.51499],
        [0.114, 0.436, -0.10001]
    ], dtype=np.float32))


def _yuv_to_rgb(yuv):
    return np.dot(yuv, np.array([
        [1, 1, 1],
        [0, -0.39465, 2.03211],
        [1.13983, -0.58060, 0]
    ], dtype=np.float32))


def _get_weights(gray, is_marked):
    num_rows, num_cols = gray.shape
    delta = 1
    num_pixels = num_rows * num_cols
    nnz = num_pixels * (2 * delta + 1) ** 2

    indices = np.zeros((nnz, 2), dtype=np.int64)
    values = np.zeros(nnz)
    index = x = 0
    for i, j in itertools.product(range(num_rows), range(num_cols)):
        if not is_marked[i, j]:
            row_start = max(i - delta, 0)
            row_end = min(i + delta + 1, num_rows)
            col_start = max(j - delta, 0)
            col_end = min(j + delta + 1, num_cols)
            neighborhood = gray[row_start:row_end, col_start:col_end]
            var = max(np.var(neighborhood), 2e-6)
            centre_val = gray[i, j]
            current_weights = np.exp(- (neighborhood - centre_val) ** 2 / var)
            current_weights /= np.sum(current_weights) - 1
            current_weights[i - row_start, j - col_start] = 0  # -1
            nz_this_row = [[x, k * num_cols + l] for k, l in
                           itertools.product(range(row_start, row_end), range(col_start, col_end))]
            nnz_this_row = len(nz_this_row)
            indices[index:index + nnz_this_row, :] = nz_this_row
            values[index:index + nnz_this_row] = -current_weights.reshape(-1)
            index += nnz_this_row
        else:
            # indices[index] = [x, x]
            # values[index] = 1
            # index += 1
            pass
        x += 1
    indices = indices[:index, :]
    values = values[:index]
    return indices, values


class Config:
    data_type = tf.float32
    num_iterations = 50000


def main(_):
    t = time.time()

    config = Config()

    marked = misc.imread(FLAGS.marked_path)
    original = misc.imread(FLAGS.original_path)

    marked = marked.astype(np.float32) / 255
    original = original.astype(np.float32) / 255

    n_rows, n_cols, _ = marked.shape
    n_pixels = n_rows * n_cols
    shape = [n_pixels] * 2

    yuv_marked = _rgb_to_yuv(marked)
    yuv_original = _rgb_to_yuv(original)

    gray_original = yuv_original[:, :, 0]
    mask = np.array((yuv_marked[:, :, 1] ** 2 + yuv_marked[:, :, 2] ** 2 > 1e-8), dtype=np.bool)
    weights_indices, weights_values = _get_weights(gray_original, mask)

    print('Computed weights. Time taken: %.3fs' % (time.time() - t))

    colored = np.zeros((n_rows, n_cols, 3), dtype=np.float32)

    with tf.Graph().as_default(), tf.Session() as session:
        tf.initialize_all_variables().run()
        kroma = Kroma(config)
        for c in range(1, 3):
            b = yuv_marked[:, :, c].reshape((n_pixels, 1))
            final_x = session.run(kroma.final_x, feed_dict={
                kroma.weights_indices: weights_indices,
                kroma.weights_values: weights_values,
                kroma.weights_shape: shape,
                kroma.b: b,
                kroma.initial_x: b
            })
            colored[:, :, c] = final_x.reshape((n_rows, n_cols))

    colored[:, :, 0] = yuv_original[:, :, 0]
    colored = _yuv_to_rgb(colored)
    colored[colored < 0] = 0
    colored[colored > 1] = 1
    colored = (colored * 255).astype(np.uint8)

    print('Computed color image. Time taken: %.3fs' % (time.time() - t))
    misc.imsave(FLAGS.colored_path, colored, format='bmp')

if __name__ == '__main__':
    tf.app.run()
