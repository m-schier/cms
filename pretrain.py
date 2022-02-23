def iterate_sdae(input_shape, layers, rate=0.2):
    from tensorflow.keras.layers import Dropout, MaxPooling2D, Flatten, ZeroPadding2D, Cropping2D
    import tensorflow as tf

    layer_count = len(layers)

    if layer_count % 2 != 0:
        raise ValueError("Total layer count must be divisible by 2")

    for i in range(layer_count // 2):
        # If current layer not trainable, skip iteration
        if issubclass(type(layers[i]), (MaxPooling2D, Flatten, ZeroPadding2D, Cropping2D)):
            continue

        inp = tf.keras.Input(input_shape, name="iterate_sdae_input_{}".format(i))
        x = inp

        # Append all previously trained encoder layers as untrainable
        for j in range(i):
            x = layers[j](x)
            layers[j].trainable = False

        # Apply noise to the current clean encoding
        x_noised = Dropout(rate=rate)(x)

        # Get the final encoder layer that will be trained this step
        h = layers[i](x_noised)
        layers[i].trainable = True

        # Apply noise to current embedding
        h_noised = Dropout(rate=rate)(h)

        # Get the first decoder layer that will be trained this step
        y = layers[layer_count - i - 1](h_noised)
        layers[layer_count - i - 1].trainable = True

        # Append all previously trained decoders as untrainable
        for j in range(layer_count - i, layer_count):
            y = layers[j](y)
            layers[j].trainable = False

        # Construct the current model and call back
        yield tf.keras.models.Model([inp], [y])


class AutoEncoderBuilder:
    def __init__(self, input_shape, layers):
        self._input_shape = input_shape
        self._layers = layers
        self._encoder = None
        self._decoder = None

    def iterate_for_sdae(self):
        for x in iterate_sdae(self._input_shape, self._layers):
            yield x

    def get_encoder(self):
        assert self._encoder is not None
        return self._encoder

    def get_decoder(self):
        assert self._decoder is not None
        return self._decoder

    def build_model(self):
        import tensorflow as tf

        encoder_input = tf.keras.Input(self._input_shape, name='encoder_input')
        encoder = encoder_input
        for l in self._layers[:len(self._layers) // 2]:
            l.trainable = True
            encoder = l(encoder)

        # Why is shape[1:] correct?!
        decoder_input = tf.keras.Input(encoder.shape[1:], name='decoder_input')
        decoder = decoder_input
        for l in self._layers[len(self._layers) // 2:]:
            l.trainable = True
            decoder = l(decoder)

        self._encoder = tf.keras.Model(encoder_input, encoder, name='encoder')
        self._decoder = tf.keras.Model(decoder_input, decoder, name='decoder')

        embeddings = self._encoder(encoder_input)
        reconstruction = self._decoder(embeddings)
        return tf.keras.models.Model(encoder_input, [embeddings, reconstruction], name='ae')


def make_dkmeans_mnist(input_shape, n_embeddings=10):
    # Auto encoder from Deep k-Means: Jointly clustering with k-Means and learning representations
    # d-500-500-2000-K
    import tensorflow as tf
    from tensorflow.keras.layers import Dense, Flatten, Reshape
    from tensorflow import random_normal_initializer
    from operator import mul
    from functools import reduce

    requires_flatten = len(input_shape) != 1
    flat_length = reduce(mul, input_shape)

    kernel_init = random_normal_initializer(stddev=.01)
    bias_init = random_normal_initializer(stddev=.01)

    layers = [
        Dense(500, activation='relu', name='enc_1', kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(500, activation='relu', name='enc_2', kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(2000, activation='relu', name='enc_3', kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(n_embeddings, name='enc_latent', kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(2000, activation='relu', name='dec_latent', kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(500, activation='relu', name='dec_3', kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(500, activation='relu', name='dec_2', kernel_initializer=kernel_init, bias_initializer=bias_init),
        Dense(flat_length, name='dec_1', activation=None, kernel_initializer=kernel_init, bias_initializer=bias_init),
    ]

    if requires_flatten:
        layers.insert(0, Flatten(name='enc_flatten'))
        layers.append(Reshape(input_shape, name='dec_reshape'))

    return AutoEncoderBuilder(input_shape, layers)


def pretrain_sdae(model_builder, seq, epochs_per_architecture=15, fine_tune_epochs=30, sgd=False):
    from tqdm import tqdm
    import tensorflow as tf
    import numpy as np

    pretrain_loss = tf.keras.metrics.Mean(name='pretrain_loss')

    # Used to visualize training progress
    x_sample, y_sample = seq[0]
    x_sample, y_sample = x_sample[:20], y_sample[:20]

    def train(current_model, start_epoch, stop_epoch):
        if not sgd:
            optimizer = tf.optimizers.Adam()
        else:
            # DEC suggest lr=.1 initially but that does not work
            optimizer = tf.optimizers.SGD(learning_rate=tf.optimizers.schedules.ExponentialDecay(.1 / 256, 20000, .1))

        for epoch in range(start_epoch, stop_epoch):
            pretrain_loss.reset_states()

            for batch_idx in tqdm(range(len(seq)), desc="Pretrain Epoch #{}".format(epoch)):
                x, _y = seq[batch_idx]

                with tf.GradientTape() as tape:
                    reconstruction = current_model(x, training=True)

                    # On final model, may return multiple outputs, by definition 1-index is reconstruction
                    if type(reconstruction) == list:
                        _, reconstruction = reconstruction

                    reconstruction_loss = tf.keras.losses.MSE(x, reconstruction)

                gradients = tape.gradient(reconstruction_loss, current_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, current_model.trainable_variables))
                pretrain_loss(reconstruction_loss)

                if batch_idx == 0:
                    reconstruction = current_model(x_sample, training=False)

                    # On final model, may return multiple outputs, by definition 1-index is reconstruction
                    if type(reconstruction) == list:
                        reconstruction = reconstruction[1]

                    if len(x_sample.shape) >= 3:
                        x_show = np.concatenate([x_sample, reconstruction], axis=2)
                        tf.summary.image("Inputs/Reconstructions", x_show, step=epoch, max_outputs=len(x_show))

            tf.summary.scalar('pretrain_loss', pretrain_loss.result(), step=epoch)
            print("Loss: {}".format(pretrain_loss.result()))
        return optimizer

    total_epochs = 0
    for model in model_builder.iterate_for_sdae():
        model.summary()
        train(model, total_epochs, total_epochs + epochs_per_architecture)
        total_epochs += epochs_per_architecture

    final_model = model_builder.build_model()
    return final_model, train(final_model, total_epochs, total_epochs + fine_tune_epochs)


def iterate_batch(batch_size, length):
    batch_count = (length + batch_size - 1) // batch_size

    for i in range(batch_count):
        yield batch_size * i, min(length, batch_size * (i + 1))


def encode(encoder, x, batch_size=256):
    import tensorflow as tf

    z = []

    for low, high in iterate_batch(batch_size, len(x)):
        z.append(encoder(x[low:high], training=False))

    z = tf.concat(z, axis=0)
    # z = np.concatenate([t.numpy() for t in z], axis=0)
    assert len(z) == len(x)
    return z


def __normalize_keras_dataset(train, test):
    import numpy as np

    (x_train, y_train), (x_test, y_test) = train, test

    # Only required for cifar
    y_train = y_train.flatten().astype(int)
    y_test = y_test.flatten().astype(int)

    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)

    x_train = x_train.astype(np.float32) / 255
    x_test = x_test.astype(np.float32) / 255

    if len(x_train.shape) < 4:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

    return (x_train, y_train), (x_test, y_test)


def load_mnist_normalized():
    from tensorflow.keras.datasets import mnist

    return __normalize_keras_dataset(*mnist.load_data())


def load_fashion_mnist_normalized():
    from tensorflow.keras.datasets import fashion_mnist

    return __normalize_keras_dataset(*fashion_mnist.load_data())


def make_mnist_dataset():
    from Util.Datasets import Dataset

    train, test = load_mnist_normalized()
    labels = [str(i) for i in range(10)]

    return Dataset(train, test, gt_labels=labels)


def make_fashion_mnist_dataset():
    from Util.Datasets import Dataset

    train, test = load_fashion_mnist_normalized()

    return Dataset(train, test, gt_labels=["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt",
                                           "Sneaker", "Bag", "Ankle boot"])


def main():
    from argparse import ArgumentParser
    from Util.SamplingSequence import SamplingSequence
    from Util.GTSRB import make_gtsrb
    from math import ceil
    import numpy as np
    import h5py
    import os
    import datetime

    datasets = {
        'mnist': make_mnist_dataset,
        'fashion-mnist': make_fashion_mnist_dataset,
        'gtsrb': lambda: make_gtsrb(size=(32, 32), normalize=True).filtered(labels=['Limit_50_kph', 'Limit_120_kph',
                                       'Attention_Road_Works', 'Attention_Traffic_Light',
                                       'Roundabout', 'Continue_Right_Of_Sign',
                                       'Stop', 'Yield', 'Right_Of_Way_Street',
                                       'No_Entry']),
    }

    parser = ArgumentParser()
    parser.add_argument('--data', choices=datasets.keys(), default='mnist')
    parser.add_argument('--sgd', action='store_true')
    parser.add_argument('--scale', type=float, default=1)
    parser.add_argument('--embeddings', type=int, default=10)
    parser.set_defaults(sgd=False)
    args = parser.parse_args()

    log_folder = os.path.join('tb_logs', args.data, datetime.datetime.now().isoformat())
    summary_writer = tf.summary.create_file_writer(log_folder)

    dataset = datasets[args.data]()
    train_x, train_y = dataset.train
    builder = make_dkmeans_mnist(train_x.shape[1:], n_embeddings=args.embeddings)
    seq = SamplingSequence(256, train_x, train_y)

    epa = ceil(15 * args.scale)
    fte = ceil(30 * args.scale)

    with summary_writer.as_default():
        model, optimizer = pretrain_sdae(builder, seq, epochs_per_architecture=epa, fine_tune_epochs=fte, sgd=args.sgd)

    model.save_weights(os.path.join(log_folder, "model_weights.hdf5"))

    embeddings = encode(builder.get_encoder(), train_x).numpy()

    with h5py.File(os.path.join(log_folder, 'embeddings.hdf5'), "w") as h5f:
        h5f.create_dataset(name='embeddings', data=embeddings)
        h5f.create_dataset(name='y_true', data=train_y)
        h5f.create_dataset(name='labels', data=np.array(dataset.gt_labels, dtype='S'))


if __name__ == '__main__':
    import tensorflow as tf
    import sys

    print("Using tensorflow {}".format(tf.__version__), file=sys.stderr)

    # Allow sharing the GPU with other processes if using GPU
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    main()
