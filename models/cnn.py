import tensorflow as tf
from .base import BaseModel


class SequentialCNNModel(BaseModel):
    def __init__(self, args, vocab_size):
        super(SequentialCNNModel, self).__init__(args)
        self.vocab_size = vocab_size

    def call(self, input_tensor):
        filter_size = 4
        embedding = tf.keras.layers.Embedding(self.vocab_size, 128)
        input_tensor = embedding(input_tensor)

        first_layer = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number, kernel_size=4, activation='relu',
                                             padding='valid', strides=1)(input_tensor)
        second_layer = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 2, kernel_size=6, activation='relu',
                                              padding='valid', strides=1)(first_layer)
        third_layer = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3, kernel_size=8, activation='relu',
                                             padding='valid', strides=1)(second_layer)
        third_layer = tf.keras.layers.GlobalMaxPool1D()(third_layer)

        # output = tf.keras.layers.Flatten()(third_layer)
        # output = tf.keras.layers.Dense(args.cnn_hidden_layer)(output)

        return third_layer


class NMRModel(BaseModel):
    def __init__(self, args):
        super(NMRModel, self).__init__(args)
        self.conv_1 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number,
                                             kernel_size=10, activation='relu', padding='same', strides=1)
        self.pool_1 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number,
                                             kernel_size=10, activation='relu', padding='same', strides=4)
        self.conv_2 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 2,
                                             kernel_size=4, activation='relu', padding='same', strides=1)
        self.pool_2 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 2,
                                             kernel_size=4, activation='relu', padding='same', strides=2)
        self.conv_3 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                             kernel_size=4, activation='relu', padding='same', strides=1)
        self.pool_3 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                             kernel_size=4, activation='relu', padding='same', strides=2)
        self.conv_4 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                             kernel_size=4, activation='relu', padding='same', strides=1)
        self.conv_5 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                             kernel_size=4, activation='relu', padding='same', strides=2)

        self.global_pooling = tf.keras.layers.GlobalMaxPool1D()

        self.large_conv_1 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                                   kernel_size=4, activation='relu', padding='same', strides=1)
        self.large_conv_2 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                                   kernel_size=4, activation='relu', padding='same', strides=4)
        self.large_conv_3 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                                   kernel_size=1, activation='relu', padding='same', strides=1)
        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.args.cnn_hidden_layer)
        self.concat = tf.keras.layers.Concatenate(axis=1)

    def call(self, input_tensor):
        filter_size = 4
        output = tf.keras.backend.expand_dims(input_tensor, axis=-1)

        output = self.conv_1(output)
        output = self.pool_1(output)

        output = self.conv_2(output)
        output = self.pool_2(output)

        output = self.conv_3(output)
        output = self.pool_3(output)

        output = self.conv_4(output)
        output = self.conv_5(output)

        small_structure = self.global_pooling(output)

        output = self.large_conv_1(output)
        output = self.large_conv_2(output)
        output = self.large_conv_3(output)
        large_structure = self.flatten(output)
        large_structure = self.dense(large_structure)

        output = self.concat([small_structure, large_structure])

        return output
