import tensorflow as tf
from .base import BaseModel


class CNNComponent():
    def __init__(self, filter, size):
        layer = tf.keras.layers.Conv1D(filter, size, activation='relu')


class NMRModel(BaseModel):
    def __init__(self, args, input_tensor, is_train=True):
        super(NMRModel, self).__init__(args)

        filter_size = args.cnn_filter_size

        input_tensor_reshaped = tf.keras.backend.expand_dims(input_tensor, axis=-1)

        first_layer = tf.keras.layers.Conv1D(filter_size, args.cnn_filter_number, activation='relu')(input_tensor_reshaped)
        first_layer = tf.keras.layers.MaxPool1D(pool_size=2)(first_layer)

        second_layer = tf.keras.layers.Conv1D(filter_size, args.cnn_filter_number*2, activation='relu')(first_layer)
        second_layer = tf.keras.layers.MaxPool1D(pool_size=2)(second_layer)

        third_layer = tf.keras.layers.Conv1D(filter_size, args.cnn_filter_number*3, activation='relu')(second_layer)
        third_layer = tf.keras.layers.MaxPool1D(pool_size=2)(third_layer)

        output = tf.keras.layers.Flatten()(third_layer)
        output = tf.keras.layers.Dense(args.cnn_hidden_layer)(output)

        self.output = output


