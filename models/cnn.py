import tensorflow as tf
from .base import BaseModel


class CNNComponent():
    def __init__(self, filter, size):
        layer = tf.keras.layers.Conv1D(filter, size, activation='relu')


class NMRModel(BaseModel):
    def init_model(self, args):
        filter_size = args.cnn_filter_size
        sequential = tf.keras.Sequential()
        sequential.add(tf.keras.layers.Conv1D(filter_size, args.cnn_filter_number, activation='relu'))
        sequential.add(tf.keras.layers.BatchNormalization())
        sequential.add(tf.keras.layers.MaxPool1D(pool_size=2))

        sequential.add(tf.keras.layers.Conv1D(filter_size, args.cnn_filter_number, activation='relu'))
        sequential.add(tf.keras.layers.BatchNormalization())
        sequential.add(tf.keras.layers.MaxPool1D(pool_size=2))

        sequential.add(tf.keras.layers.Conv1D(filter_size, args.cnn_filter_number, activation='relu'))
        sequential.add(tf.keras.layers.BatchNormalization())
        sequential.add(tf.keras.layers.MaxPool1D(pool_size=2))

        sequential.add(tf.keras.layers.Flatten())
        sequential.add(tf.keras.layers.Dense(args.cnn_hidden_layer))

        self.sequential = sequential

    def compile(self, input_tensor):
        return self.sequential(tf.keras.backend.expand_dims(input_tensor, axis=-1))


