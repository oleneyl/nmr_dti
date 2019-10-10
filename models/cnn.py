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

        first_layer = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number, kernel_size=4, activation='relu', padding='valid', strides=1)(input_tensor)
        second_layer = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number*2, kernel_size=6, activation='relu', padding='valid', strides=1)(first_layer)
        third_layer = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number*3, kernel_size=8, activation='relu', padding='valid', strides=1)(second_layer)
        third_layer = tf.keras.layers.GlobalMaxPool1D()(third_layer)

        #output = tf.keras.layers.Flatten()(third_layer)
        #output = tf.keras.layers.Dense(args.cnn_hidden_layer)(output)

        return third_layer

class NMRModel(BaseModel):
    def __init__(self, args, input_tensor, is_train=True):
        super(NMRModel, self).__init__(args)

        filter_size = args.cnn_filter_size

        input_tensor_reshaped = tf.keras.backend.expand_dims(input_tensor, axis=-1)

        first_layer = tf.keras.layers.Conv1D(args.cnn_filter_number, filter_size, activation='relu')(input_tensor_reshaped)
        first_layer = tf.keras.layers.MaxPool1D(pool_size=2)(first_layer)

        second_layer = tf.keras.layers.Conv1D(args.cnn_filter_number, filter_size, activation='relu')(first_layer)
        second_layer = tf.keras.layers.MaxPool1D(pool_size=2)(second_layer)

        third_layer = tf.keras.layers.Conv1D(args.cnn_filter_number, filter_size, activation='relu')(second_layer)
        third_layer = tf.keras.layers.MaxPool1D(pool_size=2)(third_layer)

        output = tf.keras.layers.Flatten()(third_layer)
        output = tf.keras.layers.Dense(args.cnn_hidden_layer)(output)

        self.output = output


