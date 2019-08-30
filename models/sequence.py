from .base import BaseModel
import tensorflow as tf


class RNNProteinModel(BaseModel):
    def init_model(self, args):
        hidden_units = 100
        rnn_cell = tf.keras.layers.GRUCell(hidden_units, activation='relu', dropout=0.2)

        sequential = tf.keras.Sequential()
        sequential.add(tf.keras.layers.RNN(rnn_cell))
        sequential.add(tf.keras.layers.Dense(64))
        self.sequential = sequential

    def compile(self, input_tensor):
        return self.sequential(input_tensor)
