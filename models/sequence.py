from .base import BaseModel
import tensorflow as tf


class RNNProteinModel(BaseModel):
    def init_model(self, args):
        rnn_cell = tf.keras.layers.GRUCell(args.sequential_hidden_size, activation='relu', dropout=args.sequential_dropout)

        sequential = tf.keras.Sequential()
        sequential.add(tf.keras.layers.RNN(rnn_cell))
        sequential.add(tf.keras.layers.Dense(args.sequential_dense))
        self.sequential = sequential

    def compile(self, input_tensor):
        return self.sequential(input_tensor)
