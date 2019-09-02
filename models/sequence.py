from .base import BaseModel
import tensorflow as tf


class RNNProteinModel(BaseModel):
    def __init__(self, args, input_tensor, is_train=True):
        super(RNNProteinModel, self).__init__(args)
        rnn_cell = tf.keras.layers.GRUCell(args.sequential_hidden_size, activation='relu',
                                           dropout=args.sequential_dropout)

        output = tf.keras.layers.RNN(rnn_cell)(input_tensor, training=is_train)
        output = tf.keras.layers.Dense(args.sequential_dense)(output)
        self.output = output
