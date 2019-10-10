from .base import BaseModel
import tensorflow as tf
from .attention import Encoder, Decoder, create_padding_mask

"""
ProteinModel :: get Input as indices_array, return Tensor with shape [None, args.sequential_dense]
"""


def transformer_args(parser):
    group = parser.add_argument_group('transformer')

    group.add_argument('--transformer_num_layers', type=int, default=6)
    group.add_argument('--transformer_model_dim', type=int, default=256)
    group.add_argument('--transformer_num_heads', type=int, default=8)
    group.add_argument('--transformer_hidden_dimension', type=int, default=256)
    group.add_argument('--transformer_dropout_rate', type=float, default=0.1)


class RNNProteinModel(BaseModel):
    def __init__(self, args, vocab_size):
        super(RNNProteinModel, self).__init__(args)
        self.embedding = tf.keras.layers.Embedding(vocab_size, args.transformer_model_dim)
        rnn_cell = tf.keras.layers.GRUCell(args.sequential_hidden_size, dropout=args.sequential_dropout)
        recurrent = tf.keras.layers.RNN(rnn_cell)
        self.bidirectional_recurrent = tf.keras.layers.Bidirectional(recurrent)

        self.dense = tf.keras.layers.Dense(args.sequential_dense)

    def call(self, input_tensor, training=None):
        output = self.embedding(input_tensor)
        output = self.bidirectional_recurrent(output, training=training)
        output = self.dense(output)
        return output

class AttentionProteinModel(BaseModel):
    def __init__(self, args, vocab_size):
        super(AttentionProteinModel, self).__init__(args)
        self.vocab_size = vocab_size

        self.encoder = Encoder(self.args.transformer_num_layers,
                         self.args.transformer_model_dim,
                         self.args.transformer_num_heads,
                         self.args.transformer_hidden_dimension,
                         self.vocab_size,
                         rate=self.args.transformer_dropout_rate)

        self.flatten = tf.keras.layers.Flatten()
        self.dense = tf.keras.layers.Dense(self.args.sequential_dense, activation='relu')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(self.args.concat_dropout)

    def call(self, input_tensor, training=None):
        output = self.encoder(input_tensor, mask=None, training=training)
        output = self.flatten(output)
        output = self.dense(output)
        output = self.batch_norm(output, training=training)
        output = self.dropout(output, training=training)
        return output
