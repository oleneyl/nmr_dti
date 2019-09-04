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
    def __init__(self, args, input_tensor, is_train=True):
        super(RNNProteinModel, self).__init__(args)
        embedding = tf.keras.layers.Embedding(args.protein_vocab_size, args.transformer_model_dim)
        rnn_cell = tf.keras.layers.GRUCell(args.sequential_hidden_size, activation='relu',
                                           dropout=args.sequential_dropout)

        output = tf.keras.layers.RNN(rnn_cell)(embedding(input_tensor), training=is_train)
        output = tf.keras.layers.Dense(args.sequential_dense)(output)
        self.output = output


class AttentionProteinModel(BaseModel):
    def __init__(self, args, input_tensor, is_train=True):
        self.encoder = Encoder(args.transformer_num_layers,
                               args.transformer_model_dim,
                               args.transformer_num_heads,
                               args.transformer_hidden_dimension,
                               args.protein_vocab_size,
                               rate=args.transformer_dropout_rate)

        output = self.encoder(input_tensor, is_train, None)
        output = tf.keras.layers.Dense(1, activation='relu')(output)
        output = tf.keras.layers.BatchNormalization()(output, training=is_train)
        output = tf.keras.layers.Dropout(rate=args.transformer_dropout_rate)(output, training=is_train)
        output = tf.squeeze(output, axis=-1)
        self.output = tf.keras.layers.Dense(args.sequential_dense)(output)
