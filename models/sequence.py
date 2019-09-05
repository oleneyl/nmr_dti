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
    def __init__(self, args, input_tensor, vocab_size, is_train=True):
        super(RNNProteinModel, self).__init__(args)
        embedding = tf.keras.layers.Embedding(vocab_size, args.transformer_model_dim)(input_tensor)
        embedding = tf.keras.layers.Dropout(rate=args.concat_dropout)(embedding, training=is_train)
        rnn_cell = tf.keras.layers.GRUCell(args.sequential_hidden_size, dropout=args.sequential_dropout)
        recurrent = tf.keras.layers.RNN(rnn_cell)
        bidirectional_recurrent = tf.keras.layers.Bidirectional(recurrent)
        output = bidirectional_recurrent(embedding, training=is_train)
        output = tf.keras.layers.Dense(args.sequential_dense)(output)
        self.output = output


class AttentionProteinModel(BaseModel):
    def __init__(self, args, input_tensor, vocab_size, is_train=True):
        self.encoder = Encoder(args.transformer_num_layers,
                               args.transformer_model_dim,
                               args.transformer_num_heads,
                               args.transformer_hidden_dimension,
                               vocab_size,
                               rate=args.transformer_dropout_rate)

        output = self.encoder(input_tensor, is_train, None)
        output = tf.keras.layers.Flatten()(output)
        '''
        output = tf.keras.layers.Dense(1, activation='relu')(output)
        output = tf.keras.layers.BatchNormalization()(output, training=is_train)
        output = tf.keras.layers.Dropout(rate=args.transformer_dropout_rate)(output, training=is_train)
        output = tf.squeeze(output, axis=-1)
        '''
        self.output = tf.keras.layers.Dense(args.sequential_dense, activation='relu')(output)
