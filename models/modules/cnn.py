import tensorflow as tf
from models.base import BaseModel
from models.sequence import AttentionProteinModel
from models.modules.attention import Encoder, VectorEncoder


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
        self.sequential_layers = [
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number,
                                   kernel_size=3, activation='relu', padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number,
                                   kernel_size=3, activation='relu', padding='same', strides=2),  # Pooling
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 2,
                                   kernel_size=3, activation='relu', padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 2,
                                   kernel_size=3, activation='relu', padding='same', strides=1),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 2,
                                   kernel_size=3, activation='relu', padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 2,
                                   kernel_size=3, activation='relu', padding='same', strides=2),  # Pooling
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                   kernel_size=3, activation='relu', padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                   kernel_size=3, activation='relu', padding='same', strides=1),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(rate=0.2),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                   kernel_size=3, activation='relu', padding='same', strides=2)]  # Pooling

        self.large_conv_1 = tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 3,
                                                   kernel_size=3, activation='relu', padding='same', strides=1)
        self.large_conv_2 = tf.keras.layers.Conv1D(filters=256,
                                                   kernel_size=1, padding='same', strides=2)
        self.attention = AttentionProteinModel(args, 0, vectorized=True)
        self.concat = tf.keras.layers.Concatenate(axis=1)

    def call(self, input_tensor, training=None):
        filter_size = 4
        output = tf.keras.backend.expand_dims(input_tensor, axis=-1)

        for layer in self.sequential_layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Dropout):
                output = layer(output, training=training)
            else:
                output = layer(output)

        output = self.large_conv_1(output)
        output = self.large_conv_2(output)
        output = self.attention(output, training=training)

        return output


class ResidualLegacy(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ResidualLegacy, self).__init__()
        self.normalize = tf.keras.layers.BatchNormalization()
        self.layer = tf.keras.layers.Conv1D(filters=filters, kernel_size=3, padding='same', strides=1)

    def call(self, input_tensor, training=None):
        output = self.layer(input_tensor)
        output = self.normalize(output, training=training)

        return input_tensor + output


class Residual(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(Residual, self).__init__()
        self.elu_1 = tf.keras.layers.ELU()
        self.normalize_1 = tf.keras.layers.BatchNormalization()
        self.layer_1 = tf.keras.layers.Conv1D(filters=filters, kernel_size=7, padding='same', strides=1)
        self.elu_2 = tf.keras.layers.ELU()
        self.normalize_2 = tf.keras.layers.BatchNormalization()
        self.layer_2 = tf.keras.layers.Conv1D(filters=filters, kernel_size=7, padding='same', strides=1)

    def call(self, input_tensor, training=None):
        output = self.elu_1(input_tensor)
        output = self.normalize_1(output, training=training)
        output = self.layer_1(output)
        output = self.elu_2(output)
        output = self.normalize_2(output, training=training)
        output = self.layer_2(output)

        return input_tensor + output


class ConvPool(tf.keras.layers.Layer):
    def __init__(self, filters):
        super(ConvPool, self).__init__()
        self.elu = tf.keras.layers.ELU()
        self.normalize = tf.keras.layers.BatchNormalization()
        self.conv_pool = tf.keras.layers.Conv1D(filters=filters, kernel_size=6, padding='same', strides=2)  # Pooling

    def call(self, input_tensor, training=None):
        output = self.elu(input_tensor)
        output = self.normalize(output, training=training)
        output = self.conv_pool(output)

        return output


class NMRModel_2(BaseModel):
    def __init__(self, args):
        super(NMRModel_2, self).__init__(args)
        self.sequential_layers = [
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number, kernel_size=3, activation='relu',
                                   padding='same', strides=1),
            Residual(self.args.cnn_filter_number),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 2,
                                   kernel_size=2, padding='same', strides=2),  # Pooling
            tf.keras.layers.BatchNormalization(),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 4,
                                   kernel_size=2, padding='same', strides=2),  # Pooling
            tf.keras.layers.BatchNormalization(),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 6,
                                   kernel_size=2, padding='same', strides=2),  # Pooling
            tf.keras.layers.BatchNormalization(),
            Residual(self.args.cnn_filter_number * 6),
            Residual(self.args.cnn_filter_number * 6),
            Residual(self.args.cnn_filter_number * 6),
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number * 8,
                                   kernel_size=2, padding='same', strides=2),  # Pooling
            tf.keras.layers.BatchNormalization(),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8)]  # Pooling

        self.large_conv = tf.keras.layers.Conv1D(filters=256,
                                                 kernel_size=1, padding='same', strides=2)
        self.attention = AttentionProteinModel(args, 0, vectorized=True)
        self.concat = tf.keras.layers.Concatenate(axis=1)

    def call(self, input_tensor, training=None):
        filter_size = 4
        output = tf.keras.backend.expand_dims(input_tensor, axis=-1)

        for layer in self.sequential_layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Dropout) or \
                    isinstance(layer, Residual):
                output = layer(output, training=training)
            else:
                output = layer(output)

        output = self.large_conv(output)
        output = self.attention(output, training=training)

        return output


class NMRModel_3(BaseModel):
    def __init__(self, args):
        super(NMRModel_3, self).__init__(args)
        self.sequential_layers = [
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number, kernel_size=8, padding='same', strides=1),
            Residual(self.args.cnn_filter_number),
            ConvPool(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            ConvPool(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            ConvPool(self.args.cnn_filter_number * 6),
            Residual(self.args.cnn_filter_number * 6),
            Residual(self.args.cnn_filter_number * 6),
            Residual(self.args.cnn_filter_number * 6),
            ConvPool(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8)]  # Pooling

        self.large_conv = tf.keras.layers.Conv1D(filters=256,
                                                 kernel_size=1, padding='same', strides=2, activation='elu')
        self.normalizer = tf.keras.layers.BatchNormalization()
        self.attention = AttentionProteinModel(args, 0, vectorized=True)
        self.concat = tf.keras.layers.Concatenate(axis=1)

    def call(self, input_tensor, training=None):
        filter_size = 4
        output = tf.keras.backend.expand_dims(input_tensor, axis=-1)

        for layer in self.sequential_layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Dropout) or \
                    isinstance(layer, Residual) or isinstance(layer, ConvPool):
                output = layer(output, training=training)
            else:
                output = layer(output)

        output = self.large_conv(output)
        output = self.normalizer(output, training=training)
        output = self.attention(output, training=training)

        return output


class NMR_Infuse(BaseModel):
    def __init__(self, args, vocab_size):
        super(NMR_Infuse, self).__init__(args)
        self.vocab_size = vocab_size

        self.encoder = Encoder(self.args.transformer_num_layers,
                               self.args.transformer_model_dim,
                               self.args.transformer_num_heads,
                               self.args.transformer_hidden_dimension,
                               self.vocab_size,
                               rate=self.args.transformer_dropout_rate)

        self.encoder_nmr = VectorEncoder(self.args.transformer_num_layers,
                                         self.args.transformer_model_dim,
                                         self.args.transformer_num_heads,
                                         self.args.transformer_hidden_dimension,
                                         rate=self.args.transformer_dropout_rate)

        self.sequential_layers = [
            tf.keras.layers.Conv1D(filters=self.args.cnn_filter_number, kernel_size=8, padding='same', strides=1),
            Residual(self.args.cnn_filter_number),
            Residual(self.args.cnn_filter_number),

            ConvPool(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),
            Residual(self.args.cnn_filter_number * 2),

            ConvPool(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),
            Residual(self.args.cnn_filter_number * 4),

            ConvPool(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8),
            Residual(self.args.cnn_filter_number * 8)]  # Pooling

        self.normalizer_1 = tf.keras.layers.BatchNormalization()
        self.large_conv = tf.keras.layers.Conv1D(filters=256,
                                                 kernel_size=1, padding='same', strides=2, activation='elu')

        self.decoder = VectorEncoder(3,
                                     self.args.transformer_model_dim,
                                     self.args.transformer_num_heads,
                                     self.args.transformer_hidden_dimension,
                                     rate=self.args.transformer_dropout_rate)

        self.dense = tf.keras.layers.Dense(self.args.sequential_dense, activation='relu')
        self.dropout = tf.keras.layers.Dropout(self.args.concat_dropout)

        self.dense_chemical = tf.keras.layers.Dense(self.args.sequential_dense, activation='relu')
        self.dropout_chemical = tf.keras.layers.Dropout(self.args.concat_dropout)

    def call(self, nmr_tensor, chemical_tensor, training=None, modal_mask=None):
        output = tf.keras.backend.expand_dims(nmr_tensor, axis=-1)

        for layer in self.sequential_layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Dropout) or \
                    isinstance(layer, Residual) or isinstance(layer, ConvPool):
                output = layer(output, training=training)
            else:
                output = layer(output)

        output = self.normalizer_1(output)
        output = self.large_conv(output)

        output = self.encoder_nmr(output, mask=None, training=training)

        chemical_tensor = self.encoder(chemical_tensor, mask=None, training=training)
        output = tf.keras.layers.concatenate([output, chemical_tensor], 1)
        # output = self.decoder(chemical_tensor, enc_output=output, training=training, look_ahead_mask=None,
        #                         padding_mask=None)
        print(output.shape, "NMR_infuser")
        output = self.decoder(output, mask=modal_mask, training=training)
        # Mixup NMR into chemical
        output = output[:, -1, :]
        output = self.dense(output)
        output = self.dropout(output, training=training)
        
        return output
        # chemical_tensor = chemical_tensor[:, 1, :]
        # chemical_tensor = self.dense_chemical(chemical_tensor)
        # chemical_tensor = self.dropout_chemical(chemical_tensor, training=training)
        # return tf.keras.layers.concatenate([output, chemical_tensor], 1)
