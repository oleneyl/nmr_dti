import tensorflow as tf
from .cnn import NMRModel
from .sequence import RNNProteinModel, AttentionProteinModel, transformer_args
from .attention import EncoderSepLayer, DecoderLayer, DecoderLayerIndiv, EncoderLayer


def add_model_args(parser):
    transformer_args(parser)
    group = parser.add_argument_group('model')

    group.add_argument('--nmr_model', type=str, default='cnn')
    group.add_argument('--protein_model', type=str, default='gru')
    group.add_argument('--chemical_model', type=str, default='gru')
    group.add_argument('--protein_embedding_size', type=int, default=128)
    group.add_argument('--chemical_embedding_size', type=int, default=128)

    # Sequencial model control
    group.add_argument('--sequential_hidden_size', type=int, default=128)
    group.add_argument('--sequential_dropout', type=float, default=0.5)
    group.add_argument('--sequential_dense', type=int, default=64)

    # CNN model control
    group.add_argument('--cnn_filter_size', type=int, default=5)
    group.add_argument('--cnn_filter_number', type=int, default=32)
    group.add_argument('--cnn_hidden_layer', type=int, default=60)

    # Concat-model control
    group.add_argument('--concat_model', type=str, default='siamese')
    group.add_argument('--concat_hidden_layer_size', type=int, default=20)
    group.add_argument('--siamese_layer_size', type=int, default=32)
    group.add_argument('--concat_dropout', type=float, default=0.5)


class BaseDTIModel(object):
    def __init__(self, args, protein_encoded, nmr_array, smiles_encoded, is_train=True):
        self._protein_encoded = protein_encoded
        self._nmr_array = nmr_array
        self._smiles_encoded = smiles_encoded
        self.args = args
        self.is_train = is_train
        self.nmr_model = NMRModel(args, nmr_array, is_train=is_train)
        self.protein_model = None
        self.chemical_model = None
        with tf.name_scope('protein'):
            if args.protein_model == 'gru':
                self.protein_model = RNNProteinModel(args, protein_encoded, args.protein_vocab_size, is_train=is_train)
            elif args.protein_model == 'att':
                self.protein_model = AttentionProteinModel(args, protein_encoded, args.protein_vocab_size,
                                                           is_train=is_train)
        with tf.name_scope('chemical'):
            if args.chemical_model == 'gru':
                self.chemical_model = RNNProteinModel(args, smiles_encoded, args.chemical_vocab_size, is_train=is_train)
            elif args.chemical_model == 'att':
                self.chemical_model = AttentionProteinModel(args, smiles_encoded, args.chemical_vocab_size,
                                                            is_train=is_train)
        with tf.name_scope('last'):
            self.output = self.predict_dti()

    def unsupervised_protein(self):
        pass

    def unsupervised_chemical(self):
        pass

    def predict_dti(self):
        return 1

    def inspect_model_output(self):
        return self.nmr_model.get_output(), self.protein_model.get_output(), self.chemical_model.get_output()


class InitialDTIModel(BaseDTIModel):
    def predict_dti(self):
        embedding = tf.concat([self.protein_model.get_output(), self.nmr_model.get_output(),
                               self.chemical_model.get_output()], 1)
        embedding = tf.keras.layers.Dense(self.args.concat_hidden_layer_size, activation='relu',
                                          name='concat_dense_1')(embedding)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding, training=self.is_train)
        embedding = tf.keras.layers.BatchNormalization()(embedding, training=self.is_train)
        embedding = tf.keras.layers.Dense(self.args.concat_hidden_layer_size, activation='relu',
                                          name='concat_dense_1')(embedding)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding, training=self.is_train)
        embedding = tf.keras.layers.BatchNormalization()(embedding, training=self.is_train)
        dense_last = tf.keras.layers.Dense(1, name='concat_dense_last')
        embedding = dense_last(embedding)
        return embedding


class SiameseDTIModel(BaseDTIModel):
    def predict_dti(self):
        protein_siamese = tf.keras.layers.Dense(self.args.concat_hidden_layer_size, activation='relu',
                                                name='siamese_protein')(self.protein_model.get_output())
        protein_siamese = tf.keras.layers.Dropout(self.args.concat_dropout)(protein_siamese, training=self.is_train)
        protein_siamese = tf.keras.layers.BatchNormalization()(protein_siamese, training=self.is_train)
        protein_siamese = tf.keras.layers.Dense(self.args.siamese_layer_size)(protein_siamese)

        nmr_siamese = tf.keras.layers.Dense(self.args.concat_hidden_layer_size, activation='relu',
                                            name='siamese_protein')(self.nmr_model.get_output())
        nmr_siamese = tf.keras.layers.Dropout(self.args.concat_dropout)(nmr_siamese, training=self.is_train)
        nmr_siamese = tf.keras.layers.BatchNormalization()(nmr_siamese, training=self.is_train)
        nmr_siamese = tf.keras.layers.Dense(self.args.siamese_layer_size)(nmr_siamese)
        return tf.sigmoid(tf.reduce_sum(tf.multiply(protein_siamese, nmr_siamese), axis=1, keep_dims=True))


class AttentiveDTIModel(BaseDTIModel):
    def predict_dti(self):
        protein_encoding = self.protein_model.encoding
        chemical_encoding = self.chemical_model.encoding

        protein_encoding = tf.keras.layers.Dense(64)(protein_encoding)
        protein_encoding = tf.keras.layers.Dense(64)(chemical_encoding)

        output = tf.keras.layers.Attention(use_scale=True)([chemical_encoding, protein_encoding])

        output = tf.keras.layers.Dense(64)(output)
        output = tf.keras.layers.Attention(use_scale=True)([chemical_encoding, output])

        output = tf.keras.layers.Flatten()(output)
        embedding = tf.keras.layers.Dense(self.args.concat_hidden_layer_size, activation='relu',
                                          name='concat_dense_1')(output)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding, training=self.is_train)
        embedding = tf.keras.layers.BatchNormalization()(embedding, training=self.is_train)
        dense_last = tf.keras.layers.Dense(1, activation='sigmoid', name='concat_dense_last')
        embedding = dense_last(embedding)
        return embedding


class MHADTIModel(BaseDTIModel):
    def predict_dti(self):
        protein_encoding = self.protein_model.encoding
        chemical_encoding = self.chemical_model.encoding

        enc = [EncoderSepLayer(self.args.transformer_model_dim,
                              self.args.transformer_num_heads,
                              self.args.transformer_hidden_dimension,
                              rate=self.args.transformer_dropout_rate) for x in range(1)]
        output = protein_encoding
        for en in enc:
            output = en(chemical_encoding, protein_encoding, self.is_train, None)
        print(output)
        output = tf.keras.layers.Flatten()(output)
        embedding = tf.keras.layers.Dense(self.args.concat_hidden_layer_size, activation='relu',
                                          name='concat_dense_1')(output)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding, training=self.is_train)
        embedding = tf.keras.layers.BatchNormalization()(embedding, training=self.is_train)
        dense_last = tf.keras.layers.Dense(1, activation='sigmoid', name='concat_dense_last')
        embedding = dense_last(embedding)
        return embedding


class AttDecModel(BaseDTIModel):
    def predict_dti(self):
        protein_detect = self.protein_model.encoding
        chemical_detect = self.chemical_model.encoding

        # Cross - attention
        protein_decoder = DecoderLayerIndiv(self.args.transformer_model_dim,
                              self.args.transformer_num_heads,
                              self.args.transformer_hidden_dimension,
                              rate=self.args.transformer_dropout_rate)
        chemical_decoder = DecoderLayerIndiv(self.args.transformer_model_dim,
                                       self.args.transformer_num_heads,
                                       self.args.transformer_hidden_dimension,
                                       rate=self.args.transformer_dropout_rate)
        protein_detect_1, _, __ = protein_decoder(protein_detect, chemical_detect, self.is_train, None, None)
        chemical_detect_1, _, __ = chemical_decoder(chemical_detect, protein_detect, self.is_train, None, None)

        # Flatten
        protein_detect = tf.keras.layers.Flatten()(protein_detect_1)
        protein_detect = tf.keras.layers.Dense(self.args.sequential_dense, activation='relu')(protein_detect)

        chemical_detect = tf.keras.layers.Flatten()(chemical_detect_1)
        chemical_detect = tf.keras.layers.Dense(self.args.sequential_dense, activation='relu')(chemical_detect)

        embedding = tf.concat([protein_detect, chemical_detect, self.chemical_model.get_output()], 1)
        embedding = tf.keras.layers.Dense(self.args.concat_hidden_layer_size, activation='relu',
                                          name='concat_dense_1')(embedding)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding, training=self.is_train)
        embedding = tf.keras.layers.BatchNormalization()(embedding, training=self.is_train)
        dense_last = tf.keras.layers.Dense(1, name='concat_dense_last')
        embedding = dense_last(embedding)
        return embedding


class ConcatAttentionModel(BaseDTIModel):
    def predict_dti(self):
        protein_detect = self.protein_model.encoding
        chemical_detect = self.chemical_model.encoding

        x = tf.concat([protein_detect, chemical_detect], axis=1)
        # Cross - attention

        total_encoder = EncoderLayer(self.args.transformer_model_dim,
                               self.args.transformer_num_heads,
                               self.args.transformer_hidden_dimension,
                               rate=self.args.transformer_dropout_rate)
        x = total_encoder(x, self.is_train, None)
        embedding = tf.keras.layers.Flatten()(x)
        embedding = tf.keras.layers.Dense(self.args.concat_hidden_layer_size, activation='relu',
                                          name='concat_dense_1')(embedding)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding, training=self.is_train)
        embedding = tf.keras.layers.BatchNormalization()(embedding, training=self.is_train)
        dense_last = tf.keras.layers.Dense(1, name='concat_dense_last')
        embedding = dense_last(embedding)
        return embedding


def build_model(args, protein_encoded, nmr_array, smiles_encoded, is_train=True):
    """
    Create output generation model from given placeholders
    """
    if args.concat_model == 'siamese':
        model = SiameseDTIModel(args, protein_encoded, nmr_array, smiles_encoded, is_train=is_train)
        return model.output, model
    elif args.concat_model == 'attentive':
        model = ConcatAttentionModel(args, protein_encoded, nmr_array, smiles_encoded, is_train=is_train)
        return model.output, model
    else:
        model = InitialDTIModel(args, protein_encoded, nmr_array, smiles_encoded, is_train=is_train)
        return model.output, model


def get_model(args, protein_encoded, nmr_array, smiles_encoded, saved_model=None, is_train=True):
    if saved_model:
        # Load model from saved_model path
        pass
    else:
        return build_model(args, protein_encoded, nmr_array, smiles_encoded, is_train=is_train)
