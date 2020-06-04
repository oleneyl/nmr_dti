import tensorflow as tf
from .sequence import NMRInferenceModel


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


class BaseNMRModel(object):
    def __init__(self, args, export_level='end'):
        self._smiles_encoded = tf.keras.Input(shape=[args.chemical_sequence_length], dtype=tf.int32, name='smile_input')
        self._modal_mask = tf.keras.Input(shape=[1, 1, args.chemical_sequence_length], dtype=tf.float32,
                                          name='modal_mask')
        self._output_mask = tf.keras.Input(shape=[args.chemical_sequence_length], dtype=tf.float32, name='output_mask')
        self._smiles_encoded_len = tf.keras.Input(shape=[1], dtype=tf.int32, name='smile_input_len')
        self._atom_embed = tf.keras.Input(shape=[args.chemical_sequence_length, 31], dtype=tf.float32, name='atom_embed')
        self.args = args

        self.protein_model = None
        self.chemical_model = None

        with tf.name_scope('chemical'):
            self.nmr_model = NMRInferenceModel(args, args.chemical_vocab_size, vectorized=True)

        self.embedding = tf.keras.layers.Embedding(args.chemical_vocab_size, args.transformer_model_dim)
        self.embedding_2 = tf.keras.layers.Dense(args.transformer_model_dim, activation=None)

    def inputs(self):
        if self.args.atom_embedding:
            return [self._smiles_encoded_len, self._smiles_encoded, self._modal_mask, self._output_mask, self._atom_embed]
        else:
            return [self._smiles_encoded_len, self._smiles_encoded, self._modal_mask, self._output_mask]
        # return [self._smiles_encoded_len, self._smiles_encoded, self._output_mask]

    def unsupervised_chemical(self):
        pass

    def predict(self):
        if self.args.atom_embedding:
            enc_1 = self.embedding(self._smiles_encoded)
            enc_2 = self.embedding_2(self._atom_embed)
            enc = enc_1 + enc_2
        else:
            enc = self.embedding(self._smiles_encoded)
        # inference_output = self.nmr_model(self._smiles_encoded, mask=self._modal_mask)
        inference_output = self.nmr_model(enc, mask=self._modal_mask)
        # inference_output = self.nmr_model(self._smiles_encoded)
        return inference_output

    def create_keras_model(self):
        model_inputs = self.inputs()
        prediction = self.predict() * self._output_mask
        keras_model = tf.keras.Model(inputs=model_inputs, outputs=prediction)
        return keras_model
