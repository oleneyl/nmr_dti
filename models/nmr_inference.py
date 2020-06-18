import tensorflow as tf
import numpy as np
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


class ComplexNMRModel(object): 
    def __init__(self, args, export_level='end'):
        self._smiles_encoded = tf.keras.Input(shape=[args.chemical_sequence_length], dtype=tf.int32, name='smile_input')
        self._modal_mask = tf.keras.Input(shape=[1, 1, args.chemical_sequence_length], dtype=tf.float32,
                                          name='modal_mask')
        self._output_mask = tf.keras.Input(shape=[args.chemical_sequence_length], dtype=tf.float32, name='output_mask')
        self._smiles_encoded_len = tf.keras.Input(shape=[1], dtype=tf.int32, name='smile_input_len')
        self._atom_embed = tf.keras.Input(shape=[args.chemical_sequence_length, 31], dtype=tf.float32, name='atom_embed')
        self._bond_mask = tf.keras.Input(shape=[args.chemical_sequence_length, args.chemical_sequence_length], dtype=tf.float32, name='bond_mask')
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


from .modules.atomic import AtomicLayer

class AtomicNet(object):
    def __init__(self, args):
        self._atom_type = tf.keras.Input(shape=[args.chemical_sequence_length], dtype=tf.int32, name='atom_input')
        self._orbit_coeff = tf.keras.Input(shape=[args.chemical_sequence_length, args.chemical_sequence_length], dtype=tf.float32)
        self._distance = tf.keras.Input(shape=[args.chemical_sequence_length, args.chemical_sequence_length], dtype=tf.float32)
        self._angle = tf.keras.Input(shape=[args.chemical_sequence_length, args.chemical_sequence_length], dtype=tf.float32)
        self._extract_matrix = tf.keras.Input(shape=[args.chemical_sequence_length, args.chemical_sequence_length], dtype=tf.float32)
        self._output_mask = tf.keras.Input(shape=[args.chemical_sequence_length], dtype=tf.float32, name='output_mask')
        self.args = args

        with tf.name_scope('chemical'):
            self.nmr_model = NMRInferenceModel(args, args.chemical_vocab_size, vectorized=True)

        self.embedding = tf.keras.layers.Embedding(args.chemical_vocab_size, args.transformer_model_dim)
        self.atomic_layer = [
            AtomicLayer(args.transformer_model_dim, args.transformer_num_heads, args.transformer_hidden_dimension) for i in range(args.transformer_num_layers)
        ]
        self.layer_num = args.transformer_num_layers
        self.output_dense = tf.keras.layers.Dense(1)

    def inputs(self):
        return [self._atom_type, self._orbit_coeff, self._distance, self._angle, self._extract_matrix, self._output_mask]

    def unsupervised_chemical(self):
        pass

    def predict(self):
        orbit_state = self.embedding(self._atom_type)
        orbit_coeff = self._orbit_coeff
        for i in range(self.layer_num):
            orbit_state, orbit_coeff = self.atomic_layer[i]([orbit_state, orbit_coeff, -1 * self._distance + tf.abs(self._angle)])
        with tf.name_scope('matmul_test_1'):
            state_output = tf.matmul(orbit_coeff, orbit_state)
        with tf.name_scope('matmul_test_2'):
            gather_state = tf.matmul(self._extract_matrix, state_output)
        inference_output = self.output_dense(gather_state)
        inference_output = tf.squeeze(inference_output, axis=-1)
        return inference_output

    def create_keras_model(self):
        model_inputs = self.inputs()
        prediction = self.predict() * self._output_mask
        keras_model = tf.keras.Model(inputs=model_inputs, outputs=prediction)
        return keras_model

