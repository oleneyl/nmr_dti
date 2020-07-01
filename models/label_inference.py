import tensorflow as tf
import numpy as np
from .sequence import NMRInferenceModel
from .modules.atomic import AtomicLayer


class InferenceAtomicNet(object):
    def __init__(self, args,
                 label_count=12,
                 is_regression=False):

        atomic_length = args.chemical_sequence_length // 2
        self._atom_type = tf.keras.Input(shape=[args.chemical_sequence_length], dtype=tf.int32, name='atom_input')
        self._orbit_coeff = tf.keras.Input(shape=[args.chemical_sequence_length, args.chemical_sequence_length], dtype=tf.float32)
        self._distance = tf.keras.Input(shape=[args.chemical_sequence_length, args.chemical_sequence_length], dtype=tf.float32)
        self._angle = tf.keras.Input(shape=[args.chemical_sequence_length, args.chemical_sequence_length], dtype=tf.float32)
        self._extract_matrix = tf.keras.Input(shape=[atomic_length, args.chemical_sequence_length], dtype=tf.float32)
        self._pad_mask = tf.keras.Input(shape=[1, 1, args.chemical_sequence_length], dtype=tf.float32,
                                          name='pad_mask')

        self.args = args
        self.label_count = label_count
        self.is_regression = is_regression


        with tf.name_scope('chemical'):
            self.nmr_model = NMRInferenceModel(args, args.chemical_vocab_size, vectorized=True)

        self.embedding = tf.keras.layers.Embedding(args.chemical_vocab_size, args.transformer_model_dim)
        self.atomic_layer = [
            AtomicLayer(args.transformer_model_dim,
                        args.transformer_num_heads,
                        args.transformer_hidden_dimension) for i in range(args.transformer_num_layers)
        ]
        self.layer_num = args.transformer_num_layers

        self.collect_dense = tf.keras.layers.Dense(1)
        self.output_dense = tf.keras.layers.Dense(self.label_count)

    def inputs(self):
        return [self._atom_type,
                self._orbit_coeff,
                self._distance,
                self._angle,
                self._extract_matrix,
                self._pad_mask]

    def unsupervised_chemical(self):
        pass

    def predict(self):
        orbit_state = self.embedding(self._atom_type)
        orbit_coeff = self._orbit_coeff
        for i in range(self.layer_num):
            # orbit_state, orbit_coeff = self.atomic_layer[i]([orbit_state, orbit_coeff, -1 * self._distance], mask=self._pad_mask)
            orbit_state = self.atomic_layer[i]([orbit_state, -1 * self._distance], mask=self._pad_mask)
        with tf.name_scope('matmul_test_1'):
            print('state_output')
            print(orbit_state.shape, orbit_coeff.shape)
            state_output = tf.matmul(orbit_coeff, orbit_state)
        with tf.name_scope('matmul_test_2'):
            print(self._extract_matrix.shape)
            print(state_output.shape)
            gather_state = tf.matmul(self._extract_matrix, state_output)

        inference_output = self.collect_dense(gather_state)
        inference_output = tf.squeeze(inference_output, axis=-1)
        inference_output = self.output_dense(inference_output)

        if not self.is_regression:
            inference_output = tf.nn.sigmoid(inference_output)
        return inference_output

    def create_keras_model(self):
        model_inputs = self.inputs()
        prediction = self.predict()
        keras_model = tf.keras.Model(inputs=model_inputs, outputs=prediction)
        return keras_model

