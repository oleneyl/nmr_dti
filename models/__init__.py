import tensorflow as tf
from .cnn import NMRModel
from .sequence import RNNProteinModel


def add_model_args(parser):
    group = parser.add_argument_group('model')

    group.add_argument('--nmr_model', type=str, default='cnn')
    group.add_argument('--protein_model', type=str, default='gru')
    group.add_argument('--chemical_model', type=str, default='gru')


def build_model(args, protein_encoded, nmr_array, smiles_encoded):
    """
    Create output generation model from given placeholders
    """
    protein_code = RNNProteinModel(args).compile(protein_encoded)
    # smiles_code = smiles_model(smiles_encoded)
    nmr_code = NMRModel(args).compile(nmr_array)

    # Concat three values
    embedding = tf.concat([protein_code, nmr_code], 1)
    embedding = tf.keras.layers.Dense(40, activation='relu')(embedding)
    embedding = tf.keras.layers.Dropout(0.5)(embedding)
    embedding = tf.keras.layers.BatchNormalization()(embedding)
    embedding = tf.keras.layers.Dense(1)(embedding)

    return embedding


def get_model(args, protein_encoded, nmr_array, smiles_encoded, saved_model=None):
    if saved_model:
        # Load model from saved_model path
        pass
    else:
        return build_model(args, protein_encoded, nmr_array, smiles_encoded)
