import tensorflow as tf
from .cnn import NMRModel
from .sequence import RNNProteinModel


def add_model_args(parser):
    group = parser.add_argument_group('model')

    group.add_argument('--nmr_model', type=str, default='cnn')
    group.add_argument('--protein_model', type=str, default='gru')
    group.add_argument('--chemical_model', type=str, default='gru')
    group.add_argument('--protein_embedding_size', type=int, default=128)
    group.add_argument('--chemical_embedding_size', type=int, default=128)

    # Sequencial model control
    group.add_argument('--sequential_hidden_size', type=int, default=100)
    group.add_argument('--sequential_dropout', type=float, default=0.5)
    group.add_argument('--sequential_dense', type=int, default=64)

    # CNN model control
    group.add_argument('--cnn_filter_size', type=int, default=5)
    group.add_argument('--cnn_filter_number', type=int, default=5)
    group.add_argument('--cnn_hidden_layer', type=int, default=60)

    # Concat-model control
    group.add_argument('--concat_hidden_layer_size', type=int, default=40)
    group.add_argument('--concat_dropout', type=float, default=0.5)


def build_model(args, protein_encoded, nmr_array, smiles_encoded, is_train=True):
    """
    Create output generation model from given placeholders
    """
    protein_embedding = tf.keras.layers.Embedding(args.protein_vocab_size, args.protein_embedding_size)
    smile_embedding = tf.keras.layers.Embedding(args.chemical_vocab_size, args.chemical_embedding_size)

    if args.protein_model == 'gru':
        protein_code = RNNProteinModel(args, protein_embedding(protein_encoded), is_train=is_train)
    # smiles_code = smiles_model(smile_embedding(smiles_encoded))
    if args.nmr_model == 'cnn':
        nmr_code = NMRModel(args, nmr_array, is_train=is_train)

    # Concat three values
    embedding = tf.concat([protein_code.output, nmr_code.output], 1)
    embedding = tf.keras.layers.Dense(args.concat_hidden_layer_size, activation='relu')(embedding)
    embedding = tf.keras.layers.Dropout(args.concat_dropout)(embedding, training=is_train)
    embedding = tf.keras.layers.BatchNormalization()(embedding, training=is_train)
    embedding = tf.keras.layers.Dense(1)(embedding)

    return embedding


def get_model(args, protein_encoded, nmr_array, smiles_encoded, saved_model=None, is_train=True):
    if saved_model:
        # Load model from saved_model path
        pass
    else:
        return build_model(args, protein_encoded, nmr_array, smiles_encoded, is_train=is_train)
