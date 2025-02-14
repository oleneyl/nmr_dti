from argparse import ArgumentParser
from preprocess.data_utils import add_data_util_args
from models import add_model_args
from progress import add_progress_args
from learning_rate import add_learning_rate_option

def get_args():
    parser = ArgumentParser()

    base_group = parser.add_argument_group('base')
    base_group.add_argument('--nmr', action='store_true', help='determine whether use nmr data or not')
    base_group.add_argument('--drop_smile', action='store_true', help='If true, do not use SMILES data for prediction')

    add_data_util_args(parser)
    add_model_args(parser)
    add_progress_args(parser)
    add_training_args(parser)
    add_learning_rate_option(parser)

    return parser.parse_args()


def add_training_args(parser):
    group = parser.add_argument_group('training')

    group.add_argument('--epoch', type=int, default=1)
    group.add_argument('--grad_clip', type=float, default=1.0)

    group = parser.add_argument_group('device')
    group.add_argument('--gpu_number', type=int, default=3)