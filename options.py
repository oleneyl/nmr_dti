from argparse import ArgumentParser
from preprocess.data_utils import add_data_util_args
from models import add_model_args
from progress import add_progress_args


def get_args():
    parser = ArgumentParser()

    base_group = parser.add_argument_group('base')

    add_data_util_args(parser)
    add_model_args(parser)
    add_progress_args(parser)
    add_training_args(parser)

    return parser.parse_args()


def add_training_args(parser):
    group = parser.add_argument_group('training')

    group.add_argument('--lr', type=float, default=0.001)
    group.add_argument('--epoch', type=int, default=1)