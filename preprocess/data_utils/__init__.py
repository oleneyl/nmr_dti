from .data_loader import data_loader_args
from .numpy_adapter import adapter_args


def add_data_util_args(parser):
    data_loader_args(parser)
    adapter_args(parser)
