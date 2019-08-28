from tqdm import tqdm
from argparse import ArgumentParser

from data_utils import create_every_alignment, create_joined_data

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default='create_all')

    return parser.parse_args()


if __name__=='__main__':
    args = get_args()

    if args.task == 'create_all':
        create_every_alignment(wrapper=tqdm)
    elif args.task == 'create_join':
        create_joined_data(wrapper=tqdm)
