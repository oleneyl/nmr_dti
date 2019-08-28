from tqdm import tqdm
from argparse import ArgumentParser

from data_utils import create_every_alignment, create_joined_data, change_configuration

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default='create_all')
    parser.add_argument('--conf', type=str, default='')
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()

    if len(args.conf) > 0:
        change_configuration(args.conf)

    if args.task == 'create_all':
        create_every_alignment(wrapper=tqdm)
    elif args.task == 'create_join':
        create_joined_data(wrapper=tqdm)
