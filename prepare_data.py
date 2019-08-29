from tqdm import tqdm
from argparse import ArgumentParser

from preprocess import create_every_alignment, create_trainable_data, change_configuration, mix_nmr
from preprocess.tokenize import create_protein_vocab

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default='create_all')
    parser.add_argument('--conf', type=str, default='')

    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--split_ratio', type=float, default=0.95)
    parser.add_argument('--nmr', action='store_true')
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()

    if len(args.conf) > 0:
        change_configuration(args.conf)

    if args.task == 'create_all':
        create_every_alignment(wrapper=tqdm)
    elif args.task == 'create_join':
        create_trainable_data(args.split_ratio, wrapper=tqdm, use_nmr=args.nmr)
    elif args.task == 'protein_vocab':
        create_protein_vocab(args.output_prefix)
    elif args.task == 'mix_nmr':
        mix_nmr(args.output_prefix + '.mix_nmr', wrapper=tqdm)