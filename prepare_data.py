from tqdm import tqdm
from argparse import ArgumentParser

from preprocess import create_every_alignment, create_trainable_data, change_configuration, mix_nmr, strict_split_data
from preprocess import create_dataset_from_ibm, create_dataset_from_kiba, create_dataset_from_davis
from preprocess.pubchem import collect_many
from preprocess.tokenizer import create_protein_vocab, create_chemical_vocab, create_chemical_vocab_from_dataset

from preprocess.download import FileDownloader

def get_args():
    parser = ArgumentParser()
    parser.add_argument('--task', type=str, default='create_all')
    parser.add_argument('--conf', type=str, default='')

    parser.add_argument('--output_prefix', type=str)
    parser.add_argument('--split_ratio', type=float, default=0.95)
    parser.add_argument('--nmr', action='store_true')
    parser.add_argument('--dataset', type=str)

    parser.add_argument('--download_dir', type=str)

    parser.add_argument('--as_binary', help='Degenerate float output into binary output', action='store_true')
    return parser.parse_args()


if __name__=='__main__':
    args = get_args()

    if len(args.conf) > 0:
        change_configuration(args.conf)

    if args.task == 'download':
        file_downloader = FileDownloader(args.download_dir)
        file_downloader.download(args.dataset)

    if args.task == 'create_all':
        create_every_alignment(wrapper=tqdm)
    elif args.task == 'create_dataset':
        # create_trainable_data(args.split_ratio, wrapper=tqdm, use_nmr=args.nmr)
        strict_split_data(args.split_ratio, save_dir=args.output_prefix, wrapper=tqdm, use_nmr=args.nmr)
    elif args.task == 'protein_vocab':
        create_protein_vocab(args.output_prefix)
    elif args.task == 'chemical_vocab':
        create_chemical_vocab(args.output_prefix)
    elif args.task == 'chemical_vocab_from_data':
        create_chemical_vocab_from_dataset(args.dataset, args.output_prefix)
    elif args.task == 'mix_nmr':
        mix_nmr(args.output_prefix + '.mix_nmr', wrapper=tqdm)
    elif args.task == 'create_from_ibm':
        create_dataset_from_ibm(args.output_prefix)
    elif args.task == 'create_from_kiba':
        create_dataset_from_kiba(args.output_prefix, as_binary=args.as_binary)
    elif args.task == 'create_from_davis':
        create_dataset_from_davis(args.output_prefix, as_binary=args.as_binary)
    elif args.task == 'get_pubchem':
        print('\n Start fetching pubchem data \n')
        collect_many(10000*10000+1, 30000*10000, task_pool_size=6)
