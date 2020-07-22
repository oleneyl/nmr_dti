import tensorflow as tf
import numpy as np
import os

from options import get_args
from models.label_inference import InferenceAtomicNet
from progress import get_progress_handler
from preprocess.moleculenet import get_dataset_loader
from learning_rate import get_learning_rate_scheduler
from pprint import pprint
from sklearn.metrics import roc_auc_score
from models.fit_moleculenet_task import get_moleculenet_task_dependent_arguments


def wrapped_roc_auc_score(y_true, y_pred):
    try:
        x = roc_auc_score(y_true, y_pred)
        return x
    except ValueError:
        return 0.5


def auroc(y_true, y_pred):  # Should be calculated over whole batch
    return tf.py_func(wrapped_roc_auc_score, (y_true, y_pred), tf.double)


def multi_auroc(y_true, y_pred, label_count):  # Should be calculated over whole batch
    print(y_true.shape)
    print(y_pred.shape)
    output = []
    for idx in range(label_count):
        y_true_v = y_true[:, idx]
        y_pred_v = y_pred[:, idx]
        try:
            score = roc_auc_score(y_true_v, y_pred_v)
        except ValueError:
            score = None

        output.append(score)

    corrected_output = [0.5 if x is None else x for x in output]
    filtered_output = [x for x in output if x is not None]

    corrected_mean = np.mean(corrected_output)
    filtered_mean = np.mean(filtered_output)

    return corrected_mean, filtered_mean, output


def train(args):
    def create_input_sample(_datum):
        _embedding_list, _distance, _angular_distance, _orbital_matrix, _labels, _pad_mask = _datum
        _output_mask = 1.0 - _pad_mask
        _pad_mask = _pad_mask[:, tf.newaxis, tf.newaxis, :]
        # _pad_mask = create_padding_mask(_smiles)
        return ([_embedding_list,
                 np.stack([np.eye(_embedding_list.shape[-1]) for i in range(_embedding_list.shape[0])], axis=0),
                 _distance,
                 _angular_distance,
                 _orbital_matrix,
                 _pad_mask,
                 _output_mask], _labels)

    print("***  Run environment  ***")
    pprint(args)
    print("\n")


    # Experiment
    for epoch in range(1):
        # TODO data loader must be changed
        train_data = get_dataset_loader(args.moleculenet_task, 'train', batch_size=args.batch_size,
                                        chemical_sequence_length=args.chemical_sequence_length)
        valid_data = get_dataset_loader(args.moleculenet_task, 'valid', batch_size=args.batch_size,
                                        chemical_sequence_length=args.chemical_sequence_length)
        test_data = get_dataset_loader(args.moleculenet_task, 'test', batch_size=args.batch_size,
                                       chemical_sequence_length=args.chemical_sequence_length)

        datas = []

        # Train
        for idx, datum in enumerate(train_data):
            xs, ys = create_input_sample(datum)
            datas += np.nan_to_num(ys)

        print(len(datas), np.std(datas), np.var(datas), np.mean(datas))

        # Validation / Test
        for dataset, set_type in ((valid_data, 'valid'), (test_data, 'test')):
            if set_type == 'test' and args.skip_test:
                continue

            datas = []
            for datum in dataset:
                xs, ys = create_input_sample(datum)
                datas += np.nan_to_num(ys)

            print(len(datas), np.std(datas), np.var(datas), np.mean(datas))

if __name__ == '__main__':
    train(get_args())
