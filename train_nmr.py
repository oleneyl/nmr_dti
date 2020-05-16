import tensorflow as tf
import numpy as np
import os

from options import get_args
from models.nmr_inference import BaseNMRModel
from models.modules.attention import create_padding_mask
from progress import get_progress_handler
from preprocess.data_utils.data_loader import NMRDataLoader
from pprint import pprint


def train(args):
    def create_modal_mask(_datum):
        _smiles_len, _smiles, _nmr_value_list, _mask = _datum
        _mask = np.zeros(_smiles.shape)
        return _mask

    def create_input_sample(_datum):
        _smiles_len, _smiles, _nmr_value_list, _mask = _datum
        _pad_mask = create_padding_mask(_smiles)
        return [_smiles_len, _smiles, _pad_mask, _mask], _nmr_value_list

    print("***  Run environment  ***")
    pprint(args)
    print("\n")

    # Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    # Create model
    nmr_interaction = BaseNMRModel(args)
    model = nmr_interaction.create_keras_model()

    # Compile model with metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss=tf.keras.losses.MSE,
                  metrics=[tf.keras.losses.MSE])

    # Summary
    model.summary()
    tensorboard, logger = get_progress_handler(args)
    tensorboard.info()  # Print Information what now tensorboard is tracking

    metrics_names = model.metrics_names
    global_step = 0

    # Experiment
    for epoch in range(1, args.epoch + 1):
        model.reset_metrics()
        # TODO data loader must be changed
        train_data = NMRDataLoader(args.nmr_dir, 'train', batch_size=args.batch_size, chemical_sequence_length=args.chemical_sequence_length)
        valid_data = NMRDataLoader(args.nmr_dir, 'valid', batch_size=args.batch_size, chemical_sequence_length=args.chemical_sequence_length)
        test_data = NMRDataLoader(args.nmr_dir, 'test', batch_size=args.batch_size, chemical_sequence_length=args.chemical_sequence_length)

        print(f'Epoch {epoch} start')

        # Train
        for idx, datum in enumerate(train_data):
            xs, ys = create_input_sample(datum)
            result = model.train_on_batch(xs, ys)
            global_step += 1
            if idx % args.log_interval == 0:
                logger.emit("Training", metrics_names, result)
                tensorboard.create_summary(global_step, result, model, prefix='train')


        # Validation / Test
        for dataset, set_type in ((valid_data, 'valid'), (test_data, 'test')):
            for datum in dataset:
                xs, ys = create_input_sample(datum)
                result = model.test_on_batch(xs, ys, reset_metrics=False)
            is_best = logger.emit(set_type, metrics_names, result)
            # if is_best:
            #    tensorboard.save_model(model, 'best')
            tensorboard.create_summary(global_step, result, model, prefix=set_type)
            model.reset_metrics()

        logger.best("valid")

    logger.emit_history("test", logger.best_index("valid"))
    tensorboard.save_model(model, 'last')


if __name__ == '__main__':
    train(get_args())
