import tensorflow as tf
import numpy as np
import os

from options import get_inference_args
from models.nmr_inference import BaseNMRModel
from models.modules.attention import create_padding_mask
from progress import get_progress_handler
from preprocess.data_utils.data_loader import NMRDataLoader
from learning_rate import get_learning_rate_scheduler
from pprint import pprint


def train(args):
    def create_modal_mask(_datum):
        _smiles_len, _smiles, _nmr_value_list, _mask = _datum
        _mask = np.zeros(_smiles.shape)
        return _mask

    def create_input_sample(_datum):
        _smiles_len, _smiles, _nmr_value_list, _pad_mask, _mask = _datum
        # _pad_mask = 1.0 - _mask
        _pad_mask = _pad_mask[:, tf.newaxis, tf.newaxis, :]
        # _pad_mask = create_padding_mask(_smiles)
        return [_smiles_len, _smiles, _pad_mask, _mask], _nmr_value_list
        # return [_smiles_len, _smiles, _mask], _nmr_value_list

    print("***  Run environment  ***")
    pprint(args)
    print("\n")

    # Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    # Create model
    nmr_interaction = BaseNMRModel(args)
    model = nmr_interaction.create_keras_model()
    model.load_weights(args.model_path)

    # Compile model with metrics
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.MSE, metrics=[tf.keras.losses.MSE, tf.keras.losses.MAE])

    # Summary
    model.summary()
    tensorboard, logger = get_progress_handler(args)
    tensorboard.info()  # Print Information what now tensorboard is tracking

    logger.print_log(str(args))
    metrics_names = model.metrics_names
    global_step = 0

    model.reset_metrics()
    # TODO data loader must be changed
    test_data = NMRDataLoader(args.nmr_dir, 'test', batch_size=args.batch_size, chemical_sequence_length=args.chemical_sequence_length)

    # Validation / Test
    set_type = 'inference'
    mae_loss = 0
    mse_loss = 0
    c_count = 0
    for idx, datum in enumerate(test_data):
        xs, ys = create_input_sample(datum)
        pred_ys = model.predict_on_batch(xs)
        delta = pred_ys - ys
        # MAE
        abs_delta = np.abs(delta)
        mae_loss += np.sum(abs_delta)
        mse_loss += np.sum(abs_delta * abs_delta)
        c_count += np.sum(xs[3])
        delta = np.sum(abs_delta, axis=-1) / np.sum(xs[3], axis=-1)
        if idx == 0:
            print('--')
            print(pred_ys)
            print('--')
            print(ys)
            print('--')
            print(abs_delta)
            print('--')
            print(xs[3])
            print('--')
            print(np.sum(abs_delta, axis=-1))
            print('--')
            print(np.sum(xs[3], axis=-1))
            print('--')
            print(delta)
            print('--')
            print(np.average(delta))



    print(mae_loss / c_count)
    print(mse_loss / c_count)


if __name__ == '__main__':
    train(get_inference_args())
