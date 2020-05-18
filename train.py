import tensorflow as tf
import numpy as np
import os

from options import get_args
from models import get_model
from progress import get_progress_handler, ProgressLogger
from preprocess.data_utils.data_loader import get_data_loader
from pprint import pprint
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras.callbacks import TensorBoard
import time


def auroc(y_true, y_pred):  # Should be calculated over whole batch
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)


def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f)


def train(args):
    def create_modal_mask(_datum):
        # Create mask to disable NMR input and attention to NMR part.
        chemical_length = args.chemical_sequence_length
        nmr_length = args.modal_size - chemical_length
        _label, _protein, _chemical, _nmr = _datum
        mask_list = []
        for i in range(_nmr.shape[0]):
            nmr_piece = _nmr[i]
            if np.sum(nmr_piece) > 1e-8:  # data exist
                mask_list.append(np.zeros([1, chemical_length + nmr_length, chemical_length + nmr_length]))
            else:
                mask = np.zeros([1, chemical_length, chemical_length])
                mask = np.concatenate([np.ones([1, chemical_length, nmr_length]), mask], axis=2)
                mask = np.concatenate([np.ones([1, nmr_length, chemical_length + nmr_length]), mask], axis=1)
                mask_list.append(mask)
         
        mask = np.concatenate(mask_list, axis=0)
        mask = np.expand_dims(mask, axis=1)
        return mask
    
    def create_input_sample(_datum):
        _label, _protein, _chemical, _nmr = _datum
        if args.nmr:
            modal_mask = create_modal_mask(_datum)
            return [_protein, _chemical, _nmr, modal_mask], _label
        else:
            return [_protein, _chemical], _label       

    print("***  Run environment  ***")
    pprint(args)
    print("\n")

    # Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    # Define Placeholders
    result_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='result_pl')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Create model
    drug_target_interaction = get_model(args)
    model = drug_target_interaction.create_keras_model()

    # Compile model with metrics
    if args.as_score:
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss='mean_squared_error',
                      metrics=['mean_squared_error', cindex_score])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss=tf.keras.losses.binary_crossentropy,
                      metrics=[tf.keras.metrics.Precision(name='precision'),
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.FalsePositives(name='false_positives'),
                               tf.keras.metrics.FalseNegatives(name='false_negatives'),
                               auroc])

    # Summary
    model.summary()
    tensorboard, logger = get_progress_handler(args)
    tensorboard.info()  # Print Information what now tensorboard is tracking

    metrics_names = model.metrics_names
    global_step = 0

    # Experiment
    for epoch in range(1, args.epoch + 1):
        model.reset_metrics()
        train_data, valid_data, test_data = get_data_loader(args)

        print(f'Epoch {epoch} start')

        # Train
        for idx, datum in enumerate(train_data):
            xs, ys = create_input_sample(datum)
            print(ys)
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
