import tensorflow as tf
import numpy as np
import os

from options import get_args
from models import get_model
from progress import get_progress_handler, get_valid_progress_handler, TensorboardTracker
from preprocess.data_utils.data_loader import get_data_loader
from learning_rate import get_learning_rate_scheduler
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

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select


def train(args):
    def create_input_sample(datum):
        _label, _protein, _chemical, _nmr = datum
        if args.nmr:
            return (_protein, _chemical), _label
        else:
            return (_protein, _chemical, _nmr), _label

    print("***  Run environment  ***")
    pprint(args)
    print("\n")

    # Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    # Define Placeholders
    result_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='result_pl')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    drug_target_interaction = get_model(args)

    model = drug_target_interaction.create_keras_model()
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

    # Create graph
    learning_rate = get_learning_rate_scheduler(args)

    # Summary
    model.summary()
    progress_handler, tensorboard = get_progress_handler(args)
    tensorboard.info()  # Print Information what now tensorboard is tracking
    validation_handler = get_valid_progress_handler(args)
    test_handler = get_valid_progress_handler(args)

    metrics_names = model.metrics_names
    global_step = 0

    for epoch in range(1, args.epoch + 1):
        model.reset_metrics()
        train_data, valid_data, test_data = get_data_loader(args)
        print(f'Epoch {epoch} start')

        for idx, datum in enumerate(train_data):

            result = model.train_on_batch(*create_input_sample(datum))
            global_step += 1
            if idx % args.log_interval == 0:
                log = "Training: "
                for i in range(len(metrics_names)):
                    log += "{}: {:.3f} | ".format(metrics_names[i], result[i])
                print(log)
                tensorboard.create_summary(global_step, result, model, prefix='train')

        for datum in valid_data:
            result = model.test_on_batch(*create_input_sample(datum), reset_metrics=False)
        log = "Validation: "
        for i in range(len(metrics_names)):
            log += "{}: {:.3f} | ".format(metrics_names[i], result[i])
        print(log)
        tensorboard.create_summary(global_step, result, model, prefix='valid')
        model.reset_metrics()


        for label, protein, chemical, nmr in test_data:
            result = model.test_on_batch((protein, chemical), label, reset_metrics=False)
        log = "Test: "
        for i in range(len(metrics_names)):
            log += "{}: {:.3f} | ".format(metrics_names[i], result[i])
        print(log)
        tensorboard.create_summary(global_step, result, model, prefix='test')
        model.reset_metrics()

if __name__ == '__main__':
    train(get_args())
