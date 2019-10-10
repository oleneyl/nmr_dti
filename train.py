import tensorflow as tf
import numpy as np
import os

from options import get_args
from models import get_model
from progress import get_progress_handler, get_valid_progress_handler, TensorboardTracker
from preprocess.data_utils.data_loader import get_data_loader
from learning_rate import get_learning_rate_scheduler
from pprint import pprint


def train(args):
    print("Run environment")
    pprint(args)

    # Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    # Define Placeholders
    result_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='result_pl')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    drug_target_interaction = get_model(args)

    model = drug_target_interaction.create_keras_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                        loss='mean_squared_error',
                        metrics=['mean_squared_error'])

    # Create graph
    learning_rate = get_learning_rate_scheduler(args)

    # Summary
    model.summary()

    progress_handler, summary_handler = get_progress_handler(args)
    validation_handler = get_valid_progress_handler(args)
    test_handler = get_valid_progress_handler(args)

    metrics_names = model.metrics_names

    for epoch in range(1, args.epoch + 1):
        model.reset_metrics()
        train_data, valid_data, test_data = get_data_loader(args)
        print(f'Epoch {epoch} start')

        for idx, (label, protein, chemical, nmr) in enumerate(train_data):
            result = model.train_on_batch((protein, chemical), label)

            if idx % args.log_interval == 0:
                print("Training: ",
                      "{}: {:.3f}".format(metrics_names[0], result[0]))

        for label, protein, chemical, nmr in valid_data:
            result = model.test_on_batch((protein, chemical), label, reset_metrics=False)
        print("Validation: ",
              "{}: {:.3f}".format(metrics_names[0], result[0]))

        for label, protein, chemical, nmr in test_data:
            result = model.test_on_batch((protein, chemical), label, reset_metrics=False)
        print("Test: ",
              "{}: {:.3f}".format(metrics_names[0], result[0]))

        tf.keras.backend.set_value(model.optimizer.lr, learning_rate(epoch))


if __name__ == '__main__':
    train(get_args())
