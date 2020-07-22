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

        # Device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)

    # Create model
    model_fitting_information = get_moleculenet_task_dependent_arguments(args.moleculenet_task)
    nmr_interaction = InferenceAtomicNet(args,
                                         label_count=model_fitting_information['label_count'],
                                         is_regression=model_fitting_information['is_regression'])

    model = nmr_interaction.create_keras_model()

    # Learning rate
    learning_rate_scheduler = get_learning_rate_scheduler(model, args)

    # Compile model with metrics
    if model_fitting_information['is_regression']:
        print('--- Regression Task ---')
        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss=tf.keras.losses.MSE,
                      metrics=[tf.keras.losses.MSE])
    else:
        metrics = ['accuracy',
                   tf.keras.metrics.Precision(name='precision'),
                   tf.keras.metrics.Recall(name='recall'),
                   tf.keras.metrics.FalsePositives(name='false_positives'),
                   tf.keras.metrics.FalseNegatives(name='false_negatives')]
        if model_fitting_information['label_count'] == 1:
            metrics += [auroc]

        model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=metrics)

    # Summary
    model.summary()
    tensorboard, logger = get_progress_handler(args)
    tensorboard.info()  # Print Information what now tensorboard is tracking

    logger.print_log(str(args))
    metrics_names = model.metrics_names
    global_step = 0

    # Experiment
    for epoch in range(1, args.epoch + 1):
        model.reset_metrics()
        # TODO data loader must be changed
        train_data = get_dataset_loader(args.moleculenet_task, 'train', batch_size=args.batch_size,
                                        chemical_sequence_length=args.chemical_sequence_length)
        valid_data = get_dataset_loader(args.moleculenet_task, 'valid', batch_size=args.batch_size,
                                        chemical_sequence_length=args.chemical_sequence_length)
        test_data = get_dataset_loader(args.moleculenet_task, 'test', batch_size=args.batch_size,
                                       chemical_sequence_length=args.chemical_sequence_length)

        print(f'Epoch {epoch} start')
        learning_rate_scheduler.update_learning_rate(epoch)
        logger.print_log(learning_rate_scheduler.report())

        # Train
        for idx, datum in enumerate(train_data):
            xs, ys = create_input_sample(datum)
            # print(ys[:10])
            # print([x[0] for x in xs])
            global_step += 1
            if idx % args.log_interval != 0:
                result = model.train_on_batch(xs, ys, reset_metrics=False)
            else:
                result = model.train_on_batch(xs, ys, reset_metrics=True)
                # print([(x[0],'\n') for x in xs])
                logger.emit("Training", metrics_names, result)

                if not model_fitting_information['is_regression']:
                    corrected_mean, filtered_mean, output = multi_auroc(ys, model.predict(xs),
                                                                        model_fitting_information['label_count'])
                    logger.print_log(f"AUC-ROC : corrected | {corrected_mean}, filtered | {filtered_mean}")

                tensorboard.create_summary(global_step, result, model, prefix='train')

        # Validation / Test
        for dataset, set_type in ((valid_data, 'valid'), (test_data, 'test')):
            if set_type == 'test' and args.skip_test:
                continue
            for datum in dataset:
                xs, ys = create_input_sample(datum)
                result = model.test_on_batch(xs, ys, reset_metrics=False)
            is_best = logger.emit(set_type, metrics_names, result)
            if is_best and args.save_best:
                tensorboard.save_model(model, 'best')
            tensorboard.create_summary(global_step, result, model, prefix=set_type)
            model.reset_metrics()

        logger.best("valid")

    logger.emit_history("test", logger.best_index("valid"))
    tensorboard.save_model(model, 'last')


if __name__ == '__main__':
    train(get_args())
