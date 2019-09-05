import tensorflow as tf
import numpy as np
import os

from options import get_args
from models import get_model
from progress import get_progress_handler, get_valid_progress_handler, TensorboardTracker
from preprocess.data_utils.data_loader import get_data_loader


def get_optimizer(args, logits, labels, global_step):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    # Batch normalization
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        grads = optimizer.compute_gradients(loss)
        clipped_grads = [(tf.clip_by_value(grad, -1 * args.grad_clip,  args.grad_clip), var) for grad, var in grads
                         if grad is not None]
        optim_op = optimizer.apply_gradients(clipped_grads, global_step=global_step)

    return loss, optim_op


def train(args):
    print(args)
    # Define Placeholders
    protein_input = tf.placeholder(shape=[None, args.protein_sequence_length], dtype=tf.int32, name='protein_input')
    nmr_input = tf.placeholder(shape=[None, args.nmr_array_size], dtype=tf.float32, name='nmr_input')
    smile_input = tf.placeholder(shape=[None, args.chemical_sequence_length], dtype=tf.int32, name='smile_input')
    result_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='result_pl')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Create graph
    is_train = tf.placeholder(shape=[], dtype=tf.bool, name='trainable')
    binary_result, model = get_model(args, protein_input, nmr_input, smile_input, is_train=is_train)
    loss_op, optimize_op = get_optimizer(args, binary_result, result_pl, global_step)

    var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size
                 for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    print('Variable size', sum(var_sizes) / (1024 ** 2), 'MB')

    initializer = tf.initializers.global_variables()

    progress_handler = get_progress_handler(args)
    validation_handler = get_valid_progress_handler(args)

    # Tensorboard tracking
    summary_handler = TensorboardTracker(args.log_interval)
    for variable in tf.trainable_variables():
        summary_handler.hist(variable.name, variable)
    summary_handler.track('loss', loss_op)
    summary_handler.create_summary(global_step)

    sess = tf.Session()
    sess.run(initializer)
    summary_handler.fix_summary(sess)

    print('Training start!')

    def train_step(data_loader, progress, training_mode='train', manual_event_tide=1000, summary=None):
        _is_train = (training_mode == 'train')
        for idx, (result, protein, smile, nmr) in enumerate(data_loader):
            if _is_train:
                output, loss, _ = sess.run([binary_result, loss_op, optimize_op], feed_dict={
                    protein_input: protein,
                    nmr_input: nmr,
                    smile_input: smile,
                    result_pl: result,
                    is_train: True
                })
            else:
                output, loss = sess.run([binary_result, loss_op], feed_dict={
                    protein_input: protein,
                    nmr_input: nmr,
                    smile_input: smile,
                    result_pl: result,
                    is_train: False
                })
            progress.log({
                'loss': loss,
                'acc': [1 if (r - 0.5) * (o - 0.5) > 0 else 0 for r, o in zip(result, output)]
            })
            if summary:
                summary.create_summary(sess.run(global_step), feed_dict={
                    protein_input: protein,
                    nmr_input: nmr,
                    smile_input: smile,
                    result_pl: result,
                    is_train: False
                })
            if (idx +1) % manual_event_tide == 0:
                # Define manula events
                print('--- Displaying model output inspection ---')
                result = sess.run(model.inspect_model_output(), feed_dict={
                    protein_input: protein,
                    nmr_input: nmr,
                    smile_input: smile,
                    result_pl: result,
                    is_train: False
                })
                for x, name in zip(result, ['nmr','protein','chemical']):
                    print(name, x[0])

    for epoch in range(1, args.epoch+1):
        print(f'Epoch {epoch} start')
        train_data, valid_data = get_data_loader(args)

        # Training
        train_step(train_data, progress_handler, training_mode='train', summary=summary_handler)
        progress_handler.emit()
        # Validation
        train_step(valid_data, validation_handler, training_mode='valid')
        print('--VALIDATION RESULT--')
        validation_handler.emit()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train(get_args())
