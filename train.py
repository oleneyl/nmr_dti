import tensorflow as tf
import numpy as np
import os

from options import get_args
from models import get_model
from progress import get_progress_handler, get_valid_progress_handler
from preprocess.data_utils.data_loader import get_data_loader


def get_optimizer(args, logits, labels):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optim_op = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)

    return loss, optim_op


def train(args):
    print(args)
    # Define Placeholders
    protein_input = tf.placeholder(shape=[None, args.protein_sequence_length], dtype=tf.int32, name='protein_input')
    nmr_input = tf.placeholder(shape=[None, args.nmr_array_size], dtype=tf.float32, name='nmr_input')
    smile_input = tf.placeholder(shape=[None, args.chemical_sequence_length], dtype=tf.int32, name='smile_input')
    result_pl = tf.placeholder(shape=[None, 1], dtype=tf.float32, name='result_pl')

    # Create graph
    binary_result = get_model(args, protein_input, nmr_input, smile_input)
    loss_op, optimize_op = get_optimizer(args, binary_result, result_pl)

    var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size
                 for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)]
    print('Variable size', sum(var_sizes) / (1024 ** 2), 'MB')

    initializer = tf.initializers.global_variables()

    progress_handler = get_progress_handler(args)
    validation_handler = get_valid_progress_handler(args)

    sess = tf.Session()
    sess.run(initializer)

    print('Training start!')

    def train_step(data_loader, progress):
        for result, protein, smile, nmr in data_loader:
            output, loss, _ = sess.run([binary_result, loss_op, optimize_op], feed_dict={
                protein_input: protein,
                nmr_input: nmr,
                smile_input: smile,
                result_pl: result
            })
            progress.log({
                'loss': loss,
                'acc': [1 if (r - 0.5) * (o - 0.5) > 0 else 0 for r, o in zip(result, output)]
            })

    for epoch in range(1, args.epoch+1):
        print(f'Epoch {epoch} start')
        train_data, valid_data = get_data_loader(args)

        # Training
        train_step(train_data, progress_handler)

        # Validation
        train_step(valid_data, validation_handler)
        print('--VALIDATION RESULT--')
        validation_handler.emit()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    train(get_args())
