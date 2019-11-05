import math
import tensorflow as tf
import numpy as np

def add_learning_rate_option(parser):
    group = parser.add_argument_group(parser)

    group.add_argument('--lr', type=float, default=0.001)

    group.add_argument('--lr_scheduler', type=str, default='base')
    group.add_argument('--warm_up_step_size', type=int, default=20000)

    group.add_argument('--decay_rate', type=int, default=500)
    group.add_argument('--min_lr', type=float, default=1e-7)


def get_learning_rate_scheduler(args):
    batch_size = args.batch_size

    def get_triangle_lr(epoch):
        float_step = tf.cast(epoch, tf.float32)
        warm_up_step_size = tf.constant(args.warm_up_step_size, dtype=tf.float32)
        warm_up_rate = args.lr * tf.cast(float_step, tf.float32) / warm_up_step_size
        decay_rate = tf.maximum(args.min_lr, args.lr * (2 * warm_up_step_size - float_step) / warm_up_step_size)
        return tf.minimum(warm_up_rate, decay_rate)

    def get_decay_lr(epoch):
        return args.lr * tf.exp(-1 * epoch / args.decay_rate)

    if args.lr_scheduler == 'triangle':
        return get_triangle_lr
    elif args.lr_scheduler == 'decay':
        return get_decay_lr
    else:
        return lambda x: args.lr

def get_scheduler(args):
    batch_size = args.batch_size

    def get_triangle_lr(epoch):
        float_step = float(epoch)
        warm_up_step_size = args.warm_up_step_size
        warm_up_rate = args.lr * float_step / warm_up_step_size
        decay_rate = max(args.min_lr, args.lr * (2 * warm_up_step_size - float_step) / warm_up_step_size)
        return min(warm_up_rate, decay_rate)

    def get_decay_lr(epoch):
        return args.lr * np.exp(-1 * epoch / args.decay_rate)

    if args.lr_scheduler == 'triangle':
        return get_triangle_lr
    elif args.lr_scheduler == 'decay':
        return get_decay_lr
    else:
        return lambda x: args.lr