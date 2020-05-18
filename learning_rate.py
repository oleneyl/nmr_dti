import math
import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K


def add_learning_rate_option(parser):
    group = parser.add_argument_group(parser)

    group.add_argument('--lr', type=float, default=0.001)

    group.add_argument('--lr_scheduler', type=str, default='base')
    group.add_argument('--warm_up_step_size', type=int, default=20000)

    group.add_argument('--decay_rate', type=int, default=500)
    group.add_argument('--min_lr', type=float, default=1e-7)


def get_learning_rate_scheduler(model, args):
    def get_decay_lr(epoch):
        return args.lr * tf.exp(-1 * epoch / args.decay_rate)

    if args.lr_scheduler == 'triangle':
        return TriangleLearningRateScheduler(model, args)
    elif args.lr_scheduler == 'decay':
        return get_decay_lr
    else:
        return StaticLearningRateScheduler.from_argument(model, args)


class AbstractLearningRateScheduler:
    def __init__(self, model):
        self.model = model
        self._current_learning_rate = 0

    def update_learning_rate(self, epoch, step=None):
        learning_rate = self._learning_rate(epoch, step=step)
        self._current_learning_rate = learning_rate
        K.set_value(self.model.optimizer.learning_rate, learning_rate)

    def _learning_rate(self, epoch, step=None):
        raise NotImplementedError("Learning rate scheduler must implement _learning_rate")

    def report(self):
        return f"Current learning rate : {self._current_learning_rate}"

class StaticLearningRateScheduler(AbstractLearningRateScheduler):
    def __init__(self, model, lr):
        super(StaticLearningRateScheduler, self).__init__(model)
        self.lr = lr

    def _learning_rate(self, epoch, step=None):
        return self.lr

    @classmethod
    def from_argument(cls, model, args):
        return StaticLearningRateScheduler(model, args.lr)


class TriangleLearningRateScheduler(AbstractLearningRateScheduler):
    def __init__(self, model, lr, warm_up_step_size, min_lr):
        super(TriangleLearningRateScheduler, self).__init__(self, model=model)
        self.lr = lr
        self.warm_up_step_size = warm_up_step_size
        self.min_lr = min_lr

    def _learning_rate(self, epoch, step=None):
        float_step = float(epoch)
        warm_up_step_size = self.warm_up_step_size
        warm_up_rate = self.lr * float_step / warm_up_step_size
        decay_rate = max(self.min_lr, self.lr * (2 * warm_up_step_size - float_step) / warm_up_step_size)
        return min(warm_up_rate, decay_rate)

    @classmethod
    def from_argument(cls, model, args):
        return TriangleLearningRateScheduler(model, args.lr, args.warm_up_step_size, args.min_lr)
