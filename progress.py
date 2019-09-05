"""
progress.py :: Monitor result, handle tensorboard, save best models.
"""
from collections import defaultdict
import time
import tensorflow as tf
import os

def add_progress_args(parser):
    group = parser.add_argument_group('progress')

    group.add_argument('--log_interval', type=int, default=100)


def get_progress_handler(args):
    return NormalProgressHandler(args.log_interval), TensorboardTracker(log_interval=args.log_interval)


def get_valid_progress_handler(args):
    return NormalProgressHandler(log_interval = -1)


class TensorboardTracker(object):
    def __init__(self, log_interval=100):
        self._graph_fixed = False
        self.log_dir_path = f'./.tfLog/{time.time()}'
        os.makedirs(self.log_dir_path)
        self.log_interval = log_interval

    def track(self, name, value):
        if self._graph_fixed:
            raise ValueError('Tracking additional variable after fixing summary is not available.')
        tf.summary.scalar(name, value)

    def hist(self, name, value):
        tf.summary.histogram(name, value)

    def fix_summary(self, sess):
        self.summary_op = tf.summary.merge_all()
        self._graph_fixed = True
        self.writer = tf.summary.FileWriter(self.log_dir_path)
        self.writer.add_graph(sess.graph)
        self.sess = sess

    def create_summary(self, global_step, feed_dict={}):
        if global_step % self.log_interval == 0:
            summary = self.sess.run(self.summary_op, feed_dict=feed_dict)
            self.writer.add_summary(summary, global_step=global_step)

class ProgressHandler(object):
    RESERVED_KWD = ['input_sample']
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.global_step = 0
        self.reset_progress()
        self.step = 0
        self.input_sample = []

    def reset_progress(self):
        self.logged_stats = defaultdict(list)
        self.start_time = time.time()

    def log(self, stats):
        for k, v in stats.items():
            if k == 'input_sample':
                self.input_sample = v
            else:
                if isinstance(v, list):
                    self.logged_stats[k] += v
                else:
                    self.logged_stats[k].append(v)
        self.step += 1
        self.global_step += 1

        if self.step % self.log_interval == 0 and self.log_interval > 0:
            self.emit()

    def elapsed_time(self):
        return time.time() - self.start_time

    def emit(self):
        self.reset_progress()

    def flush(self):
        self.step = 0


class NormalProgressHandler(ProgressHandler):
    def emit(self):
        result_str = f'At step {self.step} | Elapsed time {"%.3f"%self.elapsed_time()}|'
        for k, v in self.logged_stats.items():
            value = '%.3f' % (sum(v) / len(v))
            result_str += f'{k} : {value}|'

        print(result_str)
        super(NormalProgressHandler, self).emit()
