"""
progress.py :: Monitor result, handle tensorboard, save best models.
"""
from collections import defaultdict
import time
import tensorflow as tf
import sklearn.metrics
import os


def add_progress_args(parser):
    group = parser.add_argument_group('progress')

    group.add_argument('--log_interval', type=int, default=100)


def get_progress_handler(args):
    return NormalProgressHandler(args, args.log_interval), TensorboardTracker()


def get_valid_progress_handler(args):
    return NormalProgressHandler(args, log_interval=-1)


class TensorboardTracker(object):
    def __init__(self):
        self.log_dir_path = f'./.tfLog/{time.time()}'
        os.makedirs(self.log_dir_path)
        self.writer = tf.summary.FileWriter(self.log_dir_path)

    def info(self):
        print("***  Tensorboard API is tracking output from model!  ***")
        print(f"Watching at :: {self.log_dir_path}\n")

    @staticmethod
    def named_logs(model, logs, prefix=''):
        result = {}
        for name, log in zip(model.metrics_names, logs):
            result[os.path.join(prefix, name)] = log
        return result

    def create_summary(self, global_step, logs, model, prefix=''):
        named_logs = TensorboardTracker.named_logs(model, logs, prefix=prefix)
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=name, simple_value=named_logs[name]) for name in named_logs
        ])
        self.writer.add_summary(summary, global_step=global_step)


class ProgressHandler(object):
    RESERVED_KWD = ['input_sample']
    def __init__(self, args, log_interval=100):
        self.args = args
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
    def emit(self, show_examples=False):
        result_str = f'At step {self.step} | Elapsed time {"%.3f"%self.elapsed_time()}|'
        for k, v in self.logged_stats.items():
            if k[0:2] != '__':
                value = '%.3f' % (sum(v) / len(v))
                result_str += f'{k} : {value}|'

        # Calculate some..good thing.. ROC curve
        if not self.args.as_score:
            auc = sklearn.metrics.roc_auc_score(self.logged_stats['__label'], self.logged_stats['__pred'])
            result_str += f'auc : %.3f|' % auc
        print(result_str)

        # Show examples
        if show_examples:
            print('-- show examples --')
            print('Labels :', ['%.3f' % float(x) for x in self.logged_stats['__label'][:10]])
            print('Preds  :', ['%.3f' % x for x in self.logged_stats['__pred'][:10]])
            print('Accu   :', [self.logged_stats['acc'][:10]])

        super(NormalProgressHandler, self).emit()
