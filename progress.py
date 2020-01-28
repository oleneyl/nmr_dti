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
    tensorboard_tracker = TensorboardTracker()
    logger = ProgressLogger(tensorboard_tracker)
    return tensorboard_tracker, logger


def get_valid_progress_handler(args):
    return NormalProgressHandler(args, log_interval=-1)


class TensorboardTracker(object):
    def __init__(self):
        self.log_dir_path = f'./.tfLog/{time.time()}'
        os.makedirs(self.log_dir_path)
        self.writer = tf.summary.FileWriter(self.log_dir_path)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

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

    def save_model(self, model, tag):
        model.save_weights(os.path.join(self.log_dir_path, tag))

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
    def __init__(self, args, log_interval=100, tensorboard_tracker=None, use_stdout=True):
        super(NormalProgressHandler, self).__init__(args, log_interval=log_interval)
        self.tensorboard_tracker = tensorboard_tracker
        self.use_stdout = use_stdout

    def print_log(self, log):
        if self.use_stdout:
            print(log)
        if self.tensorboard_tracker is not None:
            print("Using TFTracker reference..")
            with open(os.path.join(self.tensorboard_tracker.log_dir_path, "Log"), "a+") as f:
                f.write(log+"\n")

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
        self.print_log(result_str)

        # Show examples
        if show_examples:
            self.print_log('-- show examples --')
            self.print_log('Labels :', ['%.3f' % float(x) for x in self.logged_stats['__label'][:10]])
            self.print_log('Preds  :', ['%.3f' % x for x in self.logged_stats['__pred'][:10]])
            self.print_log('Accu   :', [self.logged_stats['acc'][:10]])

        super(NormalProgressHandler, self).emit()


class ProgressLogger(object):
    def __init__(self, tensorboard_tracker=None, base_metric='loss', metric_polarity=-1):
        self.tensorboard_tracker = tensorboard_tracker
        self.base_metric = base_metric
        self.history = defaultdict(list)
        self.best_result = {}
        self.metric_polarity = metric_polarity

    def print_log(self, log):
        print(log)
        if self.tensorboard_tracker is not None:
            with open(os.path.join(self.tensorboard_tracker.log_dir_path, "Log"), "a+") as f:
                f.write(log+"\n")

    def report(self, prefix, parsed_result):
        is_best = False
        if prefix not in self.best_result or \
                self.metric_polarity * (self.history[prefix][self.best_result[prefix]][self.base_metric] -
                                        parsed_result[self.base_metric]) < 0:
            self.best_result[prefix] = len(self.history[prefix])
            is_best = True
        self.history[prefix].append(parsed_result)
        return is_best

    def _parse(self, parsed_result):
        log = ""
        for metrics_names, result in parsed_result.items():
            log += "{}: {:.3f} | ".format(metrics_names, result)
        return log

    def best(self, prefix):
        log = f"best: {self.best_result[prefix]} : "
        log += self._parse(self.history[prefix][self.best_result[prefix]])
        self.print_log(log)

    def emit_history(self, prefix, index):
        self.print_log(self._parse(self.history[prefix][index]))

    def best_index(self, prefix):
        return self.best_result[prefix]

    def emit(self, prefix, metrics_names, result):
        parsed_result = {}
        log = f"{prefix}: "
        for i in range(len(metrics_names)):
            parsed_result[metrics_names[i]] = result[i]
        log += self._parse(parsed_result)
        self.print_log(log)
        return self.report(prefix, parsed_result)
