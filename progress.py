"""
progress.py :: Monitor result, handle tensorboard, save best models.
"""
from collections import defaultdict


def add_progress_args(parser):
    group = parser.add_argument_group('progress')

    group.add_argument('--log_interval', type=int, default=100)


def get_progress_handler(args):
    return NormalProgressHandler(args.log_interval)


class ProgressHandler(object):
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.global_step = 0
        self.reset_progress()

    def reset_progress(self):
        self.step = 0
        self.logged_stats = defaultdict(list)

    def log(self, stats):
        for k, v in stats.items():
            self.logged_stats[k] += v
            self.step += 1

        if self.step % self.log_interval == 0:
            self.emit()

    def emit(self):
        self.reset_progress()


class NormalProgressHandler(ProgressHandler):
    def emit(self):
        result_str = ''
        for k, v in self.logged_stats.items():
            result_str += f'{k} : {sum(v)}|'

        print(result_str)
        super(NormalProgressHandler, self).emit()