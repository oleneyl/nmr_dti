"""
progress.py :: Monitor result, handle tensorboard, save best models.
"""
from collections import defaultdict
import time

def add_progress_args(parser):
    group = parser.add_argument_group('progress')

    group.add_argument('--log_interval', type=int, default=100)


def get_progress_handler(args):
    return NormalProgressHandler(args.log_interval)


def get_valid_progress_handler(args):
    return NormalProgressHandler(log_interval = -1)


class ProgressHandler(object):
    def __init__(self, log_interval=100):
        self.log_interval = log_interval
        self.global_step = 0
        self.reset_progress()

    def reset_progress(self):
        self.step = 0
        self.logged_stats = defaultdict(list)
        self.start_time = time.time()

    def log(self, stats):
        for k, v in stats.items():
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


class NormalProgressHandler(ProgressHandler):
    def emit(self):
        result_str = f'Elapsed time {"%.3f"%self.elapsed_time()}|'
        for k, v in self.logged_stats.items():
            value = '%.3f' % (sum(v) / len(v))
            result_str += f'{k} : {value}|'

        print(result_str)
        super(NormalProgressHandler, self).emit()
