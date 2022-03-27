""" Util functions for writing logs
Author: Zhao Na
Date: September, 2020
"""
import os

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def print_args(logger, args):
    opt = vars(args)
    logger.cprint('------------ Options -------------')
    for k, v in sorted(opt.items()):
        logger.cprint('%s: %s' % (str(k), str(v)))
    logger.cprint('-------------- End ----------------\n')


def init_logger(args):
    if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)
    if args.phase == 'test':
        filename_suffix = 'test_%dclass' %args.n_novel_class
    else:
        filename_suffix = args.phase
    log_file = os.path.join(args.log_dir, 'log_%s.txt' %filename_suffix)
    logger = IOStream(log_file)
    # logger.cprint(str(args))
    ## print arguments in format
    print_args(logger, args)
    return logger