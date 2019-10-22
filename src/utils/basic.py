import logging
import os
import sys


def setup_logger(name, save_dir, distributed_rank):
    logger = logging.Logger(name)
    logger.setLevel(logging.DEBUG)
    if distributed_rank > 0:
        return logger
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(formatter)
    sh.setLevel(logging.DEBUG)
    logger.addHandler(sh)

    if save_dir:
        fh = logging.FileHandler(filename=os.path.join(save_dir, "log.txt"), mode='a', encoding='UTF')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def early_stopping(losses, paitence):
    EPS = 1e-4
    if abs(losses[-1] - losses[-2]) <= EPS:
        if wait < paitence:
            wait+=1
            return wait, -1
        else:
            wait = 0
            return wait, 0

def setup_exp(exp_name, save_dir):
    if not os.path.exists(os.path.join(save_dir, exp_name)):
        os.mkdir(os.path.join(save_dir, exp_name))
    base_name = os.path.join(save_dir, exp_name)
    # runs
    tb_root = os.path.join(save_dir, 'runs')
    if not os.path.exists(os.path.join(tb_root, exp_name)):
        tb_dir = os.mkdir(os.path.join(tb_root, exp_name))
    else:
        tb_dir = os.path.join(tb_root, exp_name)
    # logs
    if not os.path.exists(os.path.join(base_name, 'logs')):
        log_dir = os.mkdir(os.path.join(base_name, 'logs'))
    else:
        log_dir = os.path.join(base_name, 'logs')
    # chkpoint
    if not os.path.exists(os.path.join(base_name, 'chkpt')):
        chkpt_dir = os.mkdir(os.path.join(base_name, 'chkpt'))
    else:
        chkpt_dir = os.path.join(base_name, 'chkpt')

    return chkpt_dir, log_dir, tb_dir
