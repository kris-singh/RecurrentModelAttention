import logging
import os
import sys
import shutil
import time

class EarlyStopping:
    def __init__(self, cfg):
        self.cfg = cfg
        self.best_loss = None
        self.wait = 0
        self.paitence = cfg.SOLVER.PAITENCE
        self.converged = False

    def reset(self):
        self.wait = 0
        self.paitence = False

    def is_converged(self, val_loss):
        val_loss = val_loss.meters['val_loss'].avg
        if self.best_loss is None:
            self.best_loss = val_loss
        if self.best_loss < val_loss:
            if self.wait > self.paitence:
                self.converged = True
            else:
                self.wait += 1
        else:
            self.best_loss = val_loss

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

def setup_exp(save_dir, exp_name, clean_run):
    if not os.path.exists(os.path.join(save_dir, exp_name)):
        os.makedirs(os.path.join(save_dir, exp_name))
    base_name = os.path.join(save_dir, exp_name)
    # runs
    tb_root = os.path.join(save_dir, 'runs')
    tb_dir = os.path.join(tb_root, exp_name)
    log_dir = os.path.join(base_name, 'logs')
    chkpt_dir = os.path.join(base_name, 'chkpt')

    if clean_run:
        if os.path.exists(tb_dir):
            shutil.rmtree(tb_dir)
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        if os.path.exists(chkpt_dir):
            shutil.rmtree(chkpt_dir)


    if not os.path.exists(os.path.join(tb_root, exp_name)):
        os.makedirs(os.path.join(tb_root, exp_name))
    tb_dir = os.path.join(tb_root, exp_name)
    # logs
    if not os.path.exists(os.path.join(base_name, 'logs')):
        os.makedirs(os.path.join(base_name, 'logs'))
    log_dir = os.path.join(base_name, 'logs')
    # chkpoint
    if not os.path.exists(os.path.join(base_name, 'chkpt')):
        os.makedirs(os.path.join(base_name, 'chkpt'))
    chkpt_dir = os.path.join(base_name, 'chkpt')
    return chkpt_dir, log_dir, tb_dir
