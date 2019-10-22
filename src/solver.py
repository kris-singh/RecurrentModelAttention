#!/usr/bin/env python

import argparse

import torch
from torch.optim import Adam
from torch.utils import tensorboard

from config import cfg
from model import CoreNetwork, GlimpseNetwork
from small_dataset import get_loader
from utils.basic import early_stopping, setup_exp, setup_logger
from utils.checkpoint import Checkpointer
from utils.meter import MetricLogger


def train(cfg, model, loader, optimizer, scheduler, writer):
    loss_logger = MetricLogger()
    acc_logger = MetricLogger()
    val_loss = MetricLogger()
    val_acc = MetricLogger()
    train_loader, val_loader = loader
    wait = 0
    num_epochs = cfg.TRAIN.NUM_EPOCHS
    for epoch_idx in range(0, num_epochs):
        for idx, data in enumerate(train_loader):
            writer_idx = epoch_idx*len(train_loader) + idx
            x = data[0]
            y = data[1]
            init_loc = (0, 0)
            loc = torch.tensor(init_loc).type(torch.float).unsqueeze(dim=0)
            loc = loc.repeat(x.shape[0], 1)
            init_hidden = torch.zeros(x.shape[0], cfg.CORE_NETWORK.HIDDEN_SIZE)
            log_p_locs = []
            baselines = []
            classification_criterion = torch.nn.NLLLoss()
            baseline_criterion = torch.nn.MSELoss()
            pred_y, log_p_locs, baselines = model(x, loc)
            baselines = torch.squeeze(baselines)

            reward = torch.tensor([1 if torch.argmax(pred_y[i]) == y[i] else 0 for i in range(0, len(pred_y))], dtype=torch.float)
            reward = reward.view(len(reward), 1)
            reward = reward.repeat(1, cfg.GLIMPSE_NETWORK.NUM_GLIMPSE)

            baseline_loss = baseline_criterion(baselines, reward) / cfg.TRAIN.BATCH_SIZE
            reinforce_loss = torch.mean((-log_p_locs * (reward - baselines)))
            classification_loss = classification_criterion(pred_y, y)

            total_loss = baseline_loss + classification_loss + reinforce_loss
            accuracy = torch.sum(torch.argmax(pred_y, 1)==y) / (cfg.TRAIN.BATCH_SIZE * 1.0)

            # logger.info(f'Epoch: {epoch_idx}, Batch:{idx}, CLoss: {classification_loss}, Acc:{accuracy}')
            if idx % cfg.SYSTEM.LOG_FREQ == 0:
                print()
                logger.info(f'Epoch: {epoch_idx}, Batch:{idx}, Loss: {total_loss}, Acc:{accuracy}')
                pass

            loss_logger.update(**{'loss': total_loss})
            acc_logger.update(**{'acc': accuracy})
            writer.add_scalar('loss', total_loss, writer_idx)
            writer.add_scalar('acc', accuracy, writer_idx)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        # validate(cfg, model, val_loader, writer_idx, val_loss, val_acc)
        # wait, cgt = early_stopping(cfg, val_loss, wait)
        # if cgt==0:
        #     logger.info(f'Loss Converged, Early Stopping')
        #     logger.info(f'Epoch: {epoch_idx}, Batch:{idx}, Train Loss: {total_loss}, Train Acc:{accuracy}, Val Loss: {val_loss}')
        #     break;
        scheduler.step(total_loss)
        chkpt.save(f'epoch_{epoch_idx}')


def validate(cfg, model, val_loader, writer_idx, val_loss, val_acc):
    total_loss = 0
    model.eval()
    for idx, data in enumerate(val_loader):
        x = data[0]
        y = data[1]
        init_loc = (0, 0)
        loc = torch.tensor(init_loc).type(torch.float).unsqueeze(dim=0)
        loc = loc.repeat(x.shape[0], 1)
        classification_criterion = torch.nn.NLLLoss()
        baseline_criterion = torch.nn.MSELoss()
        pred_y, log_p_locs, baselines = model(x, loc)
        reward = torch.tensor([1 if torch.argmax(pred_y[i]) == y[i] else 0 for i in range(0, len(pred_y))], dtype=torch.float)
        reward = reward.view(len(reward), 1)
        reward = reward.repeat(1, cfg.GLIMPSE_NETWORK.NUM_GLIMPSE)
        baseline_loss = baseline_criterion(baselines, reward)
        reinforce_loss = torch.mean(torch.sum(-log_p_locs * (reward - baselines)))
        classification_loss = classification_criterion(pred_y, y)
        total_loss += (reinforce_loss + baseline_loss + classification_loss) / cfg.VAL.BATCH_SIZE
        val_loss.update(**{'val_loss': total_loss})
        accuracy = torch.sum(torch.argmax(pred_y, 1)==y) / cfg.VAL.BATCH_SIZE
        val_acc.update(**{'val_acc': accuracy})
        logger.info(f'Batch:{idx}, Val Loss: {total_loss}, Val Acc:{accuracy}')
        writer.add_scalar('val_loss', total_loss, writer_idx)
        writer.add_scalar('val_acc', accuracy, writer_idx)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_loc", type=tuple, default=(0, 0))
    parser.add_argument("--clean_run", type=bool, default=True)
    parser.add_argument("--opts", nargs='*')
    args = parser.parse_args()
    opts = args.opts
    if opts:
        cfg.merge_from_list(opts)

    chkpt_dir, log_dir, tb_dir = setup_exp(cfg.SYSTEM.SAVE_ROOT, cfg.SYSTEM.EXP_NAME, args.clean_run)
    print(f'chkpr_dir:{chkpt_dir}, log_dir:{log_dir}, tb_dir:{tb_dir}')

    writer = tensorboard.SummaryWriter(log_dir=tb_dir)
    logger = setup_logger(cfg.SYSTEM.EXP_NAME, log_dir, 0)

    logger.info(f'cfg: {str(cfg)}')

    glimpse_network = GlimpseNetwork(cfg)
    model = CoreNetwork(cfg, glimpse_network)
    opt_params = cfg.TRAIN.OPTIMIZER[cfg.TRAIN.OPTIMIZER.NAME.upper()]
    schd_params = cfg.TRAIN.SCHEDULER[cfg.TRAIN.SCHEDULER.NAME.upper()]
    optimizer = getattr(torch.optim, cfg.TRAIN.OPTIMIZER.NAME)(model.parameters(), **opt_params)
    scheduler = getattr(torch.optim.lr_scheduler, cfg.TRAIN.SCHEDULER.NAME)(optimizer, **schd_params)
    chkpt = Checkpointer(model, optimizer, scheduler, chkpt_dir, save_to_disk=True, logger=logger)

    train_loader= get_loader(cfg, 'train')
    test_loader = get_loader(cfg, 'test')
    val_loader = get_loader(cfg, 'val')

    chkpt.load()
    loader = [train_loader, val_loader]
    train(cfg, model, loader, optimizer, scheduler, writer)
