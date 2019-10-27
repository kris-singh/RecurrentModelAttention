#!/usr/bin/env python

import argparse
import glob
import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from glimpse import GlimpseSensor
from torch.distributions.multivariate_normal import MultivariateNormal

from config import cfg


class BackwardFnc:
    def __init__(self):
        self.idx = 0
    def backward_fnc(self, m, i, o):
        print("=============Backward=========================")
        print(f"Module:{m}")
        self.idx +=1
        print(f"Backward called {self.idx}")
        print("----------------------------------------------")

class ForwardFnc:
    def __init__(self):
        self.idx = 0
    def forward_fnc(self, m, i, o):
        print("----------------Forward---------------------")
        print(f"Module: {m}")
        self.idx +=1
        print(f"Forward called {self.idx}")
        print("----------------------------------------------")


backward = BackwardFnc()
forward = ForwardFnc()

class GlimpseNetwork(torch.nn.Module):
    def __init__(self, cfg):
        super(GlimpseNetwork, self).__init__()
        self.width = cfg.GLIMPSE_SENSOR.WIDTH
        self.num_patches = cfg.GLIMPSE_SENSOR.NUM_SCALES
        self.hidden_size = cfg.GLIMPSE_NETWORK.HIDDEN_SIZE
        self.out_size = cfg.GLIMPSE_NETWORK.OUT_SIZE
        self.num_channels = cfg.GLIMPSE_NETWORK.NUM_CHANNELS
        self.sensor = GlimpseSensor(self.width, self.num_patches)
        self.layer1_1 = nn.Linear(self.num_patches * self.width * self.width * self.num_channels, self.hidden_size)
        self.layer1_2 = nn.Linear(2, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.out_size)

    def forward(self, img: torch.Tensor, location: torch.Tensor):
        """
        Args
        ---
        img: Image. [None, C, H, W]
        location: location inside the image. [None, 2]
        """
        crops = self.sensor.forward(img, location).view(-1).view(img.shape[0], -1)
        locations = location.view(-1).view(location.shape[0], -1)
        x1 = F.relu(self.layer1_1(crops))
        x2 = F.relu(self.layer1_2(location))
        out = F.relu(self.layer2(x1) + self.layer2(x2))
        return out


class CoreNetwork(nn.Module):
    def __init__(self, cfg, glimpse_network):
        super(CoreNetwork, self).__init__()
        self.glimpse_network = glimpse_network
        self.input_size = cfg.GLIMPSE_NETWORK.OUT_SIZE
        self.hidden_size = cfg.CORE_NETWORK.HIDDEN_SIZE
        self.num_glimpses = cfg.GLIMPSE_NETWORK.NUM_GLIMPSE
        self.num_classes = cfg.DATA.NUM_CLASSES
        self.rnn = nn.RNNCell(self.input_size, self.hidden_size, bias=False, nonlinearity='relu')
        self.baseline_nw = nn.Linear(self.hidden_size, 1)
        self.linear_action = nn.Linear(self.hidden_size, self.num_classes)
        self.linear_location = nn.Linear(self.hidden_size, 2)
        self.dist = MultivariateNormal(torch.zeros(2), torch.eye(2)+0.17)
        self.glimpse_network.register_backward_hook(backward.backward_fnc)
        self.glimpse_network.register_forward_hook(forward.forward_fnc)
        self.rnn.register_forward_hook(forward.forward_fnc)
        self.rnn.register_backward_hook(backward.backward_fnc)
        self.baseline_nw.register_forward_hook(forward.forward_fnc)
        self.baseline_nw.register_backward_hook(backward.backward_fnc)
        self.linear_action.register_forward_hook(forward.forward_fnc)
        self.linear_action.register_backward_hook(backward.backward_fnc)
        self.linear_location.register_forward_hook(forward.forward_fnc)
        self.linear_location.register_backward_hook(backward.backward_fnc)


    def forward(self, img, location):
        """
        Args
        ---
        img: Image. [None, 3, H, W]
        location: location inside the image. [None, 2]
        """
        log_p_locs, baselines=[], []
        hidden_state = torch.zeros(cfg.SOLVER.BATCH_SIZE, self.hidden_size)
        # hidden_state: [None, rnn_hidden_size]
        for i in range(0, self.num_glimpses):
            x = self.glimpse_network(img, location)
            # x: [None, GLIMPSE_NETWORK.OUT_SIZE]
            hidden_state = self.rnn(x, hidden_state)
            # Ignore action till last step
            # _ = self.linear_action(hidden_state)
            # mean: [None, 2]
            mean = torch.tanh(self.linear_location(hidden_state))

            # location: [None, 2]
            self.dist.loc = mean
            location = torch.tanh(self.dist.sample())
            #log_p_loc: [None, 1]
            log_p_loc= torch.unsqueeze(self.dist.log_prob(location), 1)
            #baseline: [None, 1]
            baseline = F.relu(self.baseline_nw(hidden_state.detach()))
            #log_p_locs: list of size num_glimpses, (None, 1)
            log_p_locs.append(log_p_loc)
            #baselines: list of size num_glimpses, (None, 1)
            baselines.append(baseline)

        action = F.log_softmax(self.linear_action(hidden_state), dim=1)
        log_p_locs = torch.stack(log_p_locs, 2).squeeze(dim=1)
        baselines = torch.stack(baselines, 2).squeeze(dim=1)
        if self.rnn.weight_ih.grad is not None:
            print("Grad Norm i2h", torch.norm(self.rnn.weight_ih.grad))
        if self.rnn.weight_hh.grad is not None:
            print("Grad Norm h2h", torch.norm(self.rnn.weight_hh.grad))
        print(f"Grad norm linear action {torch.norm(self.linear_action.weight)}")
        return action, log_p_locs, baselines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_loc", type=tuple, default=(0, 0))
    parser.add_argument("--opts", nargs='*')
    args = parser.parse_args()
    opts = args.opts
    if opts:
        cfg.merge_from_list(opts)

    glimpse_network = GlimpseNetwork(cfg)
    core_network = CoreNetwork(cfg, glimpse_network)
    img_path = os.path.join(os.curdir, 'test_data')
    files = glob.glob(os.path.join(img_path, "*.jpg"))
    images = []
    for file_name in files:
        img = Image.open(file_name)
        img = transforms.Resize((128, 128))(img)
        img = transforms.ToTensor()(img)
        images.append(img)
    images = torch.stack(images, dim=0)
    loc = torch.tensor(args.start_loc).type(torch.float).unsqueeze(dim=0)
    loc = loc.repeat(images.shape[0], 1)
    action, log_p_locs, baseline = core_network(images, loc)
    print('Class', action)
    print('Baseline:', baseline)
    print('log_p_locs:', log_p_locs)

if __name__ == "__main__":
    main()
