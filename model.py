import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class A3C(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(A3C, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.gru = nn.GRUCell(800, 256)

        num_outputs = action_space.n
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, num_outputs)


        self.gru.bias_ih.data.fill_(0)
        self.gru.bias_hh.data.fill_(0)

        self.train()

    def forward(self, inputs):
        inputs, hx = inputs
        x = F.relu(self.conv1(inputs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        # print(x.size())

        x = x.view(-1, 800)
        hx = self.gru(x, hx)
        x = hx
        return self.critic_linear(x), self.actor_linear(x), hx
