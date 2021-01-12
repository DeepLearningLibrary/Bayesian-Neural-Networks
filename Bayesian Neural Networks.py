# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:08:40 2021

@author: Grant
"""

import torch
from torch import nn, optim
from matplotlib import pyplot as plt

def set_default(figsize=(10, 10), dpi=100):
    plt.style.use(['dark_background', 'bmh'])
    plt.rc('axes', facecolor='k')
    plt.rc('figure', facecolor='k')
    plt.rc('figure', figsize=figsize, dpi=dpi)

set_default(figsize=(16, 8))

m = 100  # nb of training pairs
x = (torch.rand(m) - 0.5) * 12  # inputs, sampled from -5 to +5
y = x * torch.sin(x)  # targets

non_linear = nn.Tanh
non_linear = nn.ReLU

net = nn.Sequential(
    nn.Dropout(p=0.05),
    nn.Linear(1, 20),
    non_linear(),
    nn.Dropout(p=0.05),
    nn.Linear(20, 20),
    non_linear(),
    nn.Linear(20, 1)
)

criterion = nn.MSELoss()
optimiser = optim.SGD(net.parameters(), lr=0.01, weight_decay=0.00001)

for epoch in range(1000):
    y_hat = net(x.view(-1, 1))
    loss = criterion(y_hat, y.view(-1, 1))
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()

xx = torch.linspace(-15, 15, 1000)

net.eval()

with torch.no_grad():
    plt.plot(xx.numpy(), net(xx.view(-1, 1)).squeeze().numpy(), 'C1')
plt.plot(x.numpy(), y.numpy(), 'oC0')
plt.axis('equal')
plt.ylim([-10, 5])

