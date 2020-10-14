import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import knitropytorch
from knitropytorch import PyTorchObjective
import torch
import torch.nn as nn
from collections import OrderedDict

torch.set_default_dtype(torch.float64)

from scipy.optimize import minimize

a = 1.0
b = 1.0
c = 0.01


def quadratic_function(X):
    return a * (X ** 2) + b * X + c


def loss(model, X):
    return torch.sum((model(X) - quadratic_function(X)) ** 2) / len(X)


def test_pytorch_obj():
    net = nn.Sequential(
        OrderedDict(
            [
                ("fc1", nn.Linear(1, 2, bias=False)),
                ("sigmoid", nn.Sigmoid()),
                ("fc2", nn.Linear(2, 1)),
            ]
        )
    )
    with torch.no_grad():
        net.fc1.weight[0][0] = 0.0
        net.fc1.weight[1][0] = 0.0
        net.fc2.weight[0][0] = 1.0
        net.fc2.weight[0][1] = 1.0
        net.fc2.bias[0] = 0.00

    data = torch.rand(1000, 1)
    data_loader = torch.utils.data.DataLoader(data)
    obj = PyTorchObjective(loss, net, data_loader)
    xL = minimize(obj.fun, obj.x0, method="BFGS", jac=obj.grad)
    obj.cache_argument(xL.x)
    assert 1 == 1
