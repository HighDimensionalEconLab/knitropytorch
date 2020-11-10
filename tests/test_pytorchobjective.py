import os, sys, inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import knitropytorch
from knitropytorch import PyTorchObjective
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

torch.set_default_dtype(torch.float64)

from scipy.optimize import minimize, check_grad

from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_almost_equal,
    assert_equal,
)

from knitro import *


def quadratic_function(X):
    a = 1.0
    b = 1.0
    c = 0.01
    return a * (X ** 2) + b * X + c


def loss(model, X):
    residuals = model(X) - quadratic_function(X)
    return (residuals ** 2).sum() / len(residuals)


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

    torchCGen = torch.random.manual_seed(1235)
    data = torch.rand(1000, 1)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1000)
    obj = PyTorchObjective(loss, net, data_loader)
    xL = minimize(obj.fun, obj.x0, method="BFGS", jac=obj.grad)
    obj.cache_argument(xL.x)

    check_gradient = check_grad(obj.fun, obj.grad, obj.x0)
    assert check_gradient < 1e-6

    assert 1 == 1

    print(xL)


# This is a simple test for knitro. nn and obj are defined like previous test, however this test
# does not use any of these structure. This test calls a simple quadratic functions from  PyTorchObjective for
# objective and gradient.
def test_knitro_simple():
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

    print(torch.rand(10, 1))

    torchCGen = torch.random.manual_seed(1235)
    data = torch.rand(1000, 1)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1000)
    obj = PyTorchObjective(loss, net, data_loader)

    try:
        kc = KN_new()
    except:
        print("Failed to find a valid license.")
        quit()

    KN_add_vars(kc, 2)

    KN_set_var_primal_init_values(kc, xInitVals=[4, 5])
    cb = KN_add_eval_callback(kc, evalObj=True, funcCallback=obj.eval_f_test)
    KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE, gradCallback=obj.eval_g_test)
    KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)
    nStatus = KN_solve(kc)
    sol = Solution(kc)

    assert_array_almost_equal(sol.x, np.array([0.0, 0.0]))

    print(nStatus)

    print("objective is", sol.obj)
    print("x is", sol.x)
    KN_free(kc)


def test_knitro():
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

    torchCGen = torch.random.manual_seed(1235)
    data = torch.rand(1000, 1)
    data_loader = torch.utils.data.DataLoader(data, batch_size=1000)
    obj = PyTorchObjective(loss, net, data_loader)

    try:
        kc = KN_new()
    except:
        print("Failed to find a valid license.")
        quit()

    KN_add_vars(kc, 5)

    KN_set_var_primal_init_values(kc, xInitVals=obj.x0)
    cb = KN_add_eval_callback(kc, evalObj=True, funcCallback=obj.eval_f)
    KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE, gradCallback=obj.eval_g)
    KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)
    # KN_set_int_param(kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_FIRST)
    nStatus = KN_solve(kc)
    sol = Solution(kc)

    print(sol.x)
    # assert_array_almost_equal(sol.x, np.array([0.0, 0.0]))

    print(nStatus)

    # The objective that scipy optimizer returns is around 0.005
    assert sol.obj - 0.005 < 0.001
    KN_free(kc)
