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

from scipy.optimize import minimize

from numpy.testing import (
    assert_array_almost_equal,
    assert_array_equal,
    assert_almost_equal,
    assert_equal,
)

from knitro import *

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

    torchCGen = torch.random.manual_seed(12345)
    data = torch.rand(1000, 1)
    data_loader = torch.utils.data.DataLoader(data)
    obj = PyTorchObjective(loss, net, data_loader)
    xL = minimize(obj.fun, obj.x0, method="BFGS", jac=obj.grad)
    obj.cache_argument(xL.x)

    print(xL)
    # f_val = obj.fun(obj.x0)
    # grad_val = obj.grad(obj.x0)

    print("minimizing the objective function is:", obj.fun(xL.x))

    # print(grad_val)

    # assert_almost_equal(f_val, 0.3102050451060238)
    # assert_array_almost_equal(
    #     grad_val, [0.09091323, 0.09091323, 0.55696054, 0.55696054, 1.11392108]
    # )
    assert 1 == 1


# class TestWrappers:
#     def test_eval_f(self, kc, cb, evalRequest, evalResult, userParams):
#         if evalRequest.type != KN_RC_EVALFC:
#             print(
#                 "*** callbackEvalF incorrectly called with eval type %d"
#                 % evalRequest.type
#             )
#             return -1
#         x = evalRequest.x
#         # here x is a list of size 1
#         # Evaluate nonlinear objective
#         evalResult.obj = self.fun(x[0])
#         return 0

#     def test_eval_g(self, kc, cb, evalRequest, evalResult, userParams):
#         if evalRequest.type != KN_RC_EVALGA:
#             print(
#                 "*** callbackEvalGA incorrectly called with eval type %d"
#                 % evalRequest.type
#             )
#             return -1
#         x = evalRequest.x
#         print(x)
#         # Evaluate nonlinear objective
#         evalResult.objGrad[0] = self.grad(x[0])
#         return 0

#     def fun(self, x):
#         return x * x

#     def grad(self, x):
#         return 2 * x


# def test_fake_class_knitro():
#     try:
#         kc = KN_new()
#     except:
#         print("Failed to find a valid license.")
#         quit()
#     KN_add_vars(kc, 1)
#     fake_inst = TestWrappers()
#     # KN_set_var_primal_init_values (kc, xInitVals = [.5])
#     cb = KN_add_eval_callback(kc, evalObj=True, funcCallback=fake_inst.test_eval_f)
#     KN_set_cb_grad(
#         kc, cb, objGradIndexVars=KN_DENSE, gradCallback=fake_inst.test_eval_g
#     )
#     KN_set_int_param(kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_ALL)
#     nStatus = KN_solve(kc)

#     assert 1 == 1


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

    torchCGen = torch.random.manual_seed(12345) 
    data = torch.rand(1000, 1)
    data_loader = torch.utils.data.DataLoader(data)
    obj = PyTorchObjective(loss, net, data_loader)

    try:
        kc = KN_new ()
    except:
        print ("Failed to find a valid license.")
        quit ()
    KN_add_vars (kc, 5)
    KN_set_var_primal_init_values (kc, xInitVals = obj.x0)
    cb = KN_add_eval_callback(kc, evalObj = True, funcCallback = obj.eval_f)
    KN_set_cb_grad (kc, cb, objGradIndexVars = KN_DENSE, gradCallback = obj.eval_g)
    KN_set_obj_goal (kc, KN_OBJGOAL_MINIMIZE)
    # KN_set_int_param (kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_ALL)
    # KN_set_int_param (kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_ALL)
    nStatus = KN_solve (kc)

    print(Solution(kc))

    assert 1 == 1
    
    # print(obj.fun(np.array([-0.05037972, -0.05037972,  0.71964827,  0.71964827, -0.55942466])))
    # obj.cache_argument(np.array([-0.013799221627639344, -0.013799221627639344, 0.8606748756170269, 0.8606748756170269, -0.278378617923651]))
    # print(obj.fun(np.array([-0.013799221627639344, -0.013799221627639344, 0.8606748756170269, 0.8606748756170269, -0.278378617923651])))
    
    

    sol = Solution(kc)

    print("objective is", sol.obj)
    print("x is", sol.x)

    obj.cache_argument(np.array(sol.x))
    KN_free (kc)

# test_pytorch_obj()
# # test_fake_class_knitro()
# test_knitro()
