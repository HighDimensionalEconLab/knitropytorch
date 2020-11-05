from itertools import cycle
from knitro import *
import torch  # torch.from_numpy
import numpy as np
from functools import (
    reduce,
)  # param_len = reduce(lambda x, y: x * y, self.param_shapes[n])
from collections import OrderedDict


def cycle_no_storage(iterable):
    """
    Takes an existing iterator and makes it repeat itself.
    """
    while True:
        for element in iterable:
            yield element


class PyTorchObjective:  # not (object), since that's implied in Python 3
    """
    PyTorch objective function
    Args:
        loss: f: tensor -> positive scalar
        log_epoch(function, optional): a function f(writer, model, data, i) to call to log an epoch
        model (nn.Module): a pytorch module/neural network
        data_loader (iterator): any iterator.  Typically, but not required, to be a pytorch dataloader.
        writer (obj, optional): an object passed directly to the "log_epoch"
    """

    def __init__(
        self,
        loss,  # better than just loss, because of logging
        model,  # ML model
        data_loader,
        writer=None,
        log_epoch=None,
        epoch_log_frequency=None,
        cycle_with_storage=False,
        derivative_free=False,
    ):
        self.loss = loss
        self.model = model  # neural network
        self.data_loader = data_loader
        self.data = None  # e.g. current batch
        self.batch_size = data_loader.batch_size
        self.log_epoch = log_epoch
        self.epoch_log_frequency = epoch_log_frequency
        self.derivative_free = derivative_free

        if cycle_with_storage:
            self.data_iter = cycle(data_loader)
        else:
            self.data_iter = cycle_no_storage(data_loader)

        parameters = OrderedDict(self.model.named_parameters())
        self.param_shapes = {n: parameters[n].size() for n in parameters}
        self.x0 = np.concatenate(
            [parameters[n].data.numpy().ravel() for n in parameters]
        )
        self.writer = writer

        # counters and flags for evaluating function and deritavies.
        self.step_num = 0  # start at 0 and then iterate forward
        self.epoch_num = 0
        self.fun_evaluated = (
            False  # toggle on/off as required for calculating the function.
        )
        self.grad_evaluated = False
        self.cached_x = None

    def unpack_parameters(self, x):
        """optimize.minimize will supply 1D array, chop it up for each parameter."""
        i = 0
        named_parameters = OrderedDict()
        for n in self.param_shapes:
            param_len = reduce(lambda x, y: x * y, self.param_shapes[n])
            # slice out a section of this length
            param = x[i : i + param_len]
            # reshape according to this size, and cast to torch
            param = param.reshape(*self.param_shapes[n])
            named_parameters[n] = torch.from_numpy(param)
            # update index
            i += param_len
        return named_parameters

    def pack_grads(self):
        """pack all the gradients from the parameters in the module into a
        numpy array.
        """
        grads = []
        for p in self.model.parameters():
            grad = p.grad.data.numpy()
            grads.append(grad.ravel())
        return np.concatenate(grads)

    def is_new(self, x):
        # if this is the first thing we've seen
        if self.cached_x is None:
            return True
        else:
            # compare x to cached_x to determine if we've been given a new input
            # x, self.cached_x = np.array(x), np.array(self.cached_x)
            # error = np.abs(x - self.cached_x)
            # return error.max() > 1e-8
            return hash(bytes(x.data)) != hash(bytes(self.cached_x.data))

    def cache_argument(self, x):
        """
        updates counters/flags and loads parameters into the ML model.
        """
        self.step_num += 1
        self.epoch_num = self.step_num // len(self.data_loader)

        # unpack x and load into module
        state_dict = self.unpack_parameters(x)
        self.model.load_state_dict(state_dict)

        # get the data from the dataloader (but wrapped with a cycle).
        self.data = next(self.data_iter)

        # store the raw array as well
        self.cached_x = x
        self.fun_evaluated = False
        self.grad_evaluated = False

    def evaluate_fun(self):
        # doesn't take "x" since already extracted in setup_function
        # zero the gradient unless derivative free.
        if not self.derivative_free:
            self.model.zero_grad()

        # Forward Pass
        self.fun_value = self.loss(self.model, self.data)

        # Cache value
        self.cached_fun = self.fun_value.item()
        self.fun_evaluated = True

    def evaluate_grad(self):
        if self.derivative_free:
            raise NotImplementedError()

        self.fun_value.backward()
        self.cached_fun = self.fun_value.item()
        self.cached_grad = self.pack_grads()
        self.grad_evaluated = True

    def fun(self, x):
        if self.is_new(x):
            self.cache_argument(x)  # stores x and increments counters, resets flags
        if not self.fun_evaluated:  # in case optimizer queries the same point, etc.
            self.evaluate_fun()

        if (
            (self.step_num % len(self.data_loader) == 0)
            and (not self.epoch_log_frequency is None)
            and (self.epoch_num % self.epoch_log_frequency == 0)
        ):
            self.log_epoch(
                self.writer,
                self.model,
                self.data,
                self.epoch_num,
            )

        return self.cached_fun

    def grad(self, x):
        # Note:  optimizers may call "grad" without calling "fun"
        if self.is_new(x):
            self.cache_argument(x)
        if not self.fun_evaluated:  # In case "grad" called without "fun"
            self.evaluate_fun()  # need forward pass
        if not self.grad_evaluated:
            self.evaluate_grad()
        return self.cached_grad

    def eval_f(self, kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALFC:
            print(
                "*** callbackEvalF incorrectly called with eval type %d"
                % evalRequest.type
            )
            return -1
        x = evalRequest.x
        # print(x)
        # Evaluate nonlinear objective
        evalResult.obj = self.fun(np.array(x))

        # Evaluating the Rosenbrock function
        # evalResult.obj = self.test_fun(x)
        return 0

    def eval_g(self, kc, cb, evalRequest, evalResult, userParams):
        if evalRequest.type != KN_RC_EVALGA:
            print(
                "*** callbackEvalGA incorrectly called with eval type %d"
                % evalRequest.type
            )
            return -1
        x = evalRequest.x
        # print("input of x is ", x)
        # Evaluate nonlinear objective
        evalResult.objGrad = self.grad(np.array(x))

        # Evaluating the Rosenbrock function
        # evalResult.objGrad = self.test_grad(x)
        return 0

    #Rosenbrock function
    def test_fun(self, x):
        return (100 * (x[1] - x[0]**2)**2 + (1-x[0])**2)

    def test_grad(self, x):
        grad0 = 400*x[0]**3 - 400*x[0]*x[1] - 2*x[0] - 2
        grad1 = 200 * (x[1] - x[0]**2)

        return np.array([grad0, grad1])
