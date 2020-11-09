import time

from knitro import *
from .PyTorchObjective import *


def fit_knitro(
    loss,
    model,
    train_loader,
    validate_loader=None,  # i.e. simulation
    writer=None,
    log_initial=None,
    log_epoch=None,
    log_final=None,
    log_settings=None,  # kwargs for log functions
    epoch_log_frequency=1,  # logging lives in bellmanProblem => objective
    max_epochs=50,
    tag=None,
    # method="BFGS",
    # options={"disp": True},
):

    if validate_loader is None:
        validate_loader = train_loader

    if log_initial or log_final:
        validation_data = next(iter(validate_loader))  # will not step the iterator

    if not log_initial is None:
        log_initial(writer, model, validation_data)

    obj = PyTorchObjective(
        loss,
        model,
        train_loader,
        writer=writer,
        log_epoch=log_epoch,
        epoch_log_frequency=epoch_log_frequency,
    )

    maxiter = max_epochs * len(train_loader)  # len is batches per epoch
    options.update({"maxiter": maxiter})

    t0 = time.time()

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
    xL = Solution(kc)
    KN_free(kc)

    t1 = time.time()

    # cache final x and reconstitute model
    obj.cache_argument(xL.x)

    if not log_final is None:
        log_final(writer, model, validation_data)

    return obj, xL, t1 - t0
