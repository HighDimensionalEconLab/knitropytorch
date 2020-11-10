import time

from scipy.optimize import minimize
from .PyTorchObjective import *
from knitro import *


def fit_scipy(
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
    method="BFGS",
    options={"disp": True},
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
    xL = minimize(obj.fun, obj.x0, method=method, jac=obj.grad, options=options)
    t1 = time.time()

    # cache final x and reconstitute model
    obj.cache_argument(xL.x)

    if not log_final is None:
        log_final(writer, model, validation_data)

    return obj, xL, t1 - t0


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
):

    # print("x_old0", old_obj.x0)
   

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
    print("x_new0" ,obj.x0)

    # maxiter = max_epochs * len(train_loader)  # len is batches per epoch
    # options.update({"maxiter": maxiter})

    t0 = time.time()
    xL = knitro_minimize(obj)

    #using optimize function
    # variables = Variables(nV=len(obj.x0),
    #                     # xLoBnds=[-1.0] * len(obj.x0), # not necessary since infinite
    #                     # xUpBnds=[1.0] * len(obj.x0),
    #                     xInitVals=obj.x0
    # )
    # options = {}
    # # options['derivcheck']   = KN_DERIVCHECK_ALL
    # callback = Callback(evalObj=True,
    #                     funcCallback=obj.eval_f,
    #                     objGradIndexVars=KN_DENSE,
    #                     gradCallback=obj.eval_g,
    #                     # hessIndexVars1=KN_DENSE_ROWMAJOR,
    #                     # hessCallback=callbackEvalH,
    #                     # hessianNoFAllow=True
    #                     )
    # xL = optimize(variables=variables,
    #                   callbacks=callback,
    #                   options=options)
    t1 = time.time()

    # obj = old_obj

    # cache final x and reconstitute model
    obj.cache_argument(np.array(xL.x))

    if not log_final is None:
        log_final(writer, model, validation_data)


    return obj, xL, t1 - t0

# Wrapper function for knitro optimizer
def knitro_minimize(obj):
    try:
        kc = KN_new()
    except:
        print("Failed to find a valid license.")
        quit()
    KN_add_vars(kc, len(obj.x0))
    print("initial values",KN_set_var_primal_init_values(kc, xInitVals=obj.x0))
    cb = KN_add_eval_callback(kc, evalObj=True, funcCallback=obj.eval_f)

    # cb = KN_add_eval_callback (kc, evalObj = True, funcCallback = obj.callbackEvalFCGA)
    # KN_set_int_param (kc, KN_PARAM_EVAL_FCGA, KN_EVAL_FCGA_YES)
    KN_set_cb_grad(kc, cb, objGradIndexVars=KN_DENSE, gradCallback=obj.eval_g)
    KN_set_obj_goal(kc, KN_OBJGOAL_MINIMIZE)
    # KN_set_int_param(kc, KN_PARAM_DERIVCHECK, KN_DERIVCHECK_FIRST)
    # KN_set_int_param(kc, "algorithm", 5)
    nStatus = KN_solve(kc)
    xL = Solution(kc)
    KN_free(kc)

    return xL
