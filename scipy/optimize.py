

import inspect
import numpy as np
import operator

_iter = 100
_xtol = 2e-12
_rtol = 4 * np.finfo(float).eps

# Must agree with CONVERGED, SIGNERR, CONVERR, ...  in zeros.h
_ECONVERGED = 0
_ESIGNERR = -1  # used in _chandrupatla
_ECONVERR = -2
_EVALUEERR = -3
_ECALLBACK = -4
_EINPROGRESS = 1

CONVERGED = 'converged'
SIGNERR = 'sign error'
CONVERR = 'convergence error'
VALUEERR = 'value error'
INPROGRESS = 'No error'


flag_map = {_ECONVERGED: CONVERGED, _ESIGNERR: SIGNERR, _ECONVERR: CONVERR,
            _EVALUEERR: VALUEERR, _EINPROGRESS: INPROGRESS}

def _wrap_callback(callback, method=None):
    """Wrap a user-provided callback so that attributes can be attached."""
    if callback is None or method in {'tnc', 'slsqp', 'cobyla', 'cobyqa'}:
        return callback  # don't wrap

    sig = inspect.signature(callback)

    if set(sig.parameters) == {'intermediate_result'}:
        def wrapped_callback(res):
            return callback(intermediate_result=res)
    elif method == 'trust-constr':
        def wrapped_callback(res):
            return callback(np.copy(res.x), res)
    elif method == 'differential_evolution':
        def wrapped_callback(res):
            return callback(np.copy(res.x), res.convergence)
    else:
        def wrapped_callback(res):
            return callback(np.copy(res.x))

    wrapped_callback.stop_iteration = False
    return wrapped_callback

def _indenter(s, n=0):
    split = s.split("\n")
    indent = " "*n
    return ("\n" + indent).join(split)

def _float_formatter_10(x):
    if np.isposinf(x):
        return "       inf"
    elif np.isneginf(x):
        return "      -inf"
    elif np.isnan(x):
        return "       nan"
    return np.format_float_scientific(x, precision=3, pad_left=2, unique=False)
def _dict_formatter(d, n=0, mplus=1, sorter=None):
    if isinstance(d, dict):
        m = max(map(len, list(d.keys()))) + mplus  # width to print keys
        s = '\n'.join([k.rjust(m) + ': ' +  # right justified, width m
                       _indenter(_dict_formatter(v, m+n+2, 0, sorter), m+2)
                       for k, v in sorter(d)])  # +2 for ': '
    else:
        with np.printoptions(linewidth=76-n, edgeitems=2, threshold=12,
                             formatter={'float_kind': _float_formatter_10}):
            s = str(d)
    return s

class _RichResult(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    __setattr__ = dict.__setitem__  # type: ignore[assignment]
    __delattr__ = dict.__delitem__  # type: ignore[assignment]

    def __repr__(self):
        order_keys = ['message', 'success', 'status', 'fun', 'funl', 'x', 'xl',
                      'col_ind', 'nit', 'lower', 'upper', 'eqlin', 'ineqlin',
                      'converged', 'flag', 'function_calls', 'iterations',
                      'root']
        order_keys = getattr(self, '_order_keys', order_keys)
        # 'slack', 'con' are redundant with residuals
        # 'crossover_nit' is probably not interesting to most users
        omit_keys = {'slack', 'con', 'crossover_nit', '_order_keys'}

        def key(item):
            try:
                return order_keys.index(item[0].lower())
            except ValueError:  # item not in list
                return np.inf

        def omit_redundant(items):
            for item in items:
                if item[0] in omit_keys:
                    continue
                yield item

        def item_sorter(d):
            return sorted(omit_redundant(d.items()), key=key)

        if self.keys():
            return _dict_formatter(self, sorter=item_sorter)
        else:
            return self.__class__.__name__ + "()"

    def __dir__(self):
        return list(self.keys())

class OptimizeResult(_RichResult):
    pass


def _call_callback_maybe_halt(callback, res):
    if callback is None:
        return False
    try:
        callback(res)
        return False
    except StopIteration:
        callback.stop_iteration = True
        return True

def _wrap_scalar_function_maxfun_validation(function, args, maxfun):
    ncalls = [0]
    if function is None:
        return ncalls, None

    def function_wrapper(x, *wrapper_args):
        if ncalls[0] >= maxfun:
            raise _MaxFuncCallError("Too many function calls")
        ncalls[0] += 1
        # A copy of x is sent to the user function (gh13740)
        fx = function(np.copy(x), *(wrapper_args + args))
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError("The user-provided objective function "
                                 "must return a scalar value.") from e
        return fx

    return ncalls, function_wrapper

class _MaxFuncCallError(RuntimeError):
    pass
def _minimize_neldermead(func, x0, args=(), callback=None,
                         maxiter=None, maxfev=None, disp=False,
                         return_all=False, initial_simplex=None,
                         xatol=1e-4, fatol=1e-4, adaptive=False, bounds=None,
                         **unknown_options):
    maxfun = maxfev
    retall = return_all

    x0 = np.atleast_1d(x0).flatten()
    dtype = x0.dtype if np.issubdtype(x0.dtype, np.inexact) else np.float64
    x0 = np.asarray(x0, dtype=dtype)

    if adaptive:
        dim = float(len(x0))
        rho = 1
        chi = 1 + 2/dim
        psi = 0.75 - 1/(2*dim)
        sigma = 1 - 1/dim
    else:
        rho = 1
        chi = 2
        psi = 0.5
        sigma = 0.5

    nonzdelt = 0.05
    zdelt = 0.00025

    if bounds is not None:
        lower_bound, upper_bound = bounds.lb, bounds.ub
        # check bounds
        if (lower_bound > upper_bound).any():
            raise ValueError("Nelder Mead - one of the lower bounds "
                             "is greater than an upper bound.")
    if bounds is not None:
        x0 = np.clip(x0, lower_bound, upper_bound)

    if initial_simplex is None:
        N = len(x0)

        sim = np.empty((N + 1, N), dtype=x0.dtype)
        sim[0] = x0
        for k in range(N):
            y = np.array(x0, copy=True)
            if y[k] != 0:
                y[k] = (1 + nonzdelt)*y[k]
            else:
                y[k] = zdelt
            sim[k + 1] = y
    else:
        sim = np.atleast_2d(initial_simplex).copy()
        dtype = sim.dtype if np.issubdtype(sim.dtype, np.inexact) else np.float64
        sim = np.asarray(sim, dtype=dtype)
        if sim.ndim != 2 or sim.shape[0] != sim.shape[1] + 1:
            raise ValueError("`initial_simplex` should be an array of shape (N+1,N)")
        if len(x0) != sim.shape[1]:
            raise ValueError("Size of `initial_simplex` is not consistent with `x0`")
        N = sim.shape[1]

    if retall:
        allvecs = [sim[0]]

    # If neither are set, then set both to default
    if maxiter is None and maxfun is None:
        maxiter = N * 200
        maxfun = N * 200
    elif maxiter is None:
        # Convert remaining Nones, to np.inf, unless the other is np.inf, in
        # which case use the default to avoid unbounded iteration
        if maxfun == np.inf:
            maxiter = N * 200
        else:
            maxiter = np.inf
    elif maxfun is None:
        if maxiter == np.inf:
            maxfun = N * 200
        else:
            maxfun = np.inf

    if bounds is not None:
        msk = sim > upper_bound
        # reflect into the interior
        sim = np.where(msk, 2*upper_bound - sim, sim)
        # but make sure the reflection is no less than the lower_bound
        sim = np.clip(sim, lower_bound, upper_bound)

    one2np1 = list(range(1, N + 1))
    fsim = np.full((N + 1,), np.inf, dtype=float)

    fcalls, func = _wrap_scalar_function_maxfun_validation(func, args, maxfun)

    try:
        for k in range(N + 1):
            fsim[k] = func(sim[k])
    except _MaxFuncCallError:
        pass
    finally:
        ind = np.argsort(fsim)
        sim = np.take(sim, ind, 0)
        fsim = np.take(fsim, ind, 0)

    ind = np.argsort(fsim)
    fsim = np.take(fsim, ind, 0)
    # sort so sim[0,:] has the lowest function value
    sim = np.take(sim, ind, 0)

    iterations = 1

    while (fcalls[0] < maxfun and iterations < maxiter):
        try:
            if (np.max(np.ravel(np.abs(sim[1:] - sim[0]))) <= xatol and
                    np.max(np.abs(fsim[0] - fsim[1:])) <= fatol):
                break

            xbar = np.add.reduce(sim[:-1], 0) / N
            xr = (1 + rho) * xbar - rho * sim[-1]
            if bounds is not None:
                xr = np.clip(xr, lower_bound, upper_bound)
            fxr = func(xr)
            doshrink = 0

            if fxr < fsim[0]:
                xe = (1 + rho * chi) * xbar - rho * chi * sim[-1]
                if bounds is not None:
                    xe = np.clip(xe, lower_bound, upper_bound)
                fxe = func(xe)

                if fxe < fxr:
                    sim[-1] = xe
                    fsim[-1] = fxe
                else:
                    sim[-1] = xr
                    fsim[-1] = fxr
            else:  # fsim[0] <= fxr
                if fxr < fsim[-2]:
                    sim[-1] = xr
                    fsim[-1] = fxr
                else:  # fxr >= fsim[-2]
                    # Perform contraction
                    if fxr < fsim[-1]:
                        xc = (1 + psi * rho) * xbar - psi * rho * sim[-1]
                        if bounds is not None:
                            xc = np.clip(xc, lower_bound, upper_bound)
                        fxc = func(xc)

                        if fxc <= fxr:
                            sim[-1] = xc
                            fsim[-1] = fxc
                        else:
                            doshrink = 1
                    else:
                        # Perform an inside contraction
                        xcc = (1 - psi) * xbar + psi * sim[-1]
                        if bounds is not None:
                            xcc = np.clip(xcc, lower_bound, upper_bound)
                        fxcc = func(xcc)

                        if fxcc < fsim[-1]:
                            sim[-1] = xcc
                            fsim[-1] = fxcc
                        else:
                            doshrink = 1

                    if doshrink:
                        for j in one2np1:
                            sim[j] = sim[0] + sigma * (sim[j] - sim[0])
                            if bounds is not None:
                                sim[j] = np.clip(
                                    sim[j], lower_bound, upper_bound)
                            fsim[j] = func(sim[j])
            iterations += 1
        except _MaxFuncCallError:
            pass
        finally:
            ind = np.argsort(fsim)
            sim = np.take(sim, ind, 0)
            fsim = np.take(fsim, ind, 0)
            if retall:
                allvecs.append(sim[0])
            intermediate_result = OptimizeResult(x=sim[0], fun=fsim[0])
            if _call_callback_maybe_halt(callback, intermediate_result):
                break

    x = sim[0]
    fval = np.min(fsim)
    warnflag = 0

    result = OptimizeResult(fun=fval, nit=iterations, nfev=fcalls[0],
                            status=warnflag, success=(warnflag == 0),
                            message='', x=x, final_simplex=(sim, fsim))
    if retall:
        result['allvecs'] = allvecs
    return result


def fmin(func, x0, args=(), xtol=1e-4, ftol=1e-4, maxiter=None, maxfun=None,
         full_output=0, disp=1, retall=0, callback=None, initial_simplex=None):
    opts = {'xatol': xtol,
            'fatol': ftol,
            'maxiter': maxiter,
            'maxfev': maxfun,
            'disp': disp,
            'return_all': retall,
            'initial_simplex': initial_simplex}

    callback = _wrap_callback(callback)
    res = _minimize_neldermead(func, x0, args, callback=callback, **opts)
    if full_output:
        retlist = res['x'], res['fun'], res['nit'], res['nfev'], res['status']
        if retall:
            retlist += (res['allvecs'], )
        return retlist
    else:
        if retall:
            return res['x'], res['allvecs']
        else:
            return res['x']

class RootResults(OptimizeResult):

    def __init__(self, root, iterations, function_calls, flag, method):
        self.root = root
        self.iterations = iterations
        self.function_calls = function_calls
        self.converged = flag == _ECONVERGED
        if flag in flag_map:
            self.flag = flag_map[flag]
        else:
            self.flag = flag
        self.method = method

def results_c(full_output, r, method):
    if full_output:
        x, funcalls, iterations, flag = r
        results = RootResults(root=x,
                              iterations=iterations,
                              function_calls=funcalls,
                              flag=flag, method=method)
        return x, results
    else:
        return r

def _wrap_nan_raise(f):

    def f_raise(x, *args):
        fx = f(x, *args)
        f_raise._function_calls += 1
        if np.isnan(fx):
            msg = (f'The function value at x={x} is NaN; '
                   'solver cannot continue.')
            err = ValueError(msg)
            err._x = x
            err._function_calls = f_raise._function_calls
            raise err
        return fx

    f_raise._function_calls = 0
    return f_raise
def brentq(f, a, b, args=(),
           xtol=_xtol, rtol=_rtol, maxiter=_iter,
           full_output=False, disp=True):
    if not isinstance(args, tuple):
        args = (args,)
    maxiter = operator.index(maxiter)
    if xtol <= 0:
        raise ValueError(f"xtol too small ({xtol:g} <= 0)")
    if rtol < _rtol:
        raise ValueError(f"rtol too small ({rtol:g} < {_rtol:g})")
    f = _wrap_nan_raise(f)
    r = _zeros._brentq(f, a, b, xtol, rtol, maxiter, args, full_output, disp)
    return results_c(full_output, r, "brentq")
