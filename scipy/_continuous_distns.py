from collections import namedtuple
import keyword
import numbers
import re
import types
import numpy as np
import inspect
import sys
from numpy import (arange, putmask, ones, shape, ndarray, zeros, floor,
                   logical_and, log, sqrt, place, argmax, vectorize, asarray,
                   nan, inf, isinf, empty)
from numpy import arange, newaxis, hstack, prod, array
from .special import factorial2, digamma, beta, poch, stdtr, stdtrit, log_ndtr, ndtr, ndtri, entr, comb
from ._censored_data import CensoredData
from . import optimize

parse_arg_template = """
def _parse_args(self, %(shape_arg_str)s %(locscale_in)s):
    return (%(shape_arg_str)s), %(locscale_out)s

def _parse_args_rvs(self, %(shape_arg_str)s %(locscale_in)s, size=None):
    return self._argcheck_rvs(%(shape_arg_str)s %(locscale_out)s, size=size)

def _parse_args_stats(self, %(shape_arg_str)s %(locscale_in)s, moments='mv'):
    return (%(shape_arg_str)s), %(locscale_out)s, moments
"""
xp = np
_XMAX = np.finfo(float).max
_LOGXMAX = np.log(_XMAX)

FullArgSpec = namedtuple('FullArgSpec',
                         ['args', 'varargs', 'varkw', 'defaults',
                          'kwonlyargs', 'kwonlydefaults', 'annotations'])
def getfullargspec_no_self(func):
    sig = inspect.signature(func)
    args = [
        p.name for p in sig.parameters.values()
        if p.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.POSITIONAL_ONLY]
    ]
    varargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_POSITIONAL
    ]
    varargs = varargs[0] if varargs else None
    varkw = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.VAR_KEYWORD
    ]
    varkw = varkw[0] if varkw else None
    defaults = tuple(
        p.default for p in sig.parameters.values()
        if (p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and
            p.default is not p.empty)
    ) or None
    kwonlyargs = [
        p.name for p in sig.parameters.values()
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]
    kwdefaults = {p.name: p.default for p in sig.parameters.values()
                  if p.kind == inspect.Parameter.KEYWORD_ONLY and
                  p.default is not p.empty}
    annotations = {p.name: p.annotation for p in sig.parameters.values()
                   if p.annotation is not p.empty}
    return FullArgSpec(args, varargs, varkw, defaults, kwonlyargs,
                       kwdefaults or None, annotations)
_getfullargspec = getfullargspec_no_self



# copy-pasted from scikit-learn utils/validation.py
def check_random_state(seed):
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral | np.integer):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState | np.random.Generator):
        return seed

    raise ValueError(f"'{seed}' cannot be used to seed a numpy.random.RandomState"
                     " instance")

# Frozen RV class
class rv_frozen:

    def __init__(self, dist, *args, **kwds):
        self.args = args
        self.kwds = kwds

        # create a new instance
        self.dist = dist.__class__(**dist._updated_ctor_param())

        shapes, _, _ = self.dist._parse_args(*args, **kwds)
        self.a, self.b = self.dist._get_support(*shapes)

    @property
    def random_state(self):
        return self.dist._random_state

    @random_state.setter
    def random_state(self, seed):
        self.dist._random_state = check_random_state(seed)

    def cdf(self, x):
        return self.dist.cdf(x, *self.args, **self.kwds)

    def logcdf(self, x):
        return self.dist.logcdf(x, *self.args, **self.kwds)

    def ppf(self, q):
        return self.dist.ppf(q, *self.args, **self.kwds)

    def isf(self, q):
        return self.dist.isf(q, *self.args, **self.kwds)

    def rvs(self, size=None, random_state=None):
        kwds = self.kwds.copy()
        kwds.update({'size': size, 'random_state': random_state})
        return self.dist.rvs(*self.args, **kwds)

    def sf(self, x):
        return self.dist.sf(x, *self.args, **self.kwds)

    def logsf(self, x):
        return self.dist.logsf(x, *self.args, **self.kwds)

    def stats(self, moments='mv'):
        kwds = self.kwds.copy()
        kwds.update({'moments': moments})
        return self.dist.stats(*self.args, **kwds)

    def median(self):
        return self.dist.median(*self.args, **self.kwds)

    def mean(self):
        return self.dist.mean(*self.args, **self.kwds)

    def var(self):
        return self.dist.var(*self.args, **self.kwds)

    def std(self):
        return self.dist.std(*self.args, **self.kwds)

    def moment(self, order=None):
        return self.dist.moment(order, *self.args, **self.kwds)

    def entropy(self):
        return self.dist.entropy(*self.args, **self.kwds)

    def interval(self, confidence=None):
        return self.dist.interval(confidence, *self.args, **self.kwds)

    def expect(self, func=None, lb=None, ub=None, conditional=False, **kwds):
        # expect method only accepts shape parameters as positional args
        # hence convert self.args, self.kwds, also loc/scale
        # See the .expect method docstrings for the meaning of
        # other parameters.
        a, loc, scale = self.dist._parse_args(*self.args, **self.kwds)
        # if isinstance(self.dist, rv_discrete):
        #     return self.dist.expect(func, a, loc, lb, ub, conditional, **kwds)
        # else:
        return self.dist.expect(func, a, loc, scale, lb, ub,
                                    conditional, **kwds)

    def support(self):
        return self.dist.support(*self.args, **self.kwds)

class rv_discrete_frozen(rv_frozen):

    def pmf(self, k):
        return self.dist.pmf(k, *self.args, **self.kwds)

    def logpmf(self, k):  # No error
        return self.dist.logpmf(k, *self.args, **self.kwds)
class rv_continuous_frozen(rv_frozen):

    def pdf(self, x):
        return self.dist.pdf(x, *self.args, **self.kwds)

    def logpdf(self, x):
        return self.dist.logpdf(x, *self.args, **self.kwds)


def argsreduce(cond, *args):
    newargs = np.atleast_1d(*args)

    # np.atleast_1d returns an array if only one argument, or a list of arrays
    # if more than one argument.
    if not isinstance(newargs, (list | tuple)):
        newargs = (newargs,)

    if np.all(cond):
        # broadcast arrays with cond
        *newargs, cond = np.broadcast_arrays(*newargs, cond)
        return [arg.ravel() for arg in newargs]

    s = cond.shape
    return [(arg if np.size(arg) == 1
            else np.extract(cond, np.broadcast_to(arg, s)))
            for arg in newargs]


def _moment_from_stats(n, mu, mu2, g1, g2, moment_func, args):
    if (n == 0):
        return 1.0
    elif (n == 1):
        if mu is None:
            val = moment_func(1, *args)
        else:
            val = mu
    elif (n == 2):
        if mu2 is None or mu is None:
            val = moment_func(2, *args)
        else:
            val = mu2 + mu*mu
    elif (n == 3):
        if g1 is None or mu2 is None or mu is None:
            val = moment_func(3, *args)
        else:
            mu3 = g1 * np.power(mu2, 1.5)  # 3rd central moment
            val = mu3+3*mu*mu2+mu*mu*mu  # 3rd non-central moment
    elif (n == 4):
        if g1 is None or g2 is None or mu2 is None or mu is None:
            val = moment_func(4, *args)
        else:
            mu4 = (g2+3.0)*(mu2**2.0)  # 4th central moment
            mu3 = g1*np.power(mu2, 1.5)  # 3rd central moment
            val = mu4+4*mu*mu3+6*mu*mu*mu2+mu*mu*mu*mu
    else:
        val = moment_func(n, *args)

    return val

class rv_generic:
    def __init__(self, seed=None):
        super().__init__()

        # figure out if _stats signature has 'moments' keyword
        sig = _getfullargspec(self._stats)
        self._stats_has_moments = ((sig.varkw is not None) or
                                   ('moments' in sig.args) or
                                   ('moments' in sig.kwonlyargs))
        self._random_state = check_random_state(seed)

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, seed):
        self._random_state = check_random_state(seed)

    def __setstate__(self, state):
        try:
            self.__dict__.update(state)
            self._attach_methods()
        except ValueError:
            self._ctor_param = state[0]
            self._random_state = state[1]
            self.__init__()

    def _attach_methods(self):
        raise NotImplementedError

    def _attach_argparser_methods(self):
        ns = {}
        exec(self._parse_arg_template, ns)
        # NB: attach to the instance, not class
        for name in ['_parse_args', '_parse_args_stats', '_parse_args_rvs']:
            setattr(self, name, types.MethodType(ns[name], self))

    def _construct_argparser(
            self, meths_to_inspect, locscale_in, locscale_out):
        if self.shapes:
            # sanitize the user-supplied shapes
            if not isinstance(self.shapes, str):
                raise TypeError('shapes must be a string.')

            shapes = self.shapes.replace(',', ' ').split()

            for field in shapes:
                if keyword.iskeyword(field):
                    raise SyntaxError('keywords cannot be used as shapes.')
                if not re.match('^[_a-zA-Z][_a-zA-Z0-9]*$', field):
                    raise SyntaxError(
                        'shapes must be valid python identifiers')
        else:
            shapes_list = []
            for meth in meths_to_inspect:
                shapes_args = _getfullargspec(meth)  # NB does not contain self
                args = shapes_args.args[1:]       # peel off 'x', too

                if args:
                    shapes_list.append(args)

                    # *args or **kwargs are not allowed w/automatic shapes
                    if shapes_args.varargs is not None:
                        raise TypeError(
                            '*args are not allowed w/out explicit shapes')
                    if shapes_args.varkw is not None:
                        raise TypeError(
                            '**kwds are not allowed w/out explicit shapes')
                    if shapes_args.kwonlyargs:
                        raise TypeError(
                            'kwonly args are not allowed w/out explicit shapes')
                    if shapes_args.defaults is not None:
                        raise TypeError('defaults are not allowed for shapes')

            if shapes_list:
                shapes = shapes_list[0]

                # make sure the signatures are consistent
                for item in shapes_list:
                    if item != shapes:
                        raise TypeError('Shape arguments are inconsistent.')
            else:
                shapes = []

        # have the arguments, construct the method from template
        shapes_str = ', '.join(shapes) + ', ' if shapes else ''  # NB: not None
        dct = dict(shape_arg_str=shapes_str,
                   locscale_in=locscale_in,
                   locscale_out=locscale_out,
                   )

        # this string is used by _attach_argparser_methods
        self._parse_arg_template = parse_arg_template % dct

        self.shapes = ', '.join(shapes) if shapes else None
        if not hasattr(self, 'numargs'):
            # allows more general subclassing with *args
            self.numargs = len(shapes)

    def _construct_doc(self, docdict, shapes_vals=None):
        """Construct the instance docstring with string substitutions."""
        tempdict = docdict.copy()
        tempdict['name'] = self.name or 'distname'
        tempdict['shapes'] = self.shapes or ''

        if shapes_vals is None:
            shapes_vals = ()
        try:
            vals = ', '.join(f'{val:.3g}' for val in shapes_vals)
        except TypeError:
            vals = ', '.join(f'{val}' for val in shapes_vals)
        tempdict['vals'] = vals

        tempdict['shapes_'] = self.shapes or ''
        if self.shapes and self.numargs == 1:
            tempdict['shapes_'] += ','

        if self.shapes:
            tempdict['set_vals_stmt'] = f'>>> {self.shapes} = {vals}'
        else:
            tempdict['set_vals_stmt'] = ''

        if self.shapes is None:
            # remove shapes from call parameters if there are none
            for item in ['default', 'before_notes']:
                tempdict[item] = tempdict[item].replace(
                    "\n%(shapes)s : array_like\n    shape parameters", "")

        # correct for empty shapes
        self.__doc__ = self.__doc__.replace('(, ', '(').replace(', )', ')')


    def freeze(self, *args, **kwds):
        if isinstance(self, rv_continuous):
            return rv_continuous_frozen(self, *args, **kwds)
        else:
            return rv_discrete_frozen(self, *args, **kwds)

    def __call__(self, *args, **kwds):
        return self.freeze(*args, **kwds)
    __call__.__doc__ = freeze.__doc__

    def _stats(self, *args, **kwds):
        return None, None, None, None

    def _munp(self, n, *args):
        # Silence floating point warnings from integration.
        with np.errstate(all='ignore'):
            vals = self.generic_moment(n, *args)
        return vals

    def _argcheck_rvs(self, *args, **kwargs):
        size = kwargs.get('size', None)
        all_bcast = np.broadcast_arrays(*args)

        def squeeze_left(a):
            while a.ndim > 0 and a.shape[0] == 1:
                a = a[0]
            return a

        all_bcast = [squeeze_left(a) for a in all_bcast]
        bcast_shape = all_bcast[0].shape
        bcast_ndim = all_bcast[0].ndim

        if size is None:
            size_ = bcast_shape
        else:
            size_ = tuple(np.atleast_1d(size))

        ndiff = bcast_ndim - len(size_)
        if ndiff < 0:
            bcast_shape = (1,)*(-ndiff) + bcast_shape
        elif ndiff > 0:
            size_ = (1,)*ndiff + size_

        # This compatibility test is not standard.  In "regular" broadcasting,
        # two shapes are compatible if for each dimension, the lengths are the
        # same or one of the lengths is 1.  Here, the length of a dimension in
        # size_ must not be less than the corresponding length in bcast_shape.
        ok = all([bcdim == 1 or bcdim == szdim
                  for (bcdim, szdim) in zip(bcast_shape, size_)])
        if not ok:
            raise ValueError("size does not match the broadcast shape of "
                             f"the parameters. {size}, {size_}, {bcast_shape}")

        param_bcast = all_bcast[:-2]
        loc_bcast = all_bcast[-2]
        scale_bcast = all_bcast[-1]

        return param_bcast, loc_bcast, scale_bcast, size_


    def _argcheck(self, *args):
        cond = 1
        for arg in args:
            cond = logical_and(cond, (asarray(arg) > 0))
        return cond

    def _get_support(self, *args, **kwargs):
        return self.a, self.b

    def _support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a <= x) & (x <= b)

    def _open_support_mask(self, x, *args):
        a, b = self._get_support(*args)
        with np.errstate(invalid='ignore'):
            return (a < x) & (x < b)

    def _rvs(self, *args, size=None, random_state=None):
        U = random_state.uniform(size=size)
        Y = self._ppf(U, *args)
        return Y

    def _logcdf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._cdf(x, *args))

    def _sf(self, x, *args):
        return 1.0-self._cdf(x, *args)

    def _logsf(self, x, *args):
        with np.errstate(divide='ignore'):
            return log(self._sf(x, *args))

    def _ppf(self, q, *args):
        return self._ppfvec(q, *args)

    def _isf(self, q, *args):
        return self._ppf(1.0-q, *args)  # use correct _ppf for subclasses

    def rvs(self, *args, **kwds):
        discrete = kwds.pop('discrete', None)
        rndm = kwds.pop('random_state', None)
        args, loc, scale, size = self._parse_args_rvs(*args, **kwds)
        cond = logical_and(self._argcheck(*args), (scale >= 0))
        if not np.all(cond):
            message = ("Domain error in arguments. The `scale` parameter must "
                       "be positive for all distributions, and many "
                       "distributions have restrictions on shape parameters. "
                       f"Please see the `scipy.stats.{self.name}` "
                       "documentation for details.")
            raise ValueError(message)

        if np.all(scale == 0):
            return loc*ones(size, 'd')

        # extra gymnastics needed for a custom random_state
        if rndm is not None:
            random_state_saved = self._random_state
            random_state = check_random_state(rndm)
        else:
            random_state = self._random_state

        vals = self._rvs(*args, size=size, random_state=random_state)

        vals = vals * scale + loc

        # do not forget to restore the _random_state
        if rndm is not None:
            self._random_state = random_state_saved

        # Cast to int if discrete
        if discrete:# and not isinstance(self, rv_sample):
            if size == ():
                vals = int(vals)
            else:
                vals = vals.astype(np.int64)

        return vals

    def stats(self, *args, **kwds):
        args, loc, scale, moments = self._parse_args_stats(*args, **kwds)
        # scale = 1 by construction for discrete RVs
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = []
        default = np.full(shape(cond), fill_value=self.badvalue)

        # Use only entries that are valid in calculation
        if np.any(cond):
            goodargs = argsreduce(cond, *(args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]

            if self._stats_has_moments:
                mu, mu2, g1, g2 = self._stats(*goodargs,
                                              **{'moments': moments})
            else:
                mu, mu2, g1, g2 = self._stats(*goodargs)

            if 'm' in moments:
                if mu is None:
                    mu = self._munp(1, *goodargs)
                out0 = default.copy()
                place(out0, cond, mu * scale + loc)
                output.append(out0)

            if 'v' in moments:
                if mu2 is None:
                    mu2p = self._munp(2, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    # if mean is inf then var is also inf
                    with np.errstate(invalid='ignore'):
                        mu2 = np.where(~np.isinf(mu), mu2p - mu**2, np.inf)
                out0 = default.copy()
                place(out0, cond, mu2 * scale * scale)
                output.append(out0)

            if 's' in moments:
                if g1 is None:
                    mu3p = self._munp(3, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu2 = mu2p - mu * mu
                    with np.errstate(invalid='ignore'):
                        mu3 = (-mu*mu - 3*mu2)*mu + mu3p
                        g1 = mu3 / np.power(mu2, 1.5)
                out0 = default.copy()
                place(out0, cond, g1)
                output.append(out0)

            if 'k' in moments:
                if g2 is None:
                    mu4p = self._munp(4, *goodargs)
                    if mu is None:
                        mu = self._munp(1, *goodargs)
                    if mu2 is None:
                        mu2p = self._munp(2, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu2 = mu2p - mu * mu
                    if g1 is None:
                        mu3 = None
                    else:
                        # (mu2**1.5) breaks down for nan and inf
                        mu3 = g1 * np.power(mu2, 1.5)
                    if mu3 is None:
                        mu3p = self._munp(3, *goodargs)
                        with np.errstate(invalid='ignore'):
                            mu3 = (-mu * mu - 3 * mu2) * mu + mu3p
                    with np.errstate(invalid='ignore'):
                        mu4 = ((-mu**2 - 6*mu2) * mu - 4*mu3)*mu + mu4p
                        g2 = mu4 / mu2**2.0 - 3.0
                out0 = default.copy()
                place(out0, cond, g2)
                output.append(out0)
        else:  # no valid args
            output = [default.copy() for _ in moments]

        output = [out[()] for out in output]
        if len(output) == 1:
            return output[0]
        else:
            return tuple(output)

    def entropy(self, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        # NB: for discrete distributions scale=1 by construction in _parse_args
        loc, scale = map(asarray, (loc, scale))
        args = tuple(map(asarray, args))
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        output = zeros(shape(cond0), 'd')
        place(output, (1-cond0), self.badvalue)
        goodargs = argsreduce(cond0, scale, *args)
        goodscale = goodargs[0]
        goodargs = goodargs[1:]
        place(output, cond0, self.vecentropy(*goodargs) + log(goodscale))
        return output[()]

    def moment(self, order, *args, **kwds):
        n = order
        shapes, loc, scale = self._parse_args(*args, **kwds)
        args = np.broadcast_arrays(*(*shapes, loc, scale))
        *shapes, loc, scale = args

        i0 = np.logical_and(self._argcheck(*shapes), scale > 0)
        i1 = np.logical_and(i0, loc == 0)
        i2 = np.logical_and(i0, loc != 0)

        args = argsreduce(i0, *shapes, loc, scale)
        *shapes, loc, scale = args

        if (floor(n) != n):
            raise ValueError("Moment must be an integer.")
        if (n < 0):
            raise ValueError("Moment must be positive.")
        mu, mu2, g1, g2 = None, None, None, None
        if (n > 0) and (n < 5):
            if self._stats_has_moments:
                mdict = {'moments': {1: 'm', 2: 'v', 3: 'vs', 4: 'mvsk'}[n]}
            else:
                mdict = {}
            mu, mu2, g1, g2 = self._stats(*shapes, **mdict)
        val = np.empty(loc.shape)  # val needs to be indexed by loc
        val[...] = _moment_from_stats(n, mu, mu2, g1, g2, self._munp, shapes)

        # Convert to transformed  X = L + S*Y
        # E[X^n] = E[(L+S*Y)^n] = L^n sum(comb(n, k)*(S/L)^k E[Y^k], k=0...n)
        result = zeros(i0.shape)
        place(result, ~i0, self.badvalue)

        if i1.any():
            res1 = scale[loc == 0]**n * val[loc == 0]
            place(result, i1, res1)

        if i2.any():
            mom = [mu, mu2, g1, g2]
            arrs = [i for i in mom if i is not None]
            idx = [i for i in range(4) if mom[i] is not None]
            if any(idx):
                arrs = argsreduce(loc != 0, *arrs)
                j = 0
                for i in idx:
                    mom[i] = arrs[j]
                    j += 1
            mu, mu2, g1, g2 = mom
            args = argsreduce(loc != 0, *shapes, loc, scale, val)
            *shapes, loc, scale, val = args

            res2 = zeros(loc.shape, dtype='d')
            fac = scale / loc
            for k in range(n):
                valk = _moment_from_stats(k, mu, mu2, g1, g2, self._munp,
                                          shapes)
                res2 += comb(n, k, exact=True)*fac**k * valk
            res2 += fac**n * val
            res2 *= loc**n
            place(result, i2, res2)

        return result[()]

    def median(self, *args, **kwds):
        return self.ppf(0.5, *args, **kwds)

    def mean(self, *args, **kwds):
        kwds['moments'] = 'm'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def var(self, *args, **kwds):
        kwds['moments'] = 'v'
        res = self.stats(*args, **kwds)
        if isinstance(res, ndarray) and res.ndim == 0:
            return res[()]
        return res

    def std(self, *args, **kwds):
        kwds['moments'] = 'v'
        res = sqrt(self.stats(*args, **kwds))
        return res

    def interval(self, confidence, *args, **kwds):
        alpha = confidence

        alpha = asarray(alpha)
        if np.any((alpha > 1) | (alpha < 0)):
            raise ValueError("alpha must be between 0 and 1 inclusive")
        q1 = (1.0-alpha)/2
        q2 = (1.0+alpha)/2
        a = self.ppf(q1, *args, **kwds)
        b = self.ppf(q2, *args, **kwds)
        return a, b

    def support(self, *args, **kwargs):
        args, loc, scale = self._parse_args(*args, **kwargs)
        arrs = np.broadcast_arrays(*args, loc, scale)
        args, loc, scale = arrs[:-2], arrs[-2], arrs[-1]
        cond = self._argcheck(*args) & (scale > 0)
        _a, _b = self._get_support(*args)
        if cond.all():
            return _a * scale + loc, _b * scale + loc
        elif cond.ndim == 0:
            return self.badvalue, self.badvalue
        # promote bounds to at least float to fill in the badvalue
        _a, _b = np.asarray(_a).astype('d'), np.asarray(_b).astype('d')
        out_a, out_b = _a * scale + loc, _b * scale + loc
        place(out_a, 1-cond, self.badvalue)
        place(out_b, 1-cond, self.badvalue)
        return out_a, out_b

    def nnlf(self, theta, x):
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (asarray(x)-loc) / scale
        n_log_scale = len(x) * log(scale)
        if np.any(~self._support_mask(x, *args)):
            return inf
        return self._nnlf(x, *args) + n_log_scale

    def _nnlf(self, x, *args):
        return -np.sum(self._logpxf(x, *args), axis=0)

    def _nlff_and_penalty(self, x, args, log_fitfun):
        # negative log fit function
        cond0 = ~self._support_mask(x, *args)
        n_bad = np.count_nonzero(cond0, axis=0)
        if n_bad > 0:
            x = argsreduce(~cond0, x)[0]
        logff = log_fitfun(x, *args)
        finite_logff = np.isfinite(logff)
        n_bad += np.sum(~finite_logff, axis=0)
        if n_bad > 0:
            penalty = n_bad * log(_XMAX) * 100
            return -np.sum(logff[finite_logff], axis=0) + penalty
        return -np.sum(logff, axis=0)

    def _penalized_nnlf(self, theta, x):
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = asarray((x-loc) / scale)
        n_log_scale = len(x) * log(scale)
        return self._nlff_and_penalty(x, args, self._logpxf) + n_log_scale

    def _penalized_nlpsf(self, theta, x):
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        x = (np.sort(x) - loc)/scale

        def log_psf(x, *args):
            x, lj = np.unique(x, return_counts=True)  # fast for sorted x
            cdf_data = self._cdf(x, *args) if x.size else []
            if not (x.size and 1 - cdf_data[-1] <= 0):
                cdf = np.concatenate(([0], cdf_data, [1]))
                lj = np.concatenate((lj, [1]))
            else:
                cdf = np.concatenate(([0], cdf_data))
            # here we could use logcdf w/ logsumexp trick to take differences,
            # but in the context of the method, it seems unlikely to matter
            return lj * np.log(np.diff(cdf) / lj)

        return self._nlff_and_penalty(x, args, log_psf)


class _ShapeInfo:
    def __init__(self, name, integrality=False, domain=(-np.inf, np.inf),
                 inclusive=(True, True)):
        self.name = name
        self.integrality = integrality

        domain = list(domain)
        if np.isfinite(domain[0]) and not inclusive[0]:
            domain[0] = np.nextafter(domain[0], np.inf)
        if np.isfinite(domain[1]) and not inclusive[1]:
            domain[1] = np.nextafter(domain[1], -np.inf)
        self.domain = domain



def _quad_weight(func, a, b, args, full_output, epsabs, epsrel,
                 limlst, limit, maxp1,weight, wvar, wopts):
    if weight not in ['cos','sin','alg','alg-loga','alg-logb','alg-log','cauchy']:
        raise ValueError(f"{weight} not a recognized weighting function.")

    strdict = {'cos':1,'sin':2,'alg':1,'alg-loga':2,'alg-logb':3,'alg-log':4}

    if weight in ['cos','sin']:
        integr = strdict[weight]
        if (b != np.inf and a != -np.inf):  # finite limits
            if wopts is None:         # no precomputed Chebyshev moments
                return _quadpack._qawoe(func, a, b, wvar, integr, args, full_output,
                                        epsabs, epsrel, limit, maxp1,1)
            else:                     # precomputed Chebyshev moments
                momcom = wopts[0]
                chebcom = wopts[1]
                return _quadpack._qawoe(func, a, b, wvar, integr, args,
                                        full_output,epsabs, epsrel, limit, maxp1, 2,
                                        momcom, chebcom)

        elif (b == np.inf and a != -np.inf):
            return _quadpack._qawfe(func, a, wvar, integr, args, full_output,
                                    epsabs, limlst, limit, maxp1)
        elif (b != np.inf and a == -np.inf):  # remap function and interval
            if weight == 'cos':
                def thefunc(x,*myargs):
                    y = -x
                    func = myargs[0]
                    myargs = (y,) + myargs[1:]
                    return func(*myargs)
            else:
                def thefunc(x,*myargs):
                    y = -x
                    func = myargs[0]
                    myargs = (y,) + myargs[1:]
                    return -func(*myargs)
            args = (func,) + args
            return _quadpack._qawfe(thefunc, -b, wvar, integr, args,
                                    full_output, epsabs, limlst, limit, maxp1)
        else:
            raise ValueError("Cannot integrate with this weight from -Inf to +Inf.")
    else:
        if a in [-np.inf, np.inf] or b in [-np.inf, np.inf]:
            message = "Cannot integrate with this weight over an infinite interval."
            raise ValueError(message)

        if weight.startswith('alg'):
            integr = strdict[weight]
            return _quadpack._qawse(func, a, b, wvar, integr, args,
                                    full_output, epsabs, epsrel, limit)
        else:  # weight == 'cauchy'
            return _quadpack._qawce(func, a, b, wvar, args, full_output,
                                    epsabs, epsrel, limit)

def quad(func, a, b, args=(), full_output=0, epsabs=1.49e-8, epsrel=1.49e-8,
         limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50,
         limlst=50, complex_func=False):
    if not isinstance(args, tuple):
        args = (args,)

    # check the limits of integration: \int_a^b, expect a < b
    flip, a, b = b < a, min(a, b), max(a, b)

    if complex_func:
        def imfunc(x, *args):
            return func(x, *args).imag

        def refunc(x, *args):
            return func(x, *args).real

        re_retval = quad(refunc, a, b, args, full_output, epsabs,
                         epsrel, limit, points, weight, wvar, wopts,
                         maxp1, limlst, complex_func=False)
        im_retval = quad(imfunc, a, b, args, full_output, epsabs,
                         epsrel, limit, points, weight, wvar, wopts,
                         maxp1, limlst, complex_func=False)
        integral = re_retval[0] + 1j*im_retval[0]
        error_estimate = re_retval[1] + 1j*im_retval[1]
        retval = integral, error_estimate
        if full_output:
            msgexp = {}
            msgexp["real"] = re_retval[2:]
            msgexp["imag"] = im_retval[2:]
            retval = retval + (msgexp,)

        return retval

    if weight is None:
        retval = _quad(func, a, b, args, full_output, epsabs, epsrel, limit,
                       points)
    else:
        if points is not None:
            msg = ("Break points cannot be specified when using weighted integrand.\n"
                   "Continuing, ignoring specified points.")
        retval = _quad_weight(func, a, b, args, full_output, epsabs, epsrel,
                              limlst, limit, maxp1, weight, wvar, wopts)

    if flip:
        retval = (-retval[0],) + retval[1:]

    ier = retval[-1]
    if ier == 0:
        return retval[:-1]

    msgs = {80: "A Python error occurred possibly while calling the function.",
             1: f"The maximum number of subdivisions ({limit}) has been achieved.\n  "
                f"If increasing the limit yields no improvement it is advised to "
                f"analyze \n  the integrand in order to determine the difficulties.  "
                f"If the position of a \n  local difficulty can be determined "
                f"(singularity, discontinuity) one will \n  probably gain from "
                f"splitting up the interval and calling the integrator \n  on the "
                f"subranges.  Perhaps a special-purpose integrator should be used.",
             2: "The occurrence of roundoff error is detected, which prevents \n  "
                "the requested tolerance from being achieved.  "
                "The error may be \n  underestimated.",
             3: "Extremely bad integrand behavior occurs at some points of the\n  "
                "integration interval.",
             4: "The algorithm does not converge.  Roundoff error is detected\n  "
                "in the extrapolation table.  It is assumed that the requested "
                "tolerance\n  cannot be achieved, and that the returned result "
                "(if full_output = 1) is \n  the best which can be obtained.",
             5: "The integral is probably divergent, or slowly convergent.",
             6: "The input is invalid.",
             7: "Abnormal termination of the routine.  The estimates for result\n  "
                "and error are less reliable.  It is assumed that the requested "
                "accuracy\n  has not been achieved.",
            'unknown': "Unknown error."}

    if weight in ['cos','sin'] and (b == np.inf or a == -np.inf):
        msgs[1] = (
            "The maximum number of cycles allowed has been achieved., e.e.\n  of "
            "subintervals (a+(k-1)c, a+kc) where c = (2*int(abs(omega)+1))\n  "
            "*pi/abs(omega), for k = 1, 2, ..., lst.  "
            "One can allow more cycles by increasing the value of limlst.  "
            "Look at info['ierlst'] with full_output=1."
        )
        msgs[4] = (
            "The extrapolation table constructed for convergence acceleration\n  of "
            "the series formed by the integral contributions over the cycles, \n  does "
            "not converge to within the requested accuracy.  "
            "Look at \n  info['ierlst'] with full_output=1."
        )
        msgs[7] = (
            "Bad integrand behavior occurs within one or more of the cycles.\n  "
            "Location and type of the difficulty involved can be determined from \n  "
            "the vector info['ierlist'] obtained with full_output=1."
        )
        explain = {1: "The maximum number of subdivisions (= limit) has been \n  "
                      "achieved on this cycle.",
                   2: "The occurrence of roundoff error is detected and prevents\n  "
                      "the tolerance imposed on this cycle from being achieved.",
                   3: "Extremely bad integrand behavior occurs at some points of\n  "
                      "this cycle.",
                   4: "The integral over this cycle does not converge (to within the "
                      "required accuracy) due to roundoff in the extrapolation "
                      "procedure invoked on this cycle.  It is assumed that the result "
                      "on this interval is the best which can be obtained.",
                   5: "The integral over this cycle is probably divergent or "
                      "slowly convergent."}

    try:
        msg = msgs[ier]
    except KeyError:
        msg = msgs['unknown']

    if ier in [1,2,3,4,5,7]:
        if full_output:
            if weight in ['cos', 'sin'] and (b == np.inf or a == -np.inf):
                return retval[:-1] + (msg, explain)
            else:
                return retval[:-1] + (msg,)
        else:
            return retval[:-1]

    elif ier == 6:  # Forensic decision tree when QUADPACK throws ier=6
        if epsabs <= 0:  # Small error tolerance - applies to all methods
            if epsrel < max(50 * sys.float_info.epsilon, 5e-29):
                msg = ("If 'epsabs'<=0, 'epsrel' must be greater than both"
                       " 5e-29 and 50*(machine epsilon).")
            elif weight in ['sin', 'cos'] and (abs(a) + abs(b) == np.inf):
                msg = ("Sine or cosine weighted integrals with infinite domain"
                       " must have 'epsabs'>0.")

        elif weight is None:
            if points is None:  # QAGSE/QAGIE
                msg = ("Invalid 'limit' argument. There must be"
                       " at least one subinterval")
            else:  # QAGPE
                if not (min(a, b) <= min(points) <= max(points) <= max(a, b)):
                    msg = ("All break points in 'points' must lie within the"
                           " integration limits.")
                elif len(points) >= limit:
                    msg = (f"Number of break points ({len(points):d}) "
                           f"must be less than subinterval limit ({limit:d})")

        else:
            if maxp1 < 1:
                msg = "Chebyshev moment limit maxp1 must be >=1."

            elif weight in ('cos', 'sin') and abs(a+b) == np.inf:  # QAWFE
                msg = "Cycle limit limlst must be >=3."

            elif weight.startswith('alg'):  # QAWSE
                if min(wvar) < -1:
                    msg = "wvar parameters (alpha, beta) must both be >= -1."
                if b < a:
                    msg = "Integration limits a, b must satistfy a<b."

            elif weight == 'cauchy' and wvar in (a, b):
                msg = ("Parameter 'wvar' must not equal"
                       " integration limits 'a' or 'b'.")

    raise ValueError(msg)


def _quad(func,a,b,args,full_output,epsabs,epsrel,limit,points):
    infbounds = 0
    if (b != np.inf and a != -np.inf):
        pass   # standard integration
    elif (b == np.inf and a != -np.inf):
        infbounds = 1
        bound = a
    elif (b == np.inf and a == -np.inf):
        infbounds = 2
        bound = 0     # ignored
    elif (b != np.inf and a == -np.inf):
        infbounds = -1
        bound = b
    else:
        raise RuntimeError("Infinity comparisons don't work for you.")

    if points is None:
        if infbounds == 0:
            return _quadpack._qagse(func,a,b,args,full_output,epsabs,epsrel,limit)
        else:
            return _quadpack._qagie(func, bound, infbounds, args, full_output, 
                                    epsabs, epsrel, limit)
    else:
        if infbounds != 0:
            raise ValueError("Infinity inputs cannot be used with break points.")
        else:
            #Duplicates force function evaluation at singular points
            the_points = np.unique(points)
            the_points = the_points[a < the_points]
            the_points = the_points[the_points < b]
            the_points = np.concatenate((the_points, (0., 0.)))
            return _quadpack._qagpe(func, a, b, the_points, args, full_output,
                                    epsabs, epsrel, limit)



def _central_diff_weights(Np, ndiv=1):
    if Np < ndiv + 1:
        raise ValueError(
            "Number of points must be at least the derivative order + 1."
        )
    if Np % 2 == 0:
        raise ValueError("The number of points must be odd.")
    from scipy import linalg

    ho = Np >> 1
    x = arange(-ho, ho + 1.0)
    x = x[:, newaxis]
    X = x**0.0
    for k in range(1, Np):
        X = hstack([X, x**k])
    w = prod(arange(1, ndiv + 1), axis=0) * linalg.inv(X)[ndiv]
    return w

def _derivative(func, x0, dx=1.0, n=1, args=(), order=3):
    if order < n + 1:
        raise ValueError(
            "'order' (the number of points used to compute the derivative), "
            "must be at least the derivative order 'n' + 1."
        )
    if order % 2 == 0:
        raise ValueError(
            "'order' (the number of points used to compute the derivative) "
            "must be odd."
        )
    # pre-computed for n=1 and 2 and low-order for speed.
    if n == 1:
        if order == 3:
            weights = array([-1, 0, 1]) / 2.0
        elif order == 5:
            weights = array([1, -8, 0, 8, -1]) / 12.0
        elif order == 7:
            weights = array([-1, 9, -45, 0, 45, -9, 1]) / 60.0
        elif order == 9:
            weights = array([3, -32, 168, -672, 0, 672, -168, 32, -3]) / 840.0
        else:
            weights = _central_diff_weights(order, 1)
    elif n == 2:
        if order == 3:
            weights = array([1, -2.0, 1])
        elif order == 5:
            weights = array([-1, 16, -30, 16, -1]) / 12.0
        elif order == 7:
            weights = array([2, -27, 270, -490, 270, -27, 2]) / 180.0
        elif order == 9:
            weights = (
                array([-9, 128, -1008, 8064, -14350, 8064, -1008, 128, -9])
                / 5040.0
            )
        else:
            weights = _central_diff_weights(order, 2)
    else:
        weights = _central_diff_weights(order, n)
    val = 0.0
    ho = order >> 1
    for k in range(order):
        val += weights[k] * func(x0 + (k - ho) * dx, *args)
    return val / prod((dx,) * n, axis=0)
class FitError(RuntimeError):
    def __init__(self, msg=None):
        if msg is None:
            msg = ("An error occurred when fitting a distribution to data.")
        self.args = (msg,)


def _fit_determine_optimizer(optimizer):
    if not callable(optimizer) and isinstance(optimizer, str):
        if not optimizer.startswith('fmin_'):
            optimizer = "fmin_"+optimizer
        if optimizer == 'fmin_':
            optimizer = 'fmin'
        try:
            optimizer = getattr(optimize, optimizer)
        except AttributeError as e:
            raise ValueError(f"{optimizer} is not a valid optimizer") from e
    return optimizer


def _get_fixed_fit_value(kwds, names):
    vals = [(name, kwds.pop(name)) for name in names if name in kwds]
    if len(vals) > 1:
        repeated = [name for name, val in vals]
        raise ValueError("fit method got multiple keyword arguments to "
                         "specify the same fixed parameter: " +
                         ', '.join(repeated))
    return vals[0][1] if vals else None

def _sum_finite(x):
    finite_x = np.isfinite(x)
    bad_count = finite_x.size - np.count_nonzero(finite_x)
    return np.sum(x[finite_x]), bad_count
class rv_continuous(rv_generic):

    def __init__(self, momtype=1, a=None, b=None, xtol=1e-14,
                 badvalue=None, name=None, longname=None,
                 shapes=None, seed=None):

        super().__init__(seed)

        # save the ctor parameters, cf generic freeze
        self._ctor_param = dict(
            momtype=momtype, a=a, b=b, xtol=xtol,
            badvalue=badvalue, name=name, longname=longname,
            shapes=shapes, seed=seed)

        if badvalue is None:
            badvalue = nan
        if name is None:
            name = 'Distribution'
        self.badvalue = badvalue
        self.name = name
        self.a = a
        self.b = b
        if a is None:
            self.a = -inf
        if b is None:
            self.b = inf
        self.xtol = xtol
        self.moment_type = momtype
        self.shapes = shapes

        self._construct_argparser(meths_to_inspect=[self._pdf, self._cdf],
                                  locscale_in='loc=0, scale=1',
                                  locscale_out='loc, scale')
        self._attach_methods()

        if longname is None:
            if name[0] in ['aeiouAEIOU']:
                hstr = "An "
            else:
                hstr = "A "
            longname = hstr + name


    def __getstate__(self):
        dct = self.__dict__.copy()

        attrs = ["_parse_args", "_parse_args_stats", "_parse_args_rvs",
                 "_cdfvec", "_ppfvec", "vecentropy", "generic_moment"]
        [dct.pop(attr, None) for attr in attrs]
        return dct

    def _attach_methods(self):
        self._attach_argparser_methods()

        # nin correction
        self._ppfvec = vectorize(self._ppf_single, otypes='d')
        self._ppfvec.nin = self.numargs + 1
        self.vecentropy = vectorize(self._entropy, otypes='d')
        self._cdfvec = vectorize(self._cdf_single, otypes='d')
        self._cdfvec.nin = self.numargs + 1

        if self.moment_type == 0:
            self.generic_moment = vectorize(self._mom0_sc, otypes='d')
        else:
            self.generic_moment = vectorize(self._mom1_sc, otypes='d')
        # Because of the *args argument of _mom0_sc, vectorize cannot count the
        # number of arguments correctly.
        self.generic_moment.nin = self.numargs + 1

    def _updated_ctor_param(self):
        dct = self._ctor_param.copy()
        dct['a'] = self.a
        dct['b'] = self.b
        dct['xtol'] = self.xtol
        dct['badvalue'] = self.badvalue
        dct['name'] = self.name
        dct['shapes'] = self.shapes
        return dct

    def _ppf_to_solve(self, x, q, *args):
        return self.cdf(*(x, )+args)-q

    def _ppf_single(self, q, *args):
        factor = 10.
        left, right = self._get_support(*args)

        if np.isinf(left):
            left = min(-factor, right)
            while self._ppf_to_solve(left, q, *args) > 0.:
                left, right = left * factor, left
            # left is now such that cdf(left) <= q
            # if right has changed, then cdf(right) > q

        if np.isinf(right):
            right = max(factor, left)
            while self._ppf_to_solve(right, q, *args) < 0.:
                left, right = right, right * factor
            # right is now such that cdf(right) >= q

        return optimize.brentq(self._ppf_to_solve,
                               left, right, args=(q,)+args, xtol=self.xtol)

    # moment from definition
    def _mom_integ0(self, x, m, *args):
        return x**m * self.pdf(x, *args)

    def _mom0_sc(self, m, *args):
        _a, _b = self._get_support(*args)
        return quad(self._mom_integ0, _a, _b,
                              args=(m,)+args)[0]

    # moment calculated using ppf
    def _mom_integ1(self, q, m, *args):
        return (self.ppf(q, *args))**m

    def _mom1_sc(self, m, *args):
        return quad(self._mom_integ1, 0, 1, args=(m,)+args)[0]

    def _pdf(self, x, *args):
        return _derivative(self._cdf, x, dx=1e-5, args=args, order=5)

    # Could also define any of these
    def _logpdf(self, x, *args):
        p = self._pdf(x, *args)
        with np.errstate(divide='ignore'):
            return log(p)

    def _logpxf(self, x, *args):
        return self._logpdf(x, *args)

    def _cdf_single(self, x, *args):
        _a, _b = self._get_support(*args)
        return quad(self._pdf, _a, x, args=args)[0]

    def _cdf(self, x, *args):
        return self._cdfvec(x, *args)

    def _logcdf(self, x, *args):
        median = self._ppf(0.5, *args)
        with np.errstate(divide='ignore'):
            return _lazywhere(x < median, (x,) + args,
                              f=lambda x, *args: np.log(self._cdf(x, *args)),
                              f2=lambda x, *args: np.log1p(-self._sf(x, *args)))

    def _logsf(self, x, *args):
        median = self._ppf(0.5, *args)
        with np.errstate(divide='ignore'):
            return _lazywhere(x > median, (x,) + args,
                              f=lambda x, *args: np.log(self._sf(x, *args)),
                              f2=lambda x, *args: np.log1p(-self._cdf(x, *args)))

    # generic _argcheck, _sf, _ppf, _isf, _rvs are defined
    # in rv_generic

    def pdf(self, x, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._pdf(*goodargs) / scale)
        if output.ndim == 0:
            return output[()]
        return output

    def logpdf(self, x, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._support_mask(x, *args) & (scale > 0)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        putmask(output, (1-cond0)+np.isnan(x), self.badvalue)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args+(scale,)))
            scale, goodargs = goodargs[-1], goodargs[:-1]
            place(output, cond, self._logpdf(*goodargs) - log(scale))
        if output.ndim == 0:
            return output[()]
        return output

    def cdf(self, x, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= np.asarray(_b)) & cond0
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._cdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logcdf(self, x, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = (x >= _b) & cond0
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, (1-cond0)*(cond1 == cond1)+np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logcdf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def sf(self, x, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = zeros(shape(cond), dtyp)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 1.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._sf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def logsf(self, x, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        x, loc, scale = map(asarray, (x, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        dtyp = np.promote_types(x.dtype, np.float64)
        x = np.asarray((x - loc)/scale, dtype=dtyp)
        cond0 = self._argcheck(*args) & (scale > 0)
        cond1 = self._open_support_mask(x, *args) & (scale > 0)
        cond2 = cond0 & (x <= _a)
        cond = cond0 & cond1
        output = empty(shape(cond), dtyp)
        output.fill(-inf)
        place(output, (1-cond0)+np.isnan(x), self.badvalue)
        place(output, cond2, 0.0)
        if np.any(cond):
            goodargs = argsreduce(cond, *((x,)+args))
            place(output, cond, self._logsf(*goodargs))
        if output.ndim == 0:
            return output[()]
        return output

    def ppf(self, q, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 0)
        cond3 = cond0 & (q == 1)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):  # call only if at least 1 entry
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._ppf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def isf(self, q, *args, **kwds):
        args, loc, scale = self._parse_args(*args, **kwds)
        q, loc, scale = map(asarray, (q, loc, scale))
        args = tuple(map(asarray, args))
        _a, _b = self._get_support(*args)
        cond0 = self._argcheck(*args) & (scale > 0) & (loc == loc)
        cond1 = (0 < q) & (q < 1)
        cond2 = cond0 & (q == 1)
        cond3 = cond0 & (q == 0)
        cond = cond0 & cond1
        output = np.full(shape(cond), fill_value=self.badvalue)

        lower_bound = _a * scale + loc
        upper_bound = _b * scale + loc
        place(output, cond2, argsreduce(cond2, lower_bound)[0])
        place(output, cond3, argsreduce(cond3, upper_bound)[0])

        if np.any(cond):
            goodargs = argsreduce(cond, *((q,)+args+(scale, loc)))
            scale, loc, goodargs = goodargs[-2], goodargs[-1], goodargs[:-2]
            place(output, cond, self._isf(*goodargs) * scale + loc)
        if output.ndim == 0:
            return output[()]
        return output

    def _unpack_loc_scale(self, theta):
        try:
            loc = theta[-2]
            scale = theta[-1]
            args = tuple(theta[:-2])
        except IndexError as e:
            raise ValueError("Not enough input arguments.") from e
        return loc, scale, args

    def _nnlf_and_penalty(self, x, args):
        if isinstance(x, CensoredData):
            # Filter out the data that is not in the support.
            xs = x._supported(*self._get_support(*args))
            n_bad = len(x) - len(xs)
            i1, i2 = xs._interval.T
            terms = [
                # logpdf of the noncensored data.
                self._logpdf(xs._uncensored, *args),
                # logcdf of the left-censored data.
                self._logcdf(xs._left, *args),
                # logsf of the right-censored data.
                self._logsf(xs._right, *args),
                # log of probability of the interval-censored data.
                np.log(self._delta_cdf(i1, i2, *args)),
            ]
        else:
            cond0 = ~self._support_mask(x, *args)
            n_bad = np.count_nonzero(cond0)
            if n_bad > 0:
                x = argsreduce(~cond0, x)[0]
            terms = [self._logpdf(x, *args)]

        totals, bad_counts = zip(*[_sum_finite(term) for term in terms])
        total = sum(totals)
        n_bad += sum(bad_counts)

        return -total + n_bad * _LOGXMAX * 100

    def _penalized_nnlf(self, theta, x):
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf
        if isinstance(x, CensoredData):
            x = (x - loc) / scale
            n_log_scale = (len(x) - x.num_censored()) * log(scale)
        else:
            x = (x - loc) / scale
            n_log_scale = len(x) * log(scale)

        return self._nnlf_and_penalty(x, args) + n_log_scale

    def _fitstart(self, data, args=None):
        """Starting point for fit (shape arguments + loc + scale)."""
        if args is None:
            args = (1.0,)*self.numargs
        loc, scale = self._fit_loc_scale_support(data, *args)
        return args + (loc, scale)

    def _reduce_func(self, args, kwds, data=None):
        shapes = []
        if self.shapes:
            shapes = self.shapes.replace(',', ' ').split()
            for j, s in enumerate(shapes):
                key = 'f' + str(j)
                names = [key, 'f' + s, 'fix_' + s]
                val = _get_fixed_fit_value(kwds, names)
                if val is not None:
                    kwds[key] = val

        args = list(args)
        Nargs = len(args)
        fixedn = []
        names = ['f%d' % n for n in range(Nargs - 2)] + ['floc', 'fscale']
        x0 = []
        for n, key in enumerate(names):
            if key in kwds:
                fixedn.append(n)
                args[n] = kwds.pop(key)
            else:
                x0.append(args[n])

        methods = {"mle", "mm"}
        method = kwds.pop('method', "mle").lower()
        if method == "mm":
            n_params = len(shapes) + 2 - len(fixedn)
            exponents = (np.arange(1, n_params+1))[:, np.newaxis]
            data_moments = np.sum(data[None, :]**exponents/len(data), axis=1)

            def objective(theta, x):
                return self._moment_error(theta, x, data_moments)

        elif method == "mle":
            objective = self._penalized_nnlf
        else:
            raise ValueError(f"Method '{method}' not available; "
                             f"must be one of {methods}")

        if len(fixedn) == 0:
            func = objective
            restore = None
        else:
            if len(fixedn) == Nargs:
                raise ValueError(
                    "All parameters fixed. There is nothing to optimize.")

            def restore(args, theta):
                i = 0
                for n in range(Nargs):
                    if n not in fixedn:
                        args[n] = theta[i]
                        i += 1
                return args

            def func(theta, x):
                newtheta = restore(args[:], theta)
                return objective(newtheta, x)

        return x0, func, restore, args

    def _moment_error(self, theta, x, data_moments):
        loc, scale, args = self._unpack_loc_scale(theta)
        if not self._argcheck(*args) or scale <= 0:
            return inf

        dist_moments = np.array([self.moment(i+1, *args, loc=loc, scale=scale)
                                 for i in range(len(data_moments))])
        if np.any(np.isnan(dist_moments)):
            raise ValueError("Method of moments encountered a non-finite "
                             "distribution moment and cannot continue. "
                             "Consider trying method='MLE'.")

        return (((data_moments - dist_moments) /
                 np.maximum(np.abs(data_moments), 1e-8))**2).sum()

    def fit(self, data, *args, **kwds):
        method = kwds.get('method', "mle").lower()

        censored = isinstance(data, CensoredData)
        if censored:
            if method != 'mle':
                raise ValueError('For censored data, the method must'
                                 ' be "MLE".')
            if data.num_censored() == 0:
                # There are no censored values in data, so replace the
                # CensoredData instance with a regular array.
                data = data._uncensored
                censored = False

        Narg = len(args)
        if Narg > self.numargs:
            raise TypeError("Too many input arguments.")

        # Check the finiteness of data only if data is not an instance of
        # CensoredData.  The arrays in a CensoredData instance have already
        # been validated.
        if not censored:
            # Note: `ravel()` is called for backwards compatibility.
            data = np.asarray(data).ravel()
            if not np.isfinite(data).all():
                raise ValueError("The data contains non-finite values.")

        start = [None]*2
        if (Narg < self.numargs) or not ('loc' in kwds and
                                         'scale' in kwds):
            # get distribution specific starting locations
            start = self._fitstart(data)
            args += start[Narg:-2]
        loc = kwds.pop('loc', start[-2])
        scale = kwds.pop('scale', start[-1])
        args += (loc, scale)
        x0, func, restore, args = self._reduce_func(args, kwds, data=data)
        optimizer = kwds.pop('optimizer', optimize.fmin)
        # convert string to function in scipy.optimize
        optimizer = _fit_determine_optimizer(optimizer)
        # by now kwds must be empty, since everybody took what they needed
        if kwds:
            raise TypeError(f"Unknown arguments: {kwds}.")

        vals = optimizer(func, x0, args=(data,), disp=0)
        obj = func(vals, data)

        if restore is not None:
            vals = restore(args, vals)
        vals = tuple(vals)

        loc, scale, shapes = self._unpack_loc_scale(vals)
        if not (np.all(self._argcheck(*shapes)) and scale > 0):
            raise FitError("Optimization converged to parameters that are "
                           "outside the range allowed by the distribution.")

        if method == 'mm':
            if not np.isfinite(obj):
                raise FitError("Optimization failed: either a data moment "
                               "or fitted distribution moment is "
                               "non-finite.")

        return vals

    def _fit_loc_scale_support(self, data, *args):
        if isinstance(data, CensoredData):
            # For this estimate, "uncensor" the data by taking the
            # given endpoints as the data for the left- or right-censored
            # data, and the mean for the interval-censored data.
            data = data._uncensor()
        else:
            data = np.asarray(data)

        # Estimate location and scale according to the method of moments.
        loc_hat, scale_hat = self.fit_loc_scale(data, *args)

        # Compute the support according to the shape parameters.
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        a, b = _a, _b
        support_width = b - a

        # If the support is empty then return the moment-based estimates.
        if support_width <= 0:
            return loc_hat, scale_hat

        # Compute the proposed support according to the loc and scale
        # estimates.
        a_hat = loc_hat + a * scale_hat
        b_hat = loc_hat + b * scale_hat

        # Use the moment-based estimates if they are compatible with the data.
        data_a = np.min(data)
        data_b = np.max(data)
        if a_hat < data_a and data_b < b_hat:
            return loc_hat, scale_hat

        # Otherwise find other estimates that are compatible with the data.
        data_width = data_b - data_a
        rel_margin = 0.1
        margin = data_width * rel_margin

        # For a finite interval, both the location and scale
        # should have interesting values.
        if support_width < np.inf:
            loc_hat = (data_a - a) - margin
            scale_hat = (data_width + 2 * margin) / support_width
            return loc_hat, scale_hat

        # For a one-sided interval, use only an interesting location parameter.
        if a > -np.inf:
            return (data_a - a) - margin, 1
        elif b < np.inf:
            return (data_b - b) + margin, 1
        else:
            raise RuntimeError

    def fit_loc_scale(self, data, *args):
        mu, mu2 = self.stats(*args, **{'moments': 'mv'})
        tmp = asarray(data)
        muhat = tmp.mean()
        mu2hat = tmp.var()
        Shat = sqrt(mu2hat / mu2)
        with np.errstate(invalid='ignore'):
            Lhat = muhat - Shat*mu
        if not np.isfinite(Lhat):
            Lhat = 0
        if not (np.isfinite(Shat) and (0 < Shat)):
            Shat = 1
        return Lhat, Shat

    def _entropy(self, *args):
        def integ(x):
            val = self._pdf(x, *args)
            return entr(val)

        # upper limit is often inf, so suppress warnings when integrating
        _a, _b = self._get_support(*args)
        with np.errstate(over='ignore'):
            h = quad(integ, _a, _b)[0]

        if not np.isnan(h):
            return h
        else:
            # try with different limits if integration problems
            low, upp = self.ppf([1e-10, 1. - 1e-10], *args)
            if np.isinf(_b):
                upper = upp
            else:
                upper = _b
            if np.isinf(_a):
                lower = low
            else:
                lower = _a
            return quad(integ, lower, upper)[0]

    def expect(self, func=None, args=(), loc=0, scale=1, lb=None, ub=None,
               conditional=False, **kwds):
        lockwds = {'loc': loc,
                   'scale': scale}
        self._argcheck(*args)
        _a, _b = self._get_support(*args)
        if func is None:
            def fun(x, *args):
                return x * self.pdf(x, *args, **lockwds)
        else:
            def fun(x, *args):
                return func(x) * self.pdf(x, *args, **lockwds)
        if lb is None:
            lb = loc + _a * scale
        if ub is None:
            ub = loc + _b * scale

        cdf_bounds = self.cdf([lb, ub], *args, **lockwds)
        invfac = cdf_bounds[1] - cdf_bounds[0]

        kwds['args'] = args

        # split interval to help integrator w/ infinite support; see gh-8928
        alpha = 0.05  # split body from tails at probability mass `alpha`
        inner_bounds = np.array([alpha, 1-alpha])
        cdf_inner_bounds = cdf_bounds[0] + invfac * inner_bounds
        c, d = loc + self._ppf(cdf_inner_bounds, *args) * scale

        # Do not silence warnings from integration.
        lbc = quad(fun, lb, c, **kwds)[0]
        cd = quad(fun, c, d, **kwds)[0]
        dub = quad(fun, d, ub, **kwds)[0]
        vals = (lbc + cd + dub)

        if conditional:
            vals /= invfac
        return np.array(vals)[()]  # make it a numpy scalar like other methods

    def _param_info(self):
        shape_info = self._shape_info()
        loc_info = _ShapeInfo("loc", False, (-np.inf, np.inf), (False, False))
        scale_info = _ShapeInfo("scale", False, (0, np.inf), (False, False))
        param_info = shape_info + [loc_info, scale_info]
        return param_info

    # For now, _delta_cdf is a private method.
    def _delta_cdf(self, x1, x2, *args, loc=0, scale=1):
        cdf1 = self.cdf(x1, *args, loc=loc, scale=scale)
        result = np.where(cdf1 > 0.5,
                          (self.sf(x1, *args, loc=loc, scale=scale)
                           - self.sf(x2, *args, loc=loc, scale=scale)),
                          self.cdf(x2, *args, loc=loc, scale=scale) - cdf1)
        if result.ndim == 0:
            result = result[()]
        return result


def _lazywhere(cond, arrays, f, fillvalue=None, f2=None):

    if (f2 is fillvalue is None) or (f2 is not None and fillvalue is not None):
        raise ValueError("Exactly one of `fillvalue` or `f2` must be given.")

    args = xp.broadcast_arrays(cond, *arrays)
    bool_dtype = xp.asarray([True]).dtype  # numpy 1.xx doesn't have `bool`
    cond, arrays = xp.astype(args[0], bool_dtype, copy=False), args[1:]

    temp1 = xp.asarray(f(*(arr[cond] for arr in arrays)))

    if f2 is None:
        if type(fillvalue) in {bool, int, float, complex}:
            with np.errstate(invalid='ignore'):
                dtype = (temp1 * fillvalue).dtype
        else:
           dtype = xp.result_type(temp1.dtype, fillvalue)
        out = xp.full(cond.shape, dtype=dtype,
                      fill_value=xp.asarray(fillvalue, dtype=dtype))
    else:
        ncond = ~cond
        temp2 = xp.asarray(f2(*(arr[ncond] for arr in arrays)))
        dtype = xp.result_type(temp1, temp2)
        out = xp.empty(cond.shape, dtype=dtype)
        out[ncond] = temp2

    out[cond] = temp1

    return out


## Normal distribution

# loc = mu, scale = std
# Keep these implementations out of the class definition so they can be reused
# by other distributions.
_norm_pdf_C = np.sqrt(2*np.pi)
_norm_pdf_logC = np.log(_norm_pdf_C)


def _norm_pdf(x):
    return np.exp(-x**2/2.0) / _norm_pdf_C


def _norm_logpdf(x):
    return -x**2 / 2.0 - _norm_pdf_logC


def _norm_cdf(x):
    return ndtr(x)


def _norm_logcdf(x):
    return log_ndtr(x)


def _norm_ppf(q):
    return ndtri(q)


def _norm_sf(x):
    return _norm_cdf(-x)


def _norm_logsf(x):
    return _norm_logcdf(-x)


def _norm_isf(q):
    return -_norm_ppf(q)


def _remove_optimizer_parameters(kwds):
    kwds.pop('loc', None)
    kwds.pop('scale', None)
    kwds.pop('optimizer', None)
    kwds.pop('method', None)
    if kwds:
        raise TypeError("Unknown arguments: %s." % kwds)


class norm_gen(rv_continuous):
    def _shape_info(self):
        return []

    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)

    def _pdf(self, x):
        # norm.pdf(x) = exp(-x**2/2)/sqrt(2*pi)
        return _norm_pdf(x)

    def _logpdf(self, x):
        return _norm_logpdf(x)

    def _cdf(self, x):
        return _norm_cdf(x)

    def _logcdf(self, x):
        return _norm_logcdf(x)

    def _sf(self, x):
        return _norm_sf(x)

    def _logsf(self, x):
        return _norm_logsf(x)

    def _ppf(self, q):
        return _norm_ppf(q)

    def _isf(self, q):
        return _norm_isf(q)

    def _stats(self):
        return 0.0, 1.0, 0.0, 0.0

    def _entropy(self):
        return 0.5*(np.log(2*np.pi)+1)

    def fit(self, data, **kwds):
        floc = kwds.pop('floc', None)
        fscale = kwds.pop('fscale', None)

        _remove_optimizer_parameters(kwds)
        _remove_optimizer_parameters(kwds)

        if floc is not None and fscale is not None:
            # This check is for consistency with `rv_continuous.fit`.
            # Without this check, this function would just return the
            # parameters that were given.
            raise ValueError("All parameters fixed. There is nothing to "
                             "optimize.")

        data = np.asarray(data)

        if not np.isfinite(data).all():
            raise ValueError("The data contains non-finite values.")

        if floc is None:
            loc = data.mean()
        else:
            loc = floc

        if fscale is None:
            scale = np.sqrt(((data - loc)**2).mean())
        else:
            scale = fscale

        return loc, scale

    def _munp(self, n):
        if n % 2 == 0:
            return factorial2(n - 1)
        else:
            return 0.


norm = norm_gen(name='norm')


def _lazyselect(condlist, choicelist, arrays, default=0):
    arrays = np.broadcast_arrays(*arrays)
    tcode = np.mintypecode([a.dtype.char for a in arrays])
    out = np.full(np.shape(arrays[0]), fill_value=default, dtype=tcode)
    for func, cond in zip(choicelist, condlist):
        if np.all(cond is False):
            continue
        cond, _ = np.broadcast_arrays(cond, arrays[0])
        temp = tuple(np.extract(cond, arr) for arr in arrays)
        np.place(out, cond, func(*temp))
    return out

class t_gen(rv_continuous):
    def _shape_info(self):
        return [_ShapeInfo("df", False, (0, np.inf), (False, False))]

    def _rvs(self, df, size=None, random_state=None):
        return random_state.standard_t(df, size=size)

    def _pdf(self, x, df):
        return _lazywhere(
            df == np.inf, (x, df),
            f=lambda x, df: norm._pdf(x),
            f2=lambda x, df: (
                np.exp(self._logpdf(x, df))
            )
        )

    def _logpdf(self, x, df):

        def t_logpdf(x, df):
            return (np.log(poch(0.5 * df, 0.5))
                    - 0.5 * (np.log(df) + np.log(np.pi))
                    - (df + 1)/2*np.log1p(x * x/df))

        def norm_logpdf(x, df):
            return norm._logpdf(x)

        return _lazywhere(df == np.inf, (x, df, ), f=norm_logpdf, f2=t_logpdf)

    def _cdf(self, x, df):
        return stdtr(df, x)

    def _sf(self, x, df):
        return stdtr(df, -x)

    def _ppf(self, q, df):
        return stdtrit(df, q)

    def _isf(self, q, df):
        return -stdtrit(df, q)

    def _stats(self, df):
        # infinite df -> normal distribution (0.0, 1.0, 0.0, 0.0)
        infinite_df = np.isposinf(df)

        mu = np.where(df > 1, 0.0, np.inf)

        condlist = ((df > 1) & (df <= 2),
                    (df > 2) & np.isfinite(df),
                    infinite_df)
        choicelist = (lambda df: np.broadcast_to(np.inf, df.shape),
                      lambda df: df / (df-2.0),
                      lambda df: np.broadcast_to(1, df.shape))
        mu2 = _lazyselect(condlist, choicelist, (df,), np.nan)

        g1 = np.where(df > 3, 0.0, np.nan)

        condlist = ((df > 2) & (df <= 4),
                    (df > 4) & np.isfinite(df),
                    infinite_df)
        choicelist = (lambda df: np.broadcast_to(np.inf, df.shape),
                      lambda df: 6.0 / (df-4.0),
                      lambda df: np.broadcast_to(0, df.shape))
        g2 = _lazyselect(condlist, choicelist, (df,), np.nan)

        return mu, mu2, g1, g2

    def _entropy(self, df):
        if df == np.inf:
            return norm._entropy()

        def regular(df):
            half = df/2
            half1 = (df + 1)/2
            return (half1*(digamma(half1) - digamma(half))
                    + np.log(np.sqrt(df)*beta(half, 0.5)))

        def asymptotic(df):
            h = (norm._entropy() + 1/df + (df**-2.)/4 - (df**-3.)/6
                 - (df**-4.)/8 + 3/10*(df**-5.) + (df**-6.)/4)
            return h

        h = _lazywhere(df >= 100, (df, ), f=asymptotic, f2=regular)
        return h
