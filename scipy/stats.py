
from collections import namedtuple
from typing import Sequence
import numpy as np
from numpy import (isscalar, r_, log, around, unique, asarray, zeros,
                   arange, sort, amin, amax, sqrt, array,
                   pi, exp, ravel, count_nonzero)
from ._ansari_swilk_statistics import swilk
from dataclasses import dataclass, field
import inspect
import numbers
from .special import ndtr, ndtri, comb, factorial, betainc, betaincc, fdtrc, binom
from itertools import combinations, permutations, product
import functools
from keyword import iskeyword as _iskeyword
from . import distributions
from math import gcd
from ._continuous_distns import t

if np.lib.NumpyVersion(np.__version__) >= '1.25.0':
    from numpy.exceptions import (
        AxisError, 
        DTypePromotionError
    )
else:
    from numpy import (  # type: ignore[attr-defined, no-redef]
        AxisError, 
    )
    DTypePromotionError = TypeError  # type: ignore
try:
    from numpy.random import Generator as Generator
except ImportError:
    class Generator:  # type: ignore[no-redef]
        pass
xp = np

xp_vector_norm = np.linalg.vector_norm

ShapiroResult = namedtuple('ShapiroResult', ('statistic', 'pvalue'))
def shapiro(x):
    x = np.ravel(x).astype(np.float64)

    N = len(x)
    if N < 3:
        raise ValueError("Data must be at least length 3.")

    a = zeros(N//2, dtype=np.float64)
    init = 0

    y = sort(x)
    y -= x[N//2]  # subtract the median (or a nearby value); see gh-15777

    w, pw, ifault = swilk(y, a, init)

    return ShapiroResult(np.float64(w), np.float64(pw))

# temporary substitute for xp.moveaxis, which is not yet in all backends
# or covered by array_api_compat.
def xp_moveaxis_to_end(
        x,
        source: int,
        /, *,
        xp):
    axes = list(range(x.ndim))
    temp = axes.pop(source)
    axes = axes + [temp]
    return xp.permute_dims(x, axes)


AxisError: type[Exception]
def _broadcast_shapes(shapes, axis=None):
    """
    Broadcast shapes, ignoring incompatibility of specified axes
    """
    if not shapes:
        return shapes

    # input validation
    if axis is not None:
        axis = np.atleast_1d(axis)
        message = '`axis` must be an integer, a tuple of integers, or `None`.'
        try:
            with np.errstate(invalid='ignore'):
                axis_int = axis.astype(int)
        except ValueError as e:
            raise AxisError(message) from e
        if not np.array_equal(axis_int, axis):
            raise AxisError(message)
        axis = axis_int

    # First, ensure all shapes have same number of dimensions by prepending 1s.
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = np.ones((len(shapes), n_dims), dtype=int)
    for row, shape in zip(new_shapes, shapes):
        row[len(row)-len(shape):] = shape  # can't use negative indices (-0:)

    # Remove the shape elements of the axes to be ignored, but remember them.
    if axis is not None:
        axis[axis < 0] = n_dims + axis[axis < 0]
        axis = np.sort(axis)
        if axis[-1] >= n_dims or axis[0] < 0:
            message = (f"`axis` is out of bounds "
                       f"for array of dimension {n_dims}")
            raise AxisError(message)

        if len(np.unique(axis)) != len(axis):
            raise AxisError("`axis` must contain only distinct elements")

        removed_shapes = new_shapes[:, axis]
        new_shapes = np.delete(new_shapes, axis, axis=1)

    new_shape = np.max(new_shapes, axis=0)
    # except in case of an empty array:
    new_shape *= new_shapes.all(axis=0)

    if np.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        raise ValueError("Array shapes are incompatible for broadcasting.")

    if axis is not None:
        # Add back the shape elements that were ignored
        new_axis = axis - np.arange(len(axis))
        new_shapes = [tuple(np.insert(new_shape, new_axis, removed_shape))
                      for removed_shape in removed_shapes]
        return new_shapes
    else:
        return tuple(new_shape)


def _broadcast_arrays(arrays, axis=None, xp=None):
    if not arrays:
        return arrays
    arrays = [xp.asarray(arr) for arr in arrays]
    shapes = [arr.shape for arr in arrays]
    new_shapes = _broadcast_shapes(shapes, axis)
    if axis is None:
        new_shapes = [new_shapes]*len(arrays)
    return [xp.broadcast_to(array, new_shape)
            for array, new_shape in zip(arrays, new_shapes)]

def _broadcast_concatenate(arrays, axis, paired=False):
    arrays = _broadcast_arrays(arrays, axis if not paired else None)
    res = np.concatenate(arrays, axis=axis)
    return res

def _vectorize_statistic(statistic):
    def stat_nd(*data, axis=0):
        lengths = [sample.shape[axis] for sample in data]
        split_indices = np.cumsum(lengths)[:-1]
        z = _broadcast_concatenate(data, axis)
        z = np.moveaxis(z, axis, 0)

        def stat_1d(z):
            data = np.split(z, split_indices)
            return statistic(*data)

        return np.apply_along_axis(stat_1d, 0, z)[()]
    return stat_nd

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

def _permutation_test_iv(data, statistic, permutation_type, vectorized,
                         n_resamples, batch, alternative, axis, rng):
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    permutation_types = {'samples', 'pairings', 'independent'}
    permutation_type = permutation_type.lower()
    if permutation_type not in permutation_types:
        raise ValueError(f"`permutation_type` must be in {permutation_types}.")

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters

    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    message = "`data` must be a tuple containing at least two samples"
    try:
        if len(data) < 2 and permutation_type == 'independent':
            raise ValueError(message)
    except TypeError:
        raise TypeError(message)

    data = _broadcast_arrays(data, axis)
    data_iv = []
    for sample in data:
        sample = np.atleast_1d(sample)
        if sample.shape[axis] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    n_resamples_int = (int(n_resamples) if not np.isinf(n_resamples)
                       else np.inf)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    rng = check_random_state(rng)

    return (data_iv, statistic, permutation_type, vectorized, n_resamples_int,
            batch_iv, alternative, axis_int, rng)


def _batch_generator(iterable, batch):
    iterator = iter(iterable)
    if batch <= 0:
        raise ValueError("`batch` must be positive.")
    z = [item for i, item in zip(range(batch), iterator)]
    while z:  # we don't want StopIteration without yielding an empty list
        yield z
        z = [item for i, item in zip(range(batch), iterator)]

def _pairings_permutations_gen(n_permutations, n_samples, n_obs_sample, batch,
                               rng):
    batch = min(batch, n_permutations)

    if hasattr(rng, 'permuted'):
        def batched_perm_generator():
            indices = np.arange(n_obs_sample)
            indices = np.tile(indices, (batch, n_samples, 1))
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations-k)
                # Don't permute in place, otherwise results depend on `batch`
                permuted_indices = rng.permuted(indices, axis=-1)
                yield permuted_indices[:batch_actual]
    else:  # RandomState and early Generators don't have `permuted`
        def batched_perm_generator():
            for k in range(0, n_permutations, batch):
                batch_actual = min(batch, n_permutations-k)
                size = (batch_actual, n_samples, n_obs_sample)
                x = rng.random(size=size)
                yield np.argsort(x, axis=-1)[:batch_actual]

    return batched_perm_generator()

def _calculate_null_pairings(data, statistic, n_permutations, batch,
                             rng=None):
    n_samples = len(data)

    # compute number of permutations (factorial(n) permutations of each sample)
    n_obs_sample = data[0].shape[-1]  # observations per sample; same for each
    n_max = factorial(n_obs_sample)**n_samples

    # `perm_generator` is an iterator that produces a list of permutations of
    # indices from 0 to n_obs_sample, one for each sample.
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        batch = batch or int(n_permutations)
        # Cartesian product of the sets of all permutations of indices
        perm_generator = product(*(permutations(range(n_obs_sample))
                                   for i in range(n_samples)))
        batched_perm_generator = _batch_generator(perm_generator, batch=batch)
    else:
        exact_test = False
        batch = batch or int(n_permutations)
        args = n_permutations, n_samples, n_obs_sample, batch, rng
        batched_perm_generator = _pairings_permutations_gen(*args)

    null_distribution = []

    for indices in batched_perm_generator:
        indices = np.array(indices)

        data_batch = [None]*n_samples
        for i in range(n_samples):
            data_batch[i] = data[i][..., indices[i]]
            data_batch[i] = np.moveaxis(data_batch[i], -2, 0)

        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test

def _calculate_null_samples(data, statistic, n_permutations, batch,
                            rng=None):
    n_samples = len(data)
    if n_samples == 1:
        data = [data[0], -data[0]]
    data = np.swapaxes(data, 0, -1)
    def statistic_wrapped(*data, axis):
        data = np.swapaxes(data, 0, -1)
        if n_samples == 1:
            data = data[0:1]
        return statistic(*data, axis=axis)

    return _calculate_null_pairings(data, statistic_wrapped, n_permutations,
                                    batch, rng)


def _all_partitions_concatenated(ns):
    def all_partitions(z, n):
        for c in combinations(z, n):
            x0 = set(c)
            x1 = z - x0
            yield [x0, x1]

    def all_partitions_n(z, ns):
        if len(ns) == 0:
            yield [z]
            return
        for c in all_partitions(z, ns[0]):
            for d in all_partitions_n(c[1], ns[1:]):
                yield c[0:1] + d

    z = set(range(np.sum(ns)))
    for partitioning in all_partitions_n(z, ns[:]):
        x = np.concatenate([list(partition)
                            for partition in partitioning]).astype(int)
        yield x

def _calculate_null_both(data, statistic, n_permutations, batch,
                         rng=None):
    n_samples = len(data)

    # compute number of permutations
    # (distinct partitions of data into samples of these sizes)
    n_obs_i = [sample.shape[-1] for sample in data]  # observations per sample
    n_obs_ic = np.cumsum(n_obs_i)
    n_obs = n_obs_ic[-1]  # total number of observations
    n_max = np.prod([comb(n_obs_ic[i], n_obs_ic[i-1])
                     for i in range(n_samples-1, 0, -1)])
    if n_permutations >= n_max:
        exact_test = True
        n_permutations = n_max
        perm_generator = _all_partitions_concatenated(n_obs_i)
    else:
        exact_test = False
        perm_generator = (rng.permutation(n_obs)
                          for i in range(n_permutations))

    batch = batch or int(n_permutations)
    null_distribution = []
    data = np.concatenate(data, axis=-1)
    for indices in _batch_generator(perm_generator, batch=batch):
        indices = np.array(indices)
        data_batch = data[..., indices]
        data_batch = np.moveaxis(data_batch, -2, 0)
        data_batch = np.split(data_batch, n_obs_ic[:-1], axis=-1)
        null_distribution.append(statistic(*data_batch, axis=-1))
    null_distribution = np.concatenate(null_distribution, axis=0)

    return null_distribution, n_permutations, exact_test

@dataclass
class PermutationTestResult:
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray


# SPEC 7
def _transition_to_rng(old_name, *, position_num=None, end_version=None,
                       replace_doc=True):
    NEW_NAME = "rng"

    cmn_msg = (
        "To silence this warning and ensure consistent behavior in SciPy "
        f"{end_version}, control the RNG using argument `{NEW_NAME}`. Arguments passed "
        f"to keyword `{NEW_NAME}` will be validated by `np.random.default_rng`, so the "
        "behavior corresponding with a given value may change compared to use of "
        f"`{old_name}`. For example, "
        "1) `None` will result in unpredictable random numbers, "
        "2) an integer will result in a different stream of random numbers, (with the "
        "same distribution), and "
        "3) `np.random` or `RandomState` instances will result in an error. "
        "See the documentation of `default_rng` for more information."
    )

    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            # Determine how PRNG was passed
            as_old_kwarg = old_name in kwargs
            as_new_kwarg = NEW_NAME in kwargs
            as_pos_arg = position_num is not None and len(args) >= position_num + 1
            emit_warning = end_version is not None

            # Can only specify PRNG one of the three ways
            if int(as_old_kwarg) + int(as_new_kwarg) + int(as_pos_arg) > 1:
                message = (
                    f"{fun.__name__}() got multiple values for "
                    f"argument now known as `{NEW_NAME}`. Specify one of "
                    f"`{NEW_NAME}` or `{old_name}`."
                )
                raise TypeError(message)

            # Check whether global random state has been set
            global_seed_set = np.random.mtrand._rand._bit_generator._seed_seq is None

            if as_old_kwarg:  # warn about deprecated use of old kwarg
                kwargs[NEW_NAME] = kwargs.pop(old_name)

            elif as_pos_arg:
                arg = args[position_num]
                ok_classes = (
                    np.random.Generator,
                    np.random.SeedSequence,
                    np.random.BitGenerator,
                )
                if (arg is None and not global_seed_set) or isinstance(arg, ok_classes):
                    pass

            elif as_new_kwarg:  # no warnings; this is the preferred use
                # After the removal of the decorator, normalization with
                # np.random.default_rng will be done inside the decorated function
                kwargs[NEW_NAME] = np.random.default_rng(kwargs[NEW_NAME])


            return fun(*args, **kwargs)

        return wrapper

    return decorator

@_transition_to_rng('random_state')
def permutation_test(data, statistic, *, permutation_type='independent',
                     vectorized=None, n_resamples=9999, batch=None,
                     alternative="two-sided", axis=0, rng=None):
    args = _permutation_test_iv(data, statistic, permutation_type, vectorized,
                                n_resamples, batch, alternative, axis,
                                rng)
    (data, statistic, permutation_type, vectorized, n_resamples, batch,
     alternative, axis, rng) = args

    observed = statistic(*data, axis=-1)

    null_calculators = {"pairings": _calculate_null_pairings,
                        "samples": _calculate_null_samples,
                        "independent": _calculate_null_both}
    null_calculator_args = (data, statistic, n_resamples,
                            batch, rng)
    calculate_null = null_calculators[permutation_type]
    null_distribution, n_resamples, exact_test = (
        calculate_null(*null_calculator_args))

    # See References [2] and [3]
    adjustment = 0 if exact_test else 1

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps =  (0 if not np.issubdtype(observed.dtype, np.inexact)
            else np.finfo(observed.dtype).eps*100)
    gamma = np.abs(eps * observed)

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        pvalues = (cmps.sum(axis=0) + adjustment) / (n_resamples + adjustment)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = np.clip(pvalues, 0, 1)

    return PermutationTestResult(observed, pvalues, null_distribution)

@dataclass
class ResamplingMethod:
    n_resamples: int = 9999
    batch: int = None  # type: ignore[assignment]


@dataclass
class MonteCarloMethod(ResamplingMethod):
    rvs: object = None

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    rvs=self.rvs)

@dataclass
class PermutationMethod(ResamplingMethod):
    rng: object  # type: ignore[misc]
    _rng: object = field(init=False, repr=False, default=None)  # type: ignore[assignment]

    @property
    def random_state(self):
        # Uncomment in SciPy 1.17.0
        # warnings.warn(_rs_deprecation, DeprecationWarning, stacklevel=2)
        return self._rng

    @random_state.setter
    def random_state(self, val):
        # Uncomment in SciPy 1.17.0
        # warnings.warn(_rs_deprecation, DeprecationWarning, stacklevel=2)
        self._rng = val

    @property  # type: ignore[no-redef]
    def rng(self):  # noqa: F811
        return self._rng

    @random_state.setter
    def rng(self, val):  # noqa: F811
        self._rng = np.random.default_rng(val)

    @_transition_to_rng('random_state', position_num=3, replace_doc=False)
    def __init__(self, n_resamples=9999, batch=None, rng=None):
        self._rng = rng  # don't validate with `default_rng` during SPEC 7 transition
        super().__init__(n_resamples=n_resamples, batch=batch)

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        return dict(n_resamples=self.n_resamples, batch=self.batch, rng=self.rng)


def _validate_names(typename, field_names, extra_field_names):
    for name in [typename] + field_names + extra_field_names:
        if not isinstance(name, str):
            raise TypeError('typename and all field names must be strings')
        if not name.isidentifier():
            raise ValueError('typename and all field names must be valid '
                             f'identifiers: {name!r}')
        if _iskeyword(name):
            raise ValueError('typename and all field names cannot be a '
                             f'keyword: {name!r}')

    seen = set()
    for name in field_names + extra_field_names:
        if name.startswith('_'):
            raise ValueError('Field names cannot start with an underscore: '
                             f'{name!r}')
        if name in seen:
            raise ValueError(f'Duplicate field name: {name!r}')
        seen.add(name)


# Note: This code is adapted from CPython:Lib/collections/__init__.py
def _make_tuple_bunch(typename, field_names, extra_field_names=None,
                      module=None):
    if len(field_names) == 0:
        raise ValueError('field_names must contain at least one name')

    if extra_field_names is None:
        extra_field_names = []
    _validate_names(typename, field_names, extra_field_names)

    typename = _sys.intern(str(typename))
    field_names = tuple(map(_sys.intern, field_names))
    extra_field_names = tuple(map(_sys.intern, extra_field_names))

    all_names = field_names + extra_field_names
    arg_list = ', '.join(field_names)
    full_list = ', '.join(all_names)
    repr_fmt = ''.join(('(',
                        ', '.join(f'{name}=%({name})r' for name in all_names),
                        ')'))
    tuple_new = tuple.__new__
    _dict, _tuple, _zip = dict, tuple, zip

    # Create all the named tuple methods to be added to the class namespace

    s = f"""\
def __new__(_cls, {arg_list}, **extra_fields):
    return _tuple_new(_cls, ({arg_list},))

def __init__(self, {arg_list}, **extra_fields):
    for key in self._extra_fields:
        if key not in extra_fields:
            raise TypeError("missing keyword argument '%s'" % (key,))
    for key, val in extra_fields.items():
        if key not in self._extra_fields:
            raise TypeError("unexpected keyword argument '%s'" % (key,))
        self.__dict__[key] = val

def __setattr__(self, key, val):
    if key in {repr(field_names)}:
        raise AttributeError("can't set attribute %r of class %r"
                             % (key, self.__class__.__name__))
    else:
        self.__dict__[key] = val
"""
    del arg_list
    namespace = {'_tuple_new': tuple_new,
                 '__builtins__': dict(TypeError=TypeError,
                                      AttributeError=AttributeError),
                 '__name__': f'namedtuple_{typename}'}
    exec(s, namespace)
    __new__ = namespace['__new__']
    __new__.__doc__ = f'Create new instance of {typename}({full_list})'
    __init__ = namespace['__init__']
    __init__.__doc__ = f'Instantiate instance of {typename}({full_list})'
    __setattr__ = namespace['__setattr__']

    def __repr__(self):
        'Return a nicely formatted representation string'
        return self.__class__.__name__ + repr_fmt % self._asdict()

    def _asdict(self):
        'Return a new dict which maps field names to their values.'
        out = _dict(_zip(self._fields, self))
        out.update(self.__dict__)
        return out

    def __getnewargs_ex__(self):
        'Return self as a plain tuple.  Used by copy and pickle.'
        return _tuple(self), self.__dict__

    # Modify function metadata to help with introspection and debugging
    for method in (__new__, __repr__, _asdict, __getnewargs_ex__):
        method.__qualname__ = f'{typename}.{method.__name__}'

    # Build-up the class namespace dictionary
    # and use type() to build the result class
    class_namespace = {
        '__doc__': f'{typename}({full_list})',
        '_fields': field_names,
        '__new__': __new__,
        '__init__': __init__,
        '__repr__': __repr__,
        '__setattr__': __setattr__,
        '_asdict': _asdict,
        '_extra_fields': extra_field_names,
        '__getnewargs_ex__': __getnewargs_ex__,
    }
    for index, name in enumerate(field_names):

        def _get(self, index=index):
            return self[index]
        class_namespace[name] = property(_get)
    for name in extra_field_names:

        def _get(self, name=name):
            return self.__dict__[name]
        class_namespace[name] = property(_get)

    result = type(typename, (tuple,), class_namespace)

    if module is None:
        try:
            module = _sys._getframe(1).f_globals.get('__name__', '__main__')
        except (AttributeError, ValueError):
            pass
    if module is not None:
        result.__module__ = module
        __new__.__module__ = module

    return result

PearsonRResultBase = _make_tuple_bunch('PearsonRResultBase',
                                       ['statistic', 'pvalue'], [])


@dataclass
class BootstrapMethod(ResamplingMethod):
    rng: object  # type: ignore[misc]
    _rng: object = field(init=False, repr=False, default=None)  # type: ignore[assignment]
    method: str = 'BCa'

    @property
    def random_state(self):
        return self._rng

    @random_state.setter
    def random_state(self, val):
        self._rng = val

    @property  # type: ignore[no-redef]
    def rng(self):  # noqa: F811
        return self._rng

    @random_state.setter
    def rng(self, val):  # noqa: F811
        self._rng = np.random.default_rng(val)

    @_transition_to_rng('random_state', position_num=3, replace_doc=False)
    def __init__(self, n_resamples=9999, batch=None, rng=None, method='BCa'):
        self._rng = rng  # don't validate with `default_rng` during SPEC 7 transition
        self.method = method
        super().__init__(n_resamples=n_resamples, batch=batch)

    def _asdict(self):
        # `dataclasses.asdict` deepcopies; we don't want that.
        return dict(n_resamples=self.n_resamples, batch=self.batch,
                    random_state=self.random_state, method=self.method)

ConfidenceInterval = namedtuple("ConfidenceInterval", ["low", "high"])

def _bootstrap_iv(data, statistic, vectorized, paired, axis, confidence_level,
                  alternative, n_resamples, batch, method, bootstrap_result,
                  rng):
    """Input validation and standardization for `bootstrap`."""

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if vectorized is None:
        vectorized = 'axis' in inspect.signature(statistic).parameters

    if not vectorized:
        statistic = _vectorize_statistic(statistic)

    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    n_samples = 0
    try:
        n_samples = len(data)
    except TypeError:
        raise ValueError("`data` must be a sequence of samples.")

    if n_samples == 0:
        raise ValueError("`data` must contain at least one sample.")

    message = ("Ignoring the dimension specified by `axis`, arrays in `data` do not "
               "have the same shape. Beginning in SciPy 1.16.0, `bootstrap` will "
               "explicitly broadcast elements of `data` to the same shape (ignoring "
               "`axis`) before performing the calculation. To avoid this warning in "
               "the meantime, ensure that all samples have the same shape (except "
               "potentially along `axis`).")
    data = [np.atleast_1d(sample) for sample in data]
    reduced_shapes = set()
    for sample in data:
        reduced_shape = list(sample.shape)
        reduced_shape.pop(axis)
        reduced_shapes.add(tuple(reduced_shape))

    data_iv = []
    for sample in data:
        if sample.shape[axis_int] <= 1:
            raise ValueError("each sample in `data` must contain two or more "
                             "observations along `axis`.")
        sample = np.moveaxis(sample, axis_int, -1)
        data_iv.append(sample)

    if paired not in {True, False}:
        raise ValueError("`paired` must be `True` or `False`.")

    if paired:
        n = data_iv[0].shape[-1]
        for sample in data_iv[1:]:
            if sample.shape[-1] != n:
                message = ("When `paired is True`, all samples must have the "
                           "same length along `axis`")
                raise ValueError(message)

        # to generate the bootstrap distribution for paired-sample statistics,
        # resample the indices of the observations
        def statistic(i, axis=-1, data=data_iv, unpaired_statistic=statistic):
            data = [sample[..., i] for sample in data]
            return unpaired_statistic(*data, axis=axis)

        data_iv = [np.arange(n)]

    confidence_level_float = float(confidence_level)

    alternative = alternative.lower()
    alternatives = {'two-sided', 'less', 'greater'}
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be one of {alternatives}")

    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int < 0:
        raise ValueError("`n_resamples` must be a non-negative integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    methods = {'percentile', 'basic', 'bca'}
    method = method.lower()
    if method not in methods:
        raise ValueError(f"`method` must be in {methods}")

    message = "`bootstrap_result` must have attribute `bootstrap_distribution'"
    if (bootstrap_result is not None
            and not hasattr(bootstrap_result, "bootstrap_distribution")):
        raise ValueError(message)

    message = ("Either `bootstrap_result.bootstrap_distribution.size` or "
               "`n_resamples` must be positive.")
    if ((not bootstrap_result or
         not bootstrap_result.bootstrap_distribution.size)
            and n_resamples_int == 0):
        raise ValueError(message)

    rng = check_random_state(rng)

    return (data_iv, statistic, vectorized, paired, axis_int,
            confidence_level_float, alternative, n_resamples_int, batch_iv,
            method, bootstrap_result, rng)


def rng_integers(gen, low, high=None, size=None, dtype='int64',
                 endpoint=False):
    if isinstance(gen, Generator):
        return gen.integers(low, high=high, size=size, dtype=dtype,
                            endpoint=endpoint)
    else:
        if gen is None:
            # default is RandomState singleton used by np.random.
            gen = np.random.mtrand._rand
        if endpoint:
            # inclusive of endpoint
            # remember that low and high can be arrays, so don't modify in
            # place
            if high is None:
                return gen.randint(low + 1, size=size, dtype=dtype)
            if high is not None:
                return gen.randint(low, high=high + 1, size=size, dtype=dtype)

        # exclusive
        return gen.randint(low, high=high, size=size, dtype=dtype)


def _bootstrap_resample(sample, n_resamples=None, rng=None):
    n = sample.shape[-1]

    # bootstrap - each row is a random resample of original observations
    i = rng_integers(rng, 0, n, (n_resamples, n))

    resamples = sample[..., i]
    return resamples

def _percentile_of_score(a, score, axis):
    B = a.shape[axis]
    return ((a < score).sum(axis=axis) + (a <= score).sum(axis=axis)) / (2 * B)


def _jackknife_resample(sample, batch=None):
    """Jackknife resample the sample. Only one-sample stats for now."""
    n = sample.shape[-1]
    batch_nominal = batch or n

    for k in range(0, n, batch_nominal):
        # col_start:col_end are the observations to remove
        batch_actual = min(batch_nominal, n-k)

        # jackknife - each row leaves out one observation
        j = np.ones((batch_actual, n), dtype=bool)
        np.fill_diagonal(j[:, k:k+batch_actual], False)
        i = np.arange(n)
        i = np.broadcast_to(i, (batch_actual, n))
        i = i[j].reshape((batch_actual, n-1))

        resamples = sample[..., i]
        yield resamples

def _bca_interval(data, statistic, axis, alpha, theta_hat_b, batch):
    """Bias-corrected and accelerated interval."""
    # closely follows [1] 14.3 and 15.4 (Eq. 15.36)

    # calculate z0_hat
    theta_hat = np.asarray(statistic(*data, axis=axis))[..., None]
    percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)
    z0_hat = ndtri(percentile)

    # calculate a_hat
    theta_hat_ji = []  # j is for sample of data, i is for jackknife resample
    for j, sample in enumerate(data):
        # _jackknife_resample will add an axis prior to the last axis that
        # corresponds with the different jackknife resamples. Do the same for
        # each sample of the data to ensure broadcastability. We need to
        # create a copy of the list containing the samples anyway, so do this
        # in the loop to simplify the code. This is not the bottleneck...
        samples = [np.expand_dims(sample, -2) for sample in data]
        theta_hat_i = []
        for jackknife_sample in _jackknife_resample(sample, batch):
            samples[j] = jackknife_sample
            broadcasted = _broadcast_arrays(samples, axis=-1)
            theta_hat_i.append(statistic(*broadcasted, axis=-1))
        theta_hat_ji.append(theta_hat_i)

    theta_hat_ji = [np.concatenate(theta_hat_i, axis=-1)
                    for theta_hat_i in theta_hat_ji]

    n_j = [theta_hat_i.shape[-1] for theta_hat_i in theta_hat_ji]

    theta_hat_j_dot = [theta_hat_i.mean(axis=-1, keepdims=True)
                       for theta_hat_i in theta_hat_ji]

    U_ji = [(n - 1) * (theta_hat_dot - theta_hat_i)
            for theta_hat_dot, theta_hat_i, n
            in zip(theta_hat_j_dot, theta_hat_ji, n_j)]

    nums = [(U_i**3).sum(axis=-1)/n**3 for U_i, n in zip(U_ji, n_j)]
    dens = [(U_i**2).sum(axis=-1)/n**2 for U_i, n in zip(U_ji, n_j)]
    a_hat = 1/6 * sum(nums) / sum(dens)**(3/2)

    # calculate alpha_1, alpha_2
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1/(1 - a_hat*num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2/(1 - a_hat*num2))
    return alpha_1, alpha_2, a_hat  # return a_hat for testing

def _percentile_along_axis(theta_hat_b, alpha):
    shape = theta_hat_b.shape[:-1]
    alpha = np.broadcast_to(alpha, shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for indices, alpha_i in np.ndenumerate(alpha):
        if np.isnan(alpha_i):
            percentiles[indices] = np.nan
        else:
            theta_hat_b_i = theta_hat_b[indices]
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]  # return scalar instead of 0d array


@dataclass
class BootstrapResult:
    confidence_interval: ConfidenceInterval
    bootstrap_distribution: np.ndarray
    standard_error: float | np.ndarray

@_transition_to_rng('random_state')
def bootstrap(data, statistic, *, n_resamples=9999, batch=None,
              vectorized=None, paired=False, axis=0, confidence_level=0.95,
              alternative='two-sided', method='BCa', bootstrap_result=None,
              rng=None):
    # Input validation
    args = _bootstrap_iv(data, statistic, vectorized, paired, axis,
                         confidence_level, alternative, n_resamples, batch,
                         method, bootstrap_result, rng)
    (data, statistic, vectorized, paired, axis, confidence_level,
     alternative, n_resamples, batch, method, bootstrap_result,
     rng) = args

    theta_hat_b = ([] if bootstrap_result is None
                   else [bootstrap_result.bootstrap_distribution])

    batch_nominal = batch or n_resamples or 1

    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples-k)
        # Generate resamples
        resampled_data = []
        for sample in data:
            resample = _bootstrap_resample(sample, n_resamples=batch_actual,
                                           rng=rng)
            resampled_data.append(resample)

        # Compute bootstrap distribution of statistic
        theta_hat_b.append(statistic(*resampled_data, axis=-1))
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)

    # Calculate percentile interval
    alpha = ((1 - confidence_level)/2 if alternative == 'two-sided'
             else (1 - confidence_level))
    if method == 'bca':
        interval = _bca_interval(data, statistic, axis=-1, alpha=alpha,
                                 theta_hat_b=theta_hat_b, batch=batch)[:2]
        percentile_fun = _percentile_along_axis
    else:
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)
    if method == 'basic':  # see [3]
        theta_hat = statistic(*data, axis=-1)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

    if alternative == 'less':
        ci_l = np.full_like(ci_l, -np.inf)
    elif alternative == 'greater':
        ci_u = np.full_like(ci_u, np.inf)

    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u),
                           bootstrap_distribution=theta_hat_b,
                           standard_error=np.std(theta_hat_b, ddof=1, axis=-1))

def _pearsonr_bootstrap_ci(confidence_level, method, x, y, alternative, axis):
    """
    Compute the confidence interval for Pearson's R using the bootstrap.
    """
    def statistic(x, y, axis):
        statistic, _ = pearsonr(x, y, axis=axis)
        return statistic

    res = bootstrap((x, y), statistic, confidence_level=confidence_level, axis=axis,
                    paired=True, alternative=alternative, **method._asdict())
    # for one-sided confidence intervals, bootstrap gives +/- inf on one side
    res.confidence_interval = np.clip(res.confidence_interval, -1, 1)

    return ConfidenceInterval(*res.confidence_interval)


def _pearsonr_fisher_ci(r, n, confidence_level, alternative):

    with np.errstate(divide='ignore'):
        zr = xp.atanh(r)

    ones = xp.ones_like(r)
    n = xp.asarray(n, dtype=r.dtype)
    confidence_level = xp.asarray(confidence_level, dtype=r.dtype)
    if n > 3:
        se = xp.sqrt(1 / (n - 3))
        if alternative == "two-sided":
            h = ndtri(0.5 + confidence_level/2)
            zlo = zr - h*se
            zhi = zr + h*se
            rlo = xp.tanh(zlo)
            rhi = xp.tanh(zhi)
        elif alternative == "less":
            h = ndtri(confidence_level)
            zhi = zr + h*se
            rhi = xp.tanh(zhi)
            rlo = -ones
        else:
            # alternative == "greater":
            h = ndtri(confidence_level)
            zlo = zr - h*se
            rlo = xp.tanh(zlo)
            rhi = ones
    else:
        rlo, rhi = -ones, ones

    rlo = rlo[()] if rlo.ndim == 0 else rlo
    rhi = rhi[()] if rhi.ndim == 0 else rhi
    return ConfidenceInterval(low=rlo, high=rhi)


class PearsonRResult(PearsonRResultBase):
    def __init__(self, statistic, pvalue, alternative, n, x, y, axis):
        super().__init__(statistic, pvalue)
        self._alternative = alternative
        self._n = n
        self._x = x
        self._y = y
        self._axis = axis

        # add alias for consistency with other correlation functions
        self.correlation = statistic

    def confidence_interval(self, confidence_level=0.95, method=None):
        if isinstance(method, BootstrapMethod):
            message = ('`method` must be `None` if `pearsonr` '
                       'arguments were not NumPy arrays.')

            ci = _pearsonr_bootstrap_ci(confidence_level, method, self._x, self._y,
                                        self._alternative, self._axis)
        elif method is None:
            ci = _pearsonr_fisher_ci(self.statistic, self._n, confidence_level,
                                     self._alternative)
        else:
            message = ('`method` must be an instance of `BootstrapMethod` '
                       'or None.')
            raise ValueError(message)
        return ci


class _SimpleBeta:
    def __init__(self, a, b, *, loc=None, scale=None):
        self.a = a
        self.b = b
        self.loc = loc
        self.scale = scale

    def cdf(self, x):
        if self.loc is not None or self.scale is not None:
            loc = 0 if self.loc is None else self.loc
            scale = 1 if self.scale is None else self.scale
            return betainc(self.a, self.b, (x - loc)/scale)
        return betainc(self.a, self.b, x)

    def sf(self, x):
        if self.loc is not None or self.scale is not None:
            loc = 0 if self.loc is None else self.loc
            scale = 1 if self.scale is None else self.scale
            return betaincc(self.a, self.b, (x - loc)/scale)
        return betaincc(self.a, self.b, x)


def _get_pvalue(statistic, distribution, alternative, symmetric=True, xp=None):

    if alternative == 'less':
        pvalue = distribution.cdf(statistic)
    elif alternative == 'greater':
        pvalue = distribution.sf(statistic)
    elif alternative == 'two-sided':
        pvalue = 2 * (distribution.sf(xp.abs(statistic)) if symmetric
                      else xp.minimum(distribution.cdf(statistic),
                                      distribution.sf(statistic)))
    else:
        message = "`alternative` must be 'less', 'greater', or 'two-sided'."
        raise ValueError(message)

    return pvalue


SkewtestResult = namedtuple('SkewtestResult', ('statistic', 'pvalue'))


def _monte_carlo_test_iv(data, rvs, statistic, vectorized, n_resamples,
                         batch, alternative, axis):
    """Input validation for `monte_carlo_test`."""
    axis_int = int(axis)
    if axis != axis_int:
        raise ValueError("`axis` must be an integer.")

    if vectorized not in {True, False, None}:
        raise ValueError("`vectorized` must be `True`, `False`, or `None`.")

    if not isinstance(rvs, Sequence):
        rvs = (rvs,)
        data = (data,)
    for rvs_i in rvs:
        if not callable(rvs_i):
            raise TypeError("`rvs` must be callable or sequence of callables.")

    # At this point, `data` should be a sequence
    # If it isn't, the user passed a sequence for `rvs` but not `data`
    message = "If `rvs` is a sequence, `len(rvs)` must equal `len(data)`."
    try:
        len(data)
    except TypeError as e:
        raise ValueError(message) from e
    if not len(rvs) == len(data):
        raise ValueError(message)

    if not callable(statistic):
        raise TypeError("`statistic` must be callable.")

    if vectorized is None:
        try:
            signature = inspect.signature(statistic).parameters
        except ValueError as e:
            message = (f"Signature inspection of {statistic=} failed; "
                       "pass `vectorize` explicitly.")
            raise ValueError(message) from e
        vectorized = 'axis' in signature

    if not vectorized:
        statistic_vectorized = _vectorize_statistic(statistic)
    else:
        statistic_vectorized = statistic

    data = _broadcast_arrays(data, axis, xp=xp)
    data_iv = []
    for sample in data:
        sample = xp.broadcast_to(sample, (1,)) if sample.ndim == 0 else sample
        sample = xp_moveaxis_to_end(sample, axis_int, xp=xp)
        data_iv.append(sample)

    n_resamples_int = int(n_resamples)
    if n_resamples != n_resamples_int or n_resamples_int <= 0:
        raise ValueError("`n_resamples` must be a positive integer.")

    if batch is None:
        batch_iv = batch
    else:
        batch_iv = int(batch)
        if batch != batch_iv or batch_iv <= 0:
            raise ValueError("`batch` must be a positive integer or None.")

    alternatives = {'two-sided', 'greater', 'less'}
    alternative = alternative.lower()
    if alternative not in alternatives:
        raise ValueError(f"`alternative` must be in {alternatives}")

    # Infer the desired p-value dtype based on the input types
    min_float = getattr(xp, 'float16', xp.float32)
    dtype = xp.result_type(*data_iv, min_float)

    return (data_iv, rvs, statistic_vectorized, vectorized, n_resamples_int,
            batch_iv, alternative, axis_int, dtype, xp)


@dataclass
class MonteCarloTestResult:
    statistic: float | np.ndarray
    pvalue: float | np.ndarray
    null_distribution: np.ndarray

def monte_carlo_test(data, rvs, statistic, *, vectorized=None,
                     n_resamples=9999, batch=None, alternative="two-sided",
                     axis=0):
    args = _monte_carlo_test_iv(data, rvs, statistic, vectorized,
                                n_resamples, batch, alternative, axis)
    (data, rvs, statistic, vectorized, n_resamples,
     batch, alternative, axis, dtype, xp) = args

    # Some statistics return plain floats; ensure they're at least a NumPy float
    observed = xp.asarray(statistic(*data, axis=-1))
    observed = observed[()] if observed.ndim == 0 else observed

    n_observations = [sample.shape[-1] for sample in data]
    batch_nominal = batch or n_resamples
    null_distribution = []
    for k in range(0, n_resamples, batch_nominal):
        batch_actual = min(batch_nominal, n_resamples - k)
        resamples = [rvs_i(size=(batch_actual, n_observations_i))
                     for rvs_i, n_observations_i in zip(rvs, n_observations)]
        null_distribution.append(statistic(*resamples, axis=-1))
    null_distribution = xp.concat(null_distribution)
    null_distribution = xp.reshape(null_distribution, [-1] + [1]*observed.ndim)

    # relative tolerance for detecting numerically distinct but
    # theoretically equal values in the null distribution
    eps =  (0 if not xp.isdtype(observed.dtype, ('real floating'))
            else xp.finfo(observed.dtype).eps*100)
    gamma = xp.abs(eps * observed)

    def less(null_distribution, observed):
        cmps = null_distribution <= observed + gamma
        cmps = xp.asarray(cmps, dtype=dtype)
        pvalues = (xp.sum(cmps, axis=0, dtype=dtype) + 1.) / (n_resamples + 1.)
        return pvalues

    def greater(null_distribution, observed):
        cmps = null_distribution >= observed - gamma
        cmps = xp.asarray(cmps, dtype=dtype)
        pvalues = (xp.sum(cmps, axis=0, dtype=dtype) + 1.) / (n_resamples + 1.)
        return pvalues

    def two_sided(null_distribution, observed):
        pvalues_less = less(null_distribution, observed)
        pvalues_greater = greater(null_distribution, observed)
        pvalues = xp.minimum(pvalues_less, pvalues_greater) * 2
        return pvalues

    compare = {"less": less,
               "greater": greater,
               "two-sided": two_sided}

    pvalues = compare[alternative](null_distribution, observed)
    pvalues = xp.clip(pvalues, 0., 1.)

    return MonteCarloTestResult(observed, pvalues, null_distribution)


def pearsonr(x, y, *, alternative='two-sided', method=None, axis=0):
    x = xp.asarray(x)
    y = xp.asarray(y)

    if axis is None:
        x = xp.reshape(x, (-1,))
        y = xp.reshape(y, (-1,))
        axis = -1

    axis_int = int(axis)
    if axis_int != axis:
        raise ValueError('`axis` must be an integer.')
    axis = axis_int

    n = x.shape[axis]
    if n != y.shape[axis]:
        raise ValueError('`x` and `y` must have the same length along `axis`.')

    if n < 2:
        raise ValueError('`x` and `y` must have length at least 2.')

    try:
        x, y = xp.broadcast_arrays(x, y)
    except (ValueError, RuntimeError) as e:
        message = '`x` and `y` must be broadcastable.'
        raise ValueError(message) from e

    # `moveaxis` only recently added to array API, so it's not yey available in
    # array_api_strict. Replace with e.g. `xp.moveaxis(x, axis, -1)` when available.
    x = xp_moveaxis_to_end(x, axis, xp=xp)
    y = xp_moveaxis_to_end(y, axis, xp=xp)
    axis = -1

    dtype = xp.result_type(x.dtype, y.dtype)
    if xp.isdtype(dtype, "integral"):
        dtype = xp.asarray(1.).dtype

    if xp.isdtype(dtype, "complex floating"):
        raise ValueError('This function does not support complex data')

    x = xp.astype(x, dtype, copy=False)
    y = xp.astype(y, dtype, copy=False)
    threshold = xp.finfo(dtype).eps ** 0.75

    # If an input is constant, the correlation coefficient is not defined.
    const_x = xp.all(x == x[..., 0:1], axis=-1)
    const_y = xp.all(y == y[..., 0:1], axis=-1)
    const_xy = const_x | const_y

    if isinstance(method, PermutationMethod):
        def statistic(y, axis):
            statistic, _ = pearsonr(x, y, axis=axis, alternative=alternative)
            return statistic

        res = permutation_test((y,), statistic, permutation_type='pairings',
                               axis=axis, alternative=alternative, **method._asdict())

        return PearsonRResult(statistic=res.statistic, pvalue=res.pvalue, n=n,
                              alternative=alternative, x=x, y=y, axis=axis)
    elif isinstance(method, MonteCarloMethod):
        def statistic(x, y, axis):
            statistic, _ = pearsonr(x, y, axis=axis, alternative=alternative)
            return statistic

        if method.rvs is None:
            rng = np.random.default_rng()
            method.rvs = rng.normal, rng.normal

        res = monte_carlo_test((x, y,), statistic=statistic, axis=axis,
                               alternative=alternative, **method._asdict())

        return PearsonRResult(statistic=res.statistic, pvalue=res.pvalue, n=n,
                              alternative=alternative, x=x, y=y, axis=axis)
    elif method == 'invalid':
        message = '`method` must be `None` if arguments are not NumPy arrays.'
        raise ValueError(message)
    elif method is not None:
        message = ('`method` must be an instance of `PermutationMethod`,'
                   '`MonteCarloMethod`, or None.')
        raise ValueError(message)

    xmean = xp.mean(x, axis=axis, keepdims=True)
    ymean = xp.mean(y, axis=axis, keepdims=True)
    xm = x - xmean
    ym = y - ymean

    # scipy.linalg.norm(xm) avoids premature overflow when xm is e.g.
    # [-5e210, 5e210, 3e200, -3e200]
    # but not when `axis` is provided, so scale manually. scipy.linalg.norm
    # also raises an error with NaN input rather than returning NaN, so
    # use np.linalg.norm.
    xmax = xp.max(xp.abs(xm), axis=axis, keepdims=True)
    ymax = xp.max(xp.abs(ym), axis=axis, keepdims=True)
    with np.errstate(invalid='ignore'):
        normxm = xmax * xp_vector_norm(xm/xmax, axis=axis, keepdims=True)
        normym = ymax * xp_vector_norm(ym/ymax, axis=axis, keepdims=True)

    nconst_x = xp.any(normxm < threshold*xp.abs(xmean), axis=axis)
    nconst_y = xp.any(normym < threshold*xp.abs(ymean), axis=axis)
    nconst_xy = nconst_x | nconst_y

    with np.errstate(invalid='ignore', divide='ignore'):
        r = xp.sum(xm/normxm * ym/normym, axis=axis)

    # Presumably, if abs(r) > 1, then it is only some small artifact of
    # floating point arithmetic.
    one = xp.asarray(1, dtype=dtype)
    r = xp.asarray(xp.clip(r, -one, one))
    r[const_xy] = xp.nan

    # Make sure we return exact 1.0 or -1.0 values for n == 2 case as promised
    # in the docs.
    if n == 2:
        r = xp.round(r)
        one = xp.asarray(1, dtype=dtype)
        pvalue = xp.where(xp.asarray(xp.isnan(r)), xp.nan*one, one)
    else:
        # As explained in the docstring, the distribution of `r` under the null
        # hypothesis is the beta distribution on (-1, 1) with a = b = n/2 - 1.
        ab = xp.asarray(n/2 - 1)
        dist = _SimpleBeta(ab, ab, loc=-1, scale=2)
        pvalue = _get_pvalue(r, dist, alternative, xp=xp)

    r = r[()] if r.ndim == 0 else r
    pvalue = pvalue[()] if pvalue.ndim == 0 else pvalue
    return PearsonRResult(statistic=r, pvalue=pvalue, n=n,
                          alternative=alternative, x=x, y=y, axis=axis)


def _f_oneway_is_too_small(samples, kwargs=None, axis=-1):
    message = f"At least two samples are required; got {len(samples)}."
    if len(samples) < 2:
        raise TypeError(message)

    # Check this after forming alldata, so shape errors are detected
    # and reported before checking for 0 length inputs.
    if any(sample.shape[axis] == 0 for sample in samples):
        return True

    # Must have at least one group with length greater than 1.
    if all(sample.shape[axis] == 1 for sample in samples):
        return True

    return False


def normalize_axis_index(axis, ndim):
    if axis < -ndim or axis >= ndim:
        msg = f"axis {axis} is out of bounds for array of dimension {ndim}"
        raise AxisError(msg)

    if axis < 0:
        axis = axis + ndim
    return axis

def _get_nan(*data, xp=None):
    # Get NaN of appropriate dtype for data
    data = [xp.asarray(item) for item in data]
    try:
        min_float = getattr(xp, 'float16', xp.float32)
        dtype = xp.result_type(*data, min_float)  # must be at least a float
    except DTypePromotionError:
        # fallback to float64
        dtype = xp.float64
    return xp.asarray(xp.nan, dtype=dtype)[()]

F_onewayResult = namedtuple('F_onewayResult', ('statistic', 'pvalue'))

def _create_f_oneway_nan_result(shape, axis, samples):
    axis = normalize_axis_index(axis, len(shape))
    shp = shape[:axis] + shape[axis+1:]
    f = np.full(shp, fill_value=_get_nan(*samples))
    prob = f.copy()
    return F_onewayResult(f[()], prob[()])

def _first(arr, axis):
    return np.take_along_axis(arr, np.array(0, ndmin=arr.ndim), axis)

def _chk_asarray(a, axis, *, xp=xp):

    if axis is None:
        a = xp.reshape(a, (-1,))
        outaxis = 0
    else:
        a = xp.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = xp.reshape(a, (-1,))

    return a, outaxis

def _square_of_sums(a, axis=0):
    a, axis = _chk_asarray(a, axis)
    s = np.sum(a, axis)
    if not np.isscalar(s):
        return s.astype(float) * s
    else:
        return float(s) * s

def _sum_of_squares(a, axis=0):
    a, axis = _chk_asarray(a, axis)
    return np.sum(a*a, axis)

def f_oneway(*samples, axis=0):
    if len(samples) < 2:
        raise TypeError('at least two inputs are required;'
                        f' got {len(samples)}.')

    # ANOVA on N groups, each in its own array
    num_groups = len(samples)

    alldata = np.concatenate(samples, axis=axis)
    bign = alldata.shape[axis]

    # Check if the inputs are too small
    if _f_oneway_is_too_small(samples):
        return _create_f_oneway_nan_result(alldata.shape, axis, samples)

    is_const = np.concatenate(
        [(_first(sample, axis) == sample).all(axis=axis,
                                              keepdims=True)
         for sample in samples],
        axis=axis
    )
    all_const = is_const.all(axis=axis)

    all_same_const = (_first(alldata, axis) == alldata).all(axis=axis)

    offset = alldata.mean(axis=axis, keepdims=True)
    alldata = alldata - offset

    normalized_ss = _square_of_sums(alldata, axis=axis) / bign

    sstot = _sum_of_squares(alldata, axis=axis) - normalized_ss

    ssbn = 0
    for sample in samples:
        smo_ss = _square_of_sums(sample - offset, axis=axis)
        ssbn = ssbn + smo_ss / sample.shape[axis]

    # Naming: variables ending in bn/b are for "between treatments", wn/w are
    # for "within treatments"
    ssbn = ssbn - normalized_ss
    sswn = sstot - ssbn
    dfbn = num_groups - 1
    dfwn = bign - num_groups
    msb = ssbn / dfbn
    msw = sswn / dfwn
    with np.errstate(divide='ignore', invalid='ignore'):
        f = msb / msw

    prob = fdtrc(dfbn, dfwn, f)   # equivalent to stats.f.sf

    # Fix any f values that should be inf or nan because the corresponding
    # inputs were constant.
    if np.isscalar(f):
        if all_same_const:
            f = np.nan
            prob = np.nan
        elif all_const:
            f = np.inf
            prob = 0.0
    else:
        f[all_const] = np.inf
        prob[all_const] = 0.0
        f[all_same_const] = np.nan
        prob[all_same_const] = np.nan

    return F_onewayResult(f, prob)


KstestResult = _make_tuple_bunch('KstestResult', ['statistic', 'pvalue'],
                                 ['statistic_location', 'statistic_sign'])

def _rename_parameter(old_name, new_name, dep_version=None):
    def decorator(fun):
        @functools.wraps(fun)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                if new_name in kwargs:
                    message = (f"{fun.__name__}() got multiple values for "
                               f"argument now known as `{new_name}`")
                    raise TypeError(message)
                kwargs[new_name] = kwargs.pop(old_name)
            return fun(*args, **kwargs)
        return wrapper
    return decorator


def _parse_kstest_args(data1, data2, args, N):
    rvsfunc, cdf = None, None
    if isinstance(data1, str):
        rvsfunc = getattr(distributions, data1).rvs
    elif callable(data1):
        rvsfunc = data1

    if isinstance(data2, str):
        cdf = getattr(distributions, data2).cdf
        data2 = None
    elif callable(data2):
        cdf = data2
        data2 = None

    data1 = np.sort(rvsfunc(*args, size=N) if rvsfunc else data1)
    return data1, data2, cdf

def _compute_dplus(cdfvals, x):
    n = len(cdfvals)
    dplus = (np.arange(1.0, n + 1) / n - cdfvals)
    amax = dplus.argmax()
    loc_max = x[amax]
    return (dplus[amax], loc_max)


def _compute_dminus(cdfvals, x):
    n = len(cdfvals)
    dminus = (cdfvals - np.arange(0.0, n)/n)
    amax = dminus.argmax()
    loc_max = x[amax]
    return (dminus[amax], loc_max)

@_rename_parameter("mode", "method")
def ks_1samp(x, cdf, args=(), alternative='two-sided', method='auto'):
    mode = method

    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'greater', 'less']:
        raise ValueError(f"Unexpected value {alternative=}")

    N = len(x)
    x = np.sort(x)
    cdfvals = cdf(x, *args)
    np_one = np.int8(1)

    if alternative == 'greater':
        Dplus, d_location = _compute_dplus(cdfvals, x)
        return KstestResult(Dplus, distributions.ksone.sf(Dplus, N),
                            statistic_location=d_location,
                            statistic_sign=np_one)

    if alternative == 'less':
        Dminus, d_location = _compute_dminus(cdfvals, x)
        return KstestResult(Dminus, distributions.ksone.sf(Dminus, N),
                            statistic_location=d_location,
                            statistic_sign=-np_one)

    # alternative == 'two-sided':
    Dplus, dplus_location = _compute_dplus(cdfvals, x)
    Dminus, dminus_location = _compute_dminus(cdfvals, x)
    if Dplus > Dminus:
        D = Dplus
        d_location = dplus_location
        d_sign = np_one
    else:
        D = Dminus
        d_location = dminus_location
        d_sign = -np_one

    if mode == 'auto':  # Always select exact
        mode = 'exact'
    if mode == 'exact':
        prob = distributions.kstwo.sf(D, N)
    elif mode == 'asymp':
        prob = distributions.kstwobign.sf(D * np.sqrt(N))
    else:
        # mode == 'approx'
        prob = 2 * distributions.ksone.sf(D, N)
    prob = np.clip(prob, 0, 1)
    return KstestResult(D, prob,
                        statistic_location=d_location,
                        statistic_sign=d_sign)


def _compute_prob_outside_square(n, h):
    P = 0.0
    k = int(np.floor(n / h))
    while k >= 0:
        p1 = 1.0
        # Each of the Ai terms has numerator and denominator with
        # h simple terms.
        for j in range(h):
            p1 = (n - k * h - j) * p1 / (n + k * h + j + 1)
        P = p1 * (1.0 - P)
        k -= 1
    return 2 * P


def _count_paths_outside_method(m, n, g, h):

    # Probability is symmetrical in m, n.  Computation below assumes m >= n.
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    # Not every x needs to be considered.
    # xj holds the list of x values to be checked.
    # Wherever n*x/m + ng*h crosses an integer
    lxj = n + (mg-h)//mg
    xj = [(h + mg * j + ng-1)//ng for j in range(lxj)]
    # B is an array just holding a few values of B(x,y), the ones needed.
    # B[j] == B(x_j, j)
    if lxj == 0:
        return binom(m + n, n)
    B = np.zeros(lxj)
    B[0] = 1
    # Compute the B(x, y) terms
    for j in range(1, lxj):
        Bj = binom(xj[j] + j, j)
        for i in range(j):
            bin = binom(xj[j] - xj[i] + j - i, j-i)
            Bj -= bin * B[i]
        B[j] = Bj
    # Compute the number of path extensions...
    num_paths = 0
    for j in range(lxj):
        bin = binom((m-xj[j]) + (n - j), n-j)
        term = B[j] * bin
        num_paths += term
    return num_paths


#pythran export _compute_outer_prob_inside_method(int64, int64, int64, int64)
def _compute_outer_prob_inside_method(m, n, g, h):
    if m < n:
        m, n = n, m
    mg = m // g
    ng = n // g

    minj, maxj = 0, min(int(np.ceil(h / mg)), n + 1)
    curlen = maxj - minj
    # Make a vector long enough to hold maximum window needed.
    lenA = min(2 * maxj + 2, n + 1)
    dtype = np.float64
    A = np.ones(lenA, dtype=dtype)
    # Initialize the first column
    A[minj:maxj] = 0.0
    for i in range(1, m + 1):
        # Generate the next column.
        # First calculate the sliding window
        lastminj, lastlen = minj, curlen
        minj = max(int(np.floor((ng * i - h) / mg)) + 1, 0)
        minj = min(minj, n)
        maxj = min(int(np.ceil((ng * i + h) / mg)), n + 1)
        if maxj <= minj:
            return 1.0
        # Now fill in the values. We cannot use cumsum, unfortunately.
        val = 0.0 if minj == 0 else 1.0
        for jj in range(maxj - minj):
            j = jj + minj
            val = (A[jj + minj - lastminj] * i + val * j) / (i + j)
            A[jj] = val
        curlen = maxj - minj
        if lastlen > curlen:
            # Set some carried-over elements to 1
            A[maxj - minj:maxj - minj + (lastlen - curlen)] = 1

    return A[maxj - minj - 1]

def _attempt_exact_2kssamp(n1, n2, g, d, alternative):
    lcm = (n1 // g) * n2
    h = int(np.round(d * lcm))
    d = h * 1.0 / lcm
    if h == 0:
        return True, d, 1.0
    saw_fp_error, prob = False, np.nan
    try:
        with np.errstate(invalid="raise", over="raise"):
            if alternative == 'two-sided':
                if n1 == n2:
                    prob = _compute_prob_outside_square(n1, h)
                else:
                    prob = _compute_outer_prob_inside_method(n1, n2, g, h)
            else:
                if n1 == n2:
                    jrange = np.arange(h)
                    prob = np.prod((n1 - jrange) / (n1 + jrange + 1.0))
                else:
                    with np.errstate(over='raise'):
                        num_paths = _count_paths_outside_method(n1, n2, g, h)
                    bin = binom(n1 + n2, n1)
                    if num_paths > bin or np.isinf(bin):
                        saw_fp_error = True
                    else:
                        prob = num_paths / bin

    except (FloatingPointError, OverflowError):
        saw_fp_error = True

    if saw_fp_error:
        return False, d, np.nan
    if not (0 <= prob <= 1):
        return False, d, prob
    return True, d, prob

@_rename_parameter("mode", "method")
def ks_2samp(data1, data2, alternative='two-sided', method='auto'):
    mode = method

    if mode not in ['auto', 'exact', 'asymp']:
        raise ValueError(f'Invalid value for mode: {mode}')
    alternative = {'t': 'two-sided', 'g': 'greater', 'l': 'less'}.get(
        alternative.lower()[0], alternative)
    if alternative not in ['two-sided', 'less', 'greater']:
        raise ValueError(f'Invalid value for alternative: {alternative}')
    MAX_AUTO_N = 10000  # 'auto' will attempt to be exact if n1,n2 <= MAX_AUTO_N
    if np.ma.is_masked(data1):
        data1 = data1.compressed()
    if np.ma.is_masked(data2):
        data2 = data2.compressed()
    data1 = np.sort(data1)
    data2 = np.sort(data2)
    n1 = data1.shape[0]
    n2 = data2.shape[0]
    if min(n1, n2) == 0:
        raise ValueError('Data passed to ks_2samp must not be empty')

    data_all = np.concatenate([data1, data2])
    # using searchsorted solves equal data problem
    cdf1 = np.searchsorted(data1, data_all, side='right') / n1
    cdf2 = np.searchsorted(data2, data_all, side='right') / n2
    cddiffs = cdf1 - cdf2

    # Identify the location of the statistic
    argminS = np.argmin(cddiffs)
    argmaxS = np.argmax(cddiffs)
    loc_minS = data_all[argminS]
    loc_maxS = data_all[argmaxS]

    # Ensure sign of minS is not negative.
    minS = np.clip(-cddiffs[argminS], 0, 1)
    maxS = cddiffs[argmaxS]

    if alternative == 'less' or (alternative == 'two-sided' and minS > maxS):
        d = minS
        d_location = loc_minS
        d_sign = -1
    else:
        d = maxS
        d_location = loc_maxS
        d_sign = 1
    g = gcd(n1, n2)
    n1g = n1 // g
    n2g = n2 // g
    prob = -np.inf
    if mode == 'auto':
        mode = 'exact' if max(n1, n2) <= MAX_AUTO_N else 'asymp'
    elif mode == 'exact':
        # If lcm(n1, n2) is too big, switch from exact to asymp
        if n1g >= np.iinfo(np.int32).max / n2g:
            mode = 'asymp'

    if mode == 'exact':
        success, d, prob = _attempt_exact_2kssamp(n1, n2, g, d, alternative)
        if not success:
            mode = 'asymp'

    if mode == 'asymp':
        m, n = sorted([float(n1), float(n2)], reverse=True)
        en = m * n / (m + n)
        if alternative == 'two-sided':
            prob = distributions.kstwo.sf(d, np.round(en))
        else:
            z = np.sqrt(en) * d
            # Use Hodges' suggested approximation Eqn 5.3
            # Requires m to be the larger of (n1, n2)
            expt = -2 * z**2 - 2 * z * (m + 2*n)/np.sqrt(m*n*(m+n))/3.0
            prob = np.exp(expt)

    prob = np.clip(prob, 0, 1)
    return KstestResult(np.float64(d), prob, statistic_location=d_location,
                        statistic_sign=np.int8(d_sign))

@_rename_parameter("mode", "method")
def kstest(rvs, cdf, args=(), N=20, alternative='two-sided', method='auto'):
    # to not break compatibility with existing code
    if alternative == 'two_sided':
        alternative = 'two-sided'
    if alternative not in ['two-sided', 'greater', 'less']:
        raise ValueError(f"Unexpected alternative: {alternative}")
    xvals, yvals, cdf = _parse_kstest_args(rvs, cdf, args, N)
    if cdf:
        return ks_1samp(xvals, cdf, args=args, alternative=alternative,
                        method=method, _no_deco=True)
    return ks_2samp(xvals, yvals, alternative=alternative, method=method,
                    _no_deco=True)
