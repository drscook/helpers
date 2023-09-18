import copy, os, sys, pathlib, contextlib, shutil, warnings, time
import datetime, joblib, json, IPython, dataclasses, typing
import itertools as it, numpy as np, pandas as pd
import matplotlib.pyplot as plt, bokeh as bk, seaborn as sns, plotly.express as px
DTYPE_BACKEND = 'numpy_nullable'
CAP = 'casefold'

def disp(df, rows=100, cols=None):
    with pd.option_context('display.max_rows', rows, 'display.min_rows', rows, 'display.max_columns', cols):
        display(df)
pd.Series.display = disp
pd.DataFrame.display = disp

def mkdir(path, overwrite=False, exist_ok=True, parents=True):
    """Make dir, overwriting existing if desired"""
    path = pathlib.Path(path)
    if overwrite:
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(exist_ok=exist_ok, parents=parents)

def listify(X):
    """Turns almost anything into a list"""
    if X is None or X is np.nan:
        return []
    elif isinstance(X, (list, tuple, set, type({}.keys()), type({}.values()), pd.Index)):
        return list(X)
    elif isinstance(X, dict):
        return [list(X.keys()), list(X.values())]
    elif isinstance(X, np.ndarray):
        return X.tolist()
    elif isinstance(X, pd.DataFrame):
        return X.values.tolist()
    elif isinstance(X, pd.Series):
        return X.tolist()
    else:
        return [X]

def setify(X):
    return set(listify(X))

def join(x, sep=', '):
    return sep.join(listify(x))

def to_numeric(ser):
    dt = str(ser.dtype).lower()
    if 'geometry' not in dt and 'bool' not in dt:
        try:
            ser = pd.to_datetime(ser)
        except DateParseError:
            ser = pd.to_numeric(ser.astype('string').str.lower().str.strip(), errors='ignore', downcast='integer')
    ser = ser.convert_dtypes(dtype_backend=DTYPE_BACKEND)
    if pd.api.types.is_integer_dtype(ser):
        ser = ser.astype('int64[pyarrow]' if DTYPE_BACKEND=='pyarrow' else 'Int64')
    return ser

def rename_column(x, cap=CAP):
    return prep(x, cap).replace(' ','_').replace('-','_') if isinstance(x, str) else x

def prep(X, cap=CAP):
    caps = ['casefold', 'lower', 'upper', 'capitalize', 'swapcase', 'title', None, False]
    assert cap in caps, f'Unknown capitalization {cap} ... must one of {caps}'
    if X is None or X is np.nan:
        return None
    elif isinstance(X, str):
        if cap:
            X = getattr(X, cap)()
        return X.strip()
    elif isinstance(X, (list, tuple, set, pd.Index)):
        return type(X)((prep(x, cap) for x in X))
    elif isinstance(X, dict):
        return dict(zip(*prep(listify(X), cap)))
    elif isinstance(X, np.ndarray):
        return np.array(prep(listify(X), cap))
    elif isinstance(X, pd.DataFrame):
        g = lambda df:df.apply(to_numeric).rename(columns=rename_column)
        idx = pd.MultiIndex.from_frame(g(X[[]].reset_index()))
        return g(X).set_index(idx).drop(columns='index', errors='ignore')
    elif isinstance(X, pd.Series):
        return prep(X.to_frame(), cap).squeeze()
    else:
        return X
pd.Series.prep = prep
pd.DataFrame.prep = prep

def rnd(df, digits=0):
    return pd.DataFrame(df).round(digits).prep().squeeze()
pd.Series.rnd = rnd
pd.DataFrame.rnd = rnd

@dataclasses.dataclass
class MyBaseClass():
    def __getitem__(self, key):
        return getattr(self, key)
    def __delitem__(self, key):
        delattr(self, key)
    def __setitem__(self, key, val):
        setattr(self, key, val)
    def __contains__(self, key):
        return hasattr(self, key)
