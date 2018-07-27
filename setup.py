### Super-Importer created by Geir Arne Hjelle.  Will attempt to pip install module if not found by normal import.
from importlib.abc import MetaPathFinder
from importlib import util
import subprocess
import sys

class PipFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        print(f"Module {self} not installed.  Attempting to pip install")
        cmd = f"{sys.executable} -m pip install {self}"
        try:
            subprocess.run(cmd.split(), check=True)
        except subprocess.CalledProcessError:
            return None
        return util.find_spec(self)

### Commonly used modules
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import itertools as it
import math



### Preferences
np.set_printoptions(precision=4, suppress=True)





### utility functions
def listify(X):
    """
    Ensure X is a list
    """
    if isinstance(X, list):
        return X
    elif (X is None) or (X is np.nan):
        return []
    elif isinstance(X,str):
        return [X]
    else:
        try:
            return list(X)
        except:
            return [X]
