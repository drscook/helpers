import copy, os, sys, pathlib, contextlib, shutil, warnings, time, datetime, json, IPython, dataclasses, typing, itertools as it
import numpy as np, pandas as pd, pretty_html_table, matplotlib.pyplot as plt
from codetiming import Timer as Timer # https://realpython.com/python-timer/

ellipsis = ' ... '
CRS = {
    'census'  : 'EPSG:4269'  , # degrees - used by Census
    'bigquery': 'EPSG:4326'  , # WSG84 - used by Bigquery
    'area'    : 'ESRI:102003', # meters
    'length'  : 'ESRI:102005', # meters
}
pd.set_option('display.max_columns', None)
