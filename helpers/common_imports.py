import copy, os, sys, pathlib, shutil, warnings, time, datetime, json, dataclasses, typing, itertools as it
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from codetiming import Timer as Timer # https://realpython.com/python-timer/
from pretty_html_table import build_table

ellipsis = ' ... '
CRS = {
    'census'  : 'EPSG:4269'  , # degrees - used by Census
    'bigquery': 'EPSG:4326'  , # WSG84 - used by Bigquery
    'area'    : 'ESRI:102003', # meters
    'length'  : 'ESRI:102005', # meters
}
pd.set_option('display.max_columns', None)
