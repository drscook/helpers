import os, sys, pathlib, shutils, warnings, time, datetime, json, dataclasses, codetiming
import itertools as it, numpy as np, pandas as pd, matplotlib.pyplot as plt

CRS = {
    'census'  : 'EPSG:4269'  , # degrees - used by Census
    'bigquery': 'EPSG:4326'  , # WSG84 - used by Bigquery
    'area'    : 'ESRI:102003', # meters
    'length'  : 'ESRI:102005', # meters
}
