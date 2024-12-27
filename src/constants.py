############################# PACKAGES
import pandas as pd
import os
import re
import sys
import argparse
import subprocess
import math
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plot
import matplotlib.dates as mdates
import scipy.interpolate as interp
import numpy as np
import statsmodels.api as sm
import statsmodels.graphics.tsaplots as tsaplots
import statsmodels.tsa.stattools as tsatools
import statsmodels.stats.outliers_influence as outliers_influence
import statsmodels.tsa.arima.model as arima
import pmdarima as pm
import scipy.stats
import statistics as stats
import seaborn as sn
import sklearn as sk
import sklearn.linear_model as sk_lm
import sklearn.metrics as metrics
import datetime
import json
sys.path.append(r"PathToTheCoolPropFolder")
import CoolProp.CoolProp as CP
import shutil

############################# DIRECTORIES
# MAIN DIRECTORIES
PATH_MAIN   = '/app/'
PATH_DATA   = PATH_MAIN + 'data/'
PATH_OUT    = PATH_MAIN + 'output/'

# INPUT DIRECTORIES
PATH_METADATA                    = PATH_DATA + 'metadata/'

PATH_SENSORS_DATA_RAW_UF         = PATH_DATA + 'from_sensors/0_raw/UF/'
PATH_SENSORS_DATA_EXT_UF_V1      = PATH_DATA + 'from_sensors/1_extended/UF/v1/'
PATH_SENSORS_DATA_EXT_UF_V2      = PATH_DATA + 'from_sensors/1_extended/UF/v2/'
PATH_SENSORS_DATA_EXT_UF_V3      = PATH_DATA + 'from_sensors/1_extended/UF/v3/'

PATH_UPPAAL_DATA_RAW             = PATH_DATA + 'from_uppaal/0_raw/'
PATH_UPPAAL_DATA_RAW_REPAIRED    = PATH_DATA + 'from_uppaal/1_repaired/'
PATH_UPPAAL_DATA_EXT             = PATH_DATA + 'from_uppaal/2_extended/'

# OUTPUT DIRECTORIES
PATH_IMAGES                      = PATH_OUT + 'images/'
PATH_NOTEBOOK_OUTPUT             = PATH_OUT + 'notebooks_html/'
PATH_ESTIMATED_COEFFICIENTS      = PATH_OUT + 'estimated_coefficients/'

############################# FILES
# PARAMETERS FILE
FILE_PARAMETERS                  = PATH_MAIN + 'parameters.json'
FILE_DATA_SIMULATIONS_ASSOC      = PATH_MAIN + 'real_data__uppaal_simulations__association.json'

# INPUT FILES
FILE_EXPERIMENTS_METADATA         = PATH_METADATA + 'experiments.xlsx'
FILE_MEMBRANES_METADATA           = PATH_METADATA + 'membranes.xlsx'

# OUTPUT FILES (stored in PATH_OUT directory)
FILE_EST_COEFFS                  = PATH_ESTIMATED_COEFFICIENTS + 'estimated_coefficients.json'

# OUTPUT DATA (stored in PATH_DATA directory)
FILE_SENSORS_DATA_EST_PARAMS     = PATH_SENSORS_DATA_EXT_UF_V3 + '/estimated_parameters.csv'

ALL_PATHS =  [
    # MAIN DIRECTORIES
    PATH_DATA,
    PATH_OUT,
    # INPUT DIRECTORIES
    PATH_METADATA,
    PATH_SENSORS_DATA_RAW_UF,
    PATH_SENSORS_DATA_EXT_UF_V1,
    PATH_SENSORS_DATA_EXT_UF_V2,
    PATH_SENSORS_DATA_EXT_UF_V3,
    PATH_UPPAAL_DATA_RAW,
    PATH_UPPAAL_DATA_RAW_REPAIRED,
    PATH_UPPAAL_DATA_EXT,
    # OUTPUT DIRECTORIES
    PATH_IMAGES,
    PATH_NOTEBOOK_OUTPUT,
    PATH_ESTIMATED_COEFFICIENTS
]

# NEVER DELETE THESE!!!
RAW_DATA_PATHS = [
    PATH_DATA,
    PATH_METADATA,
    PATH_SENSORS_DATA_RAW_UF,
    PATH_UPPAAL_DATA_RAW,
]

# CAN BE DELETED WITHOUT ANY DAMAGE (BUT OUTPUT IS LOST)
NON_RAW_PATHS = list(set(ALL_PATHS) - set(RAW_DATA_PATHS))

############################# CONSTANTS

x_axis   = ('time [m]', 'datetime', 'index')
x_format = (None,       '%H:%M',    None)
TIME_MINS = 0
DATE_TIME = 1
ROW_IDX_AS_TIME = 2

# by default: only one global
DEFAULT_CONC_GROUP = {
    0 : ((None, None), 'unknown')   
}

# key file names are the one in folder PATH_SENSORS_DATA_RAW_UF
CONCENTRATION_INTERVALS = {
    "2023-11-08 clean water.csv" : {
        0 : ((None, None), 'clean water')
    },
    "2023-11-09 clean + dirty water.csv" : {
        0 : ((  6,   20), 'clean water'),
        1 : (( 20,   79), 'dirty water'),
        2 : (( 79,  106), 'dirty water'),
        3 : ((154, None), 'dirty water')
    }
}

# key file names are the one in folder PATH_SENSORS_DATA_EXT_UF_V1
TMP_INTERVALS = {
    "2023-11-08 0 clean water.csv" : {
        0 : (None,  68),
        1 : ( 68,  81),
        2 : ( 81, 120),
        3 : (120, 153),
        4 : (153, 181),
        5 : (181, 212),
        6 : (212, None),
    },
    "2023-11-09 0 clean water.csv" : {
        0 : (None, None) #(2, 21)
    },
    "2023-11-09 1 dirty water.csv" : {
        0 : ( 1,  7),
        #1 : (13,  17), # too few data
        2 : (17, 22),
        3 : (22, 28),
        4 : (28, 36),
        5 : (36, 44),
        6 : (44, None),
    },
    "2023-11-09 2 dirty water.csv" : {
        0 : ( 0,  9),
        #4 : (17, 21), # too few data
        5 : (22, None),
    },
    "2023-11-09 3 dirty water.csv" : {
        #0 : ( 1,  5), # too few data
        #1 : ( 5, 9), # too few data
        #2 : ( 9, 12), # too few data
        #3 : (12, 15), # too few data
        4 : (15, 27),
        5 : (27, None),
    }
}

DEFAULT_PARAMETERS = {
    "file_idx_uppaal" : 0,
    "file_idx" : 0,
    "tmp_idx" : 0,
    "log" : True,
    "reset_columns_when_OFF" : True,
    "drop_outliers" : True,
    "plot_scatterplot_matrix" : False,
    "use_default_arima_params": True,
    "default_arima_params": (1,1,0),
    "include_arima_simulations_in_analysis": True,
}

ALL_N       = [0,      1,    1.5, 2 ]
#ALL_MAX_K_N = [0.0002, 0.05, 0.2, 1 ] # AVOID OUTLIERS
ALL_MAX_K_N = [0.00005, 0.002, 0.2, 1 ] # AVOID OUTLIERS

# n -> fouling method
FOULING_NAME = {
    0   : "Cake filtration",
    1   : "Intermediate pore blocking",
    1.5 : "Internal (or Standard) pore blocking",
    2   : "Complete pore blocking",
}

COLOR_CYCLE = plot.rcParams['axes.prop_cycle'].by_key()['color']

UF_COLUMNS = {
    'Date'          : 'date',
    'Time'          : 'time',
    'Millisecond'   : 'millisecond [ms]',
    'LT1[%]'        : 'tank liters [%]',
    'PT1[bar]'      : 'prs feed_1 [bar]',
    'PT2[Bar]'      : 'prs feed_2 [bar]',
    'PT3[bar]'      : 'prs permeate [bar]',
    'PT4[bar]'      : 'prs retentate [bar]',
    'TMP[bar]'      : 'TMP [bar]',
    'FIT1[m³/h]'    : 'flow feed [m^3/h]',
    'FIT2[m³/h]'    : 'flow permeate [m^3/h]',
    'FIT3[m³/h]'    : 'flow retentate [m^3/h]',
    'TT1[°C]'       : 'temperature [°C]'
}

UNIT_MEASURES = {
    'bar'   : ('kPa', (lambda x :  100 * x)),
    'm^3/h' : ('L/h', (lambda x : 1000 * x))
}

# FEED_TANK_CAPACITY_LITERS = 150

PRS_ATM_kpa = 101.325

# mark point as outlier and drop row if res < MIN_RES or res > MAX_RES
MIN_RES = 1e12
MAX_RES = 1e15
