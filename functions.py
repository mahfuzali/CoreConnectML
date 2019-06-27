from pandas_datareader import data as web
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import linear_model
import datetime as dt
import math
import numpy as np 
import pandas as pd
import os

def calc_true_range_(row):

    high_ = row['High']
    low_ = row['Low']
    open_ = row['Open']
    close_ = row['Close']
    volume_ = row['Volume']
    prev_close_ = row['Prev Close']

    max_1_ = high_ - low_
    max_2_ = abs(high_ - prev_close_)
    max_3_ = abs(low_- prev_close_)

    TR_ = max(max_1_,max_2_,max_3_)

    return TR_
