import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

def weighted_mean(data, value_column, weight_column):
    data = data.dropna(subset=[value_column, weight_column])
    if data.empty:
        return float('nan')
    return (data[value_column] * data[weight_column]).sum() / data[weight_column].sum()

def weighted_median(data, value_column, weight_column):
    data = data.dropna(subset=[value_column, weight_column])
    if data.empty:
        return float('nan')  
    data = data.sort_values(by=value_column)
    cumsum_weights = data[weight_column].cumsum()
    cutoff = data[weight_column].sum() / 2.0
    median_idx = cumsum_weights.searchsorted(cutoff)
    
    return data.iloc[median_idx][value_column]


