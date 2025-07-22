import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class Statistic:
    @staticmethod
    def moving_average(dataseries, window_size=20):
        kernal = np.ones(window_size) / window_size
        moving_avg = np.convolve(dataseries, kernal, mode='valid')
        return moving_avg
    
    @staticmethod
    def moving_average_padded(dataseries, window_size=20):
        kernal = np.ones(window_size) / window_size
        moving_avg = np.convolve(dataseries, kernal, mode='valid')
        moving_avg_padded = np.pad(moving_avg, (window_size - 1, 0), mode='constant', constant_values=0)
        return moving_avg_padded
        
    @staticmethod
    def moving_variance(dataseries, window_size=20):
        windows = sliding_window_view(dataseries, window_shape=window_size)
        moving_var = np.var(windows, axis=1, ddof=0)
        return moving_var
    
    @staticmethod
    def moving_variance_padded(dataseries, window_size=20):
        windows = sliding_window_view(dataseries, window_shape=window_size)
        moving_var = np.var(windows, axis=1, ddof=0)
        moving_var_padded = np.pad(moving_var, (window_size - 1, 0), mode='constant', constant_values=0)
        return moving_var_padded