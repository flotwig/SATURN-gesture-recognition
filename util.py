from functools import reduce
import pandas as pd
import os
import numpy

def get_available_datasets():
    data = []

    for datafile in os.popen('find ./data -type f -name *.csv').read().split('\n')[0:-1]:
        n = datafile.split('/')
        data.append({
            'Dir': n[-2],
            'File': n[-1].split('.')[0],
            'Path': datafile
        })

    return data

def load_dataset(datum):
    return pd.read_csv(datum['Path'], names=['V'], header=None)

def vector_magnitude(data):
    """ function to calculate the magnitude of a vector

    Calculate the magnitude of the vector superposition of data (for
    example, acceleration) on x, y, and z axis

    Arguments:
        data: array of (x, y, z) tuples for a vector

    Returns:
        arra of the magnitude of a vector

    """
    return map(lambda x: numpy.sqrt(float(x[0])**2 + float(x[1])**2 + float(x[2])**2), data)


def moving_average(data, window_size):
    """ moving average filter

    Implement a simple moving average filter to use as a low pass
    filter

    Arguments:
        data: data be filtered
        window_size: window_size chosen for the data

    Returns:
        The filtered data.

    TODO:
        Finish this function. Think about how you want to handle
        the size difference between your input array and output array.
        You can write it yourself or consider using numpy.convole for
        it:
        https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html

    """
    res = []
    for (i,t) in enumerate(data):
        if i == 0:
            continue
        start = i - window_size
        if start < 0:
            start = 0
        end = i
        sl = data[start:end]
        x, y, z = (0.0, 0.0, 0.0)
        for t in sl:
            x = x + float(t[0])
            y = y + float(t[1])
            z = z + float(t[2])
        x = x / len(sl)
        y = y / len(sl)
        z = z / len(sl)
        res.append((x,y,z))
    return res
