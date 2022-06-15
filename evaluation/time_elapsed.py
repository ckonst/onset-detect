# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:13:15 2022.

@author: Christian Konstantinov
"""

from functools import wraps
from time import time

def timed(f, units='ms'):
    """Print run time."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        scalar = 1000
        if units == 's':
            scalar = 1
        elif units == 'min':
            scalar = 1/60
        elif units == 'h':
            scalar = (1/60)/60
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'Elapsed time: {(end-start) * scalar} {units}')
        return result
    return wrapper
