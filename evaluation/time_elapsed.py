# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 14:13:15 2022.

@author: Christian Konstantinov
"""

from functools import wraps
from time import time

def timed(f):
    """Print run time."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print('Elapsed time: {} ms'.format((end-start) * 1000))
        return result
    return wrapper
