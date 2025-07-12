import logging
from functools import wraps
from time import time

log = logging.getLogger(__name__)


def timed(units):
    """Print run time."""

    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            scalar = 1000
            if units == 's':
                scalar = 1
            elif units == 'min':
                scalar = 1 / 60
            elif units == 'h':
                scalar = (1 / 60) / 60
            start = time()
            result = f(*args, **kwargs)
            end = time()
            log.info(f'Elapsed time: {(end - start) * scalar} {units}')
            return result

        return wrapper

    return decorator
