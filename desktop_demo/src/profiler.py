from functools import wraps
import time
from rospy import loginfo_throttle


def profile(tag):
    def inner_f(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            start_time = time.perf_counter()
            value = f(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            loginfo_throttle(5, 'Finished %s from %s in %f', tag, f.__name__, run_time)
            return value
        return wrapper
    return inner_f
