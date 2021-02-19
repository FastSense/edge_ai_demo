import functools
import time
import rospy

def profile(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = f(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        rospy.loginfo_throttle(10, 'Finished %s in %f', f.__name__ , run_time)
        return value
    return wrapper

