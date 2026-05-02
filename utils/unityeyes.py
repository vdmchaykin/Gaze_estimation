import re
import numpy as np

def parse_point(s):
    """
    "(x, y, z)" -> (x, y)
    """
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", s)
    return float(nums[0]), float(nums[1])

def mean_point(points):
    """
    list of "(x, y, z)" -> (mean_x, mean_y)
    """
    pts = np.array([parse_point(p) for p in points])
    return pts.mean(axis=0)
