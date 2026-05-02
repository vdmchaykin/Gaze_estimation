import numpy as np
from scipy.ndimage import gaussian_filter

def extract_single_peak(heatmap, sigma=1.0):
    """
    heatmap: (H, W) numpy array
    returns:
        xy: (x, y) coordinates
        confidence: peak value after smoothing
    """
    # suppress small noisy peaks
    hm = gaussian_filter(heatmap, sigma=sigma)

    # global maximum
    y, x = np.unravel_index(np.argmax(hm), hm.shape)
    confidence = hm[y, x]

    return np.array([x, y], dtype=np.float32), confidence
