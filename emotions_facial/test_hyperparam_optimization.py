from hyperparam_optimization import gaussian_peak
import numpy as np


def test_gaussian_peak():
    assert (gaussian_peak(4, 15, 4) == np.array([4, 12, 15, 12])).all()
