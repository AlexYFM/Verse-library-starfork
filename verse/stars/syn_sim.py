import numpy as np
from typing_extensions import List
from scipy.integrate import ode
import torch

def dyn_3d(vec, t):
    x, y, z = t

    ### purely for testing, doesn't correspond to any model
    x_dot = y
    y_dot = (1 - x**2) * y - x
    z_dot = x-z

    return [x_dot, y_dot, z_dot]

def sim_test_3d(
    mode: List[str], initialCondition, time_bound, time_step, 
) -> np.ndarray:
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    # note: digit of time
    init = list(initialCondition)
    trace = [[0] + init]
    for i in range(len(t)):
        r = ode(dyn_3d)
        r.set_initial_value(init)
        res: np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init)
    return np.array(trace)

import torch


