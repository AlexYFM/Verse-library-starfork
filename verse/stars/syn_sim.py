import numpy as np
from typing_extensions import List
from scipy.integrate import ode

### reinstall torch with cuda on this fork if necessary

### synthetic dynamic and simulation function
def dynamic_test(vec, t):
    x, y = t # hack to access right variable, not sure how integrate, ode are supposed to work
    ### vanderpol
    x_dot = y
    y_dot = (1 - x**2) * y - x

    ### cardiac cell
    # x_dot = -0.9*x*x-x*x*x-0.9*x-y+1
    # y_dot = x-2*y

    ### jet engine
    # x_dot = -y-1.5*x*x-0.5*x*x*x-0.5
    # y_dot = 3*x-y

    ### brusselator 
    # x_dot = 1+x**2*y-2.5*x
    # y_dot = 1.5*x-x**2*y

    ### bucking col -- change center to around -0.5 and keep basis size low
    # x_dot = y
    # y_dot = 2*x-x*x*x-0.2*y+0.1

    ###non-descript convergent system
    # x_dot = y
    # y_dot = -5*x-5*x**3-y
    return [x_dot, y_dot]

def sim_test(
    mode: List[str], initialCondition, time_bound, time_step, 
) -> np.ndarray:
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    # note: digit of time
    init = list(initialCondition)
    trace = [[0] + init]
    for i in range(len(t)):
        r = ode(dynamic_test)
        r.set_initial_value(init)
        res: np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init)
    return np.array(trace)

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



