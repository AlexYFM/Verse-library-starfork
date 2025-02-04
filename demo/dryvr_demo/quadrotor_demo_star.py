from dryvr_agent import QuadrotorAgent 
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import * 

import plotly.graph_objects as go 
from enum import Enum, auto 

from verse.analysis.verifier import ReachabilityMethod
from verse.stars.starset import *
from verse.sensor.base_sensor_stars import *

import time
import plotly.graph_objects as go
from verse.plotter.plotterStar import *
from verse.utils.star_diams import *
class AgentMode(Enum):
    Default = auto()

if __name__ == "__main__":
    scenario = Scenario(ScenarioConfig(parallel=False))

    quad = QuadrotorAgent('quad')
    scenario.set_sensor(BaseStarSensor())
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    # The initial position of the quadrotor is uncertain in 
    # all directions within [−0.4, 0.4] [m] and also the velocity 
    # is uncertain within [−0.4, 0.4] [m/s] for all directions
    scenario.config.model_path = 'quad_svd_bench_large'

    # The inertial (north) position x1, the inertial (east) position x2, 
    # the altitude x3, the longitudinal velocity x4, 
    # the lateral velocity x5, the vertical velocity x6, 
    # the roll angle x7, the pitch angle x8, 
    # the yaw angle x9, the roll rate x10, 
    # the pitch rate x11, and the yaw rate x12.
    inf = np.array([-0.4,-0.4,-0.4,-0.4,-0.4,-0.4, 0, 0, 0, 0, 0, 0, 1.00])
    sup = np.array([ 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 1.00])

    center = (inf+sup)/2
    basis = np.eye(13)*np.diag(center-inf)
    C, g = new_pred(13)
    quad.set_initial(
        initial_state=StarSet(center, basis, C, g),
        initial_mode=(AgentMode.Default, )
    )
    scenario.add_agent(quad)

    scenario.config.overwrite = True
    start = time.time()
    traces = scenario.verify(1,1) # could try 0.1 s instead, would also have to increase ts of DryVR too
    end = time.time()
    print(f'Time: {end-start}')
    diams = time_step_diameter(traces, 1, 1)
    print(f'Initial diameter: {diams[0]}\n Final: {diams[-1]}\n Average: {sum(diams)/len(diams)}')
    plot_stars_time(traces, 3)
    # fig = go.Figure() 
    # fig = reachtube_tree(traces, None, fig, 0, 3, [1,3], "lines", "trace")
    # fig.show()