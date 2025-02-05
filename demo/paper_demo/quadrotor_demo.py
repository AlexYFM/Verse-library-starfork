from dryvr_agent import QuadrotorAgent 
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import * 

import plotly.graph_objects as go 
from enum import Enum, auto 
import time
import plotly.graph_objects as go
from verse.plotter.plotterStar import *
from verse.utils.star_diams import *
class AgentMode(Enum):
    Default = auto()

if __name__ == "__main__":
    scenario = Scenario(ScenarioConfig(parallel=False))

    quad = QuadrotorAgent('quad')
    scenario.add_agent(quad)
    # The initial position of the quadrotor is uncertain in 
    # all directions within [−0.4, 0.4] [m] and also the velocity 
    # is uncertain within [−0.4, 0.4] [m/s] for all directions
    
    # The inertial (north) position x1, the inertial (east) position x2, 
    # the altitude x3, the longitudinal velocity x4, 
    # the lateral velocity x5, the vertical velocity x6, 
    # the roll angle x7, the pitch angle x8, 
    # the yaw angle x9, the roll rate x10, 
    # the pitch rate x11, and the yaw rate x12.
    scenario.set_init(
        [
            [[-0.4,-0.4,-0.4,-0.4,-0.4,-0.4, 0, 0, 0, 0, 0, 0, 1.00],
             [ 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0, 0, 0, 0, 0, 0, 1.00]]
# [[-0.04,-0.04,-0.04,-0.04,-0.04,-0.04, 0, 0, 0, 0, 0, 0, 1.00],
# [ 0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0, 0, 0, 0, 0, 0, 1.00]]
        ],
        [
            (AgentMode.Default, )
        ]
    )
    start = time.time()
    traces = scenario.verify(5, .1)
    # sim_traces = scenario.simulate_multi(5, 0.1)
    end = time.time()
    print(f'Time: {end-start}')
    diams = time_step_diameter_rect(traces, 5, .1)
    print(f'Initial diameter: {diams[0]}\n Final: {diams[-1]}\n Average: {sum(diams)/len(diams)}')
    fig = go.Figure() 
    fig = reachtube_tree(traces, None, fig, 0, 3, [1,3], "lines", "trace")
    # for sim in sim_traces:
    #     simulation_tree(sim, None, fig, 0, 2, [1,3])
    fig.show()