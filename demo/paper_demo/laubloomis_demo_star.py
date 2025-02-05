from dryvr_agent import LaubLoomisAgent 
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import * 

import plotly.graph_objects as go 
from enum import Enum, auto 

from verse.stars.starset import *
from verse.sensor.base_sensor_stars import *
from verse.analysis.verifier import ReachabilityMethod

from verse.utils.star_diams import *

class AgentMode(Enum):
    Default = auto()

if __name__ == "__main__":
   
    # The initial position of the quadrotor is uncertain in 
    # all directions within [−0.4, 0.4] [m] and also the velocity 
    # is uncertain within [−0.4, 0.4] [m/s] for all directions
    
    # The inertial (north) position x1, the inertial (east) position x2, 
    # the altitude x3, the longitudinal velocity x4, 
    # the lateral velocity x5, the vertical velocity x6, 
    # the roll angle x7, the pitch angle x8, 
    # the yaw angle x9, the roll rate x10, 
    # the pitch rate x11, and the yaw rate x12.
    for i in range(3):
        print(f"{i} start")
        scenario = Scenario(ScenarioConfig(parallel=False))
        W = 0.01
        agent = LaubLoomisAgent('laub')
        # scenario.config.model_path = 'laubloomis'
        if i==1:
            continue 
        if i == 0:
            scenario.config.reachability_method = ReachabilityMethod.DRYVR
        else:
            scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
            scenario.set_sensor(BaseStarSensor())

        if i == 1:
            scenario.config.nn_enable = False
        elif i == 2:
            scenario.config.nn_enable = True

        #print(scenario.config.reachability_method)
        
        # scenario.config.pca = False
        basis = np.eye(7)*np.diag([W for _ in range(7)])
        center = np.array([1.2,1.05,1.5,2.4,1,0.1,0.45])
        C, g = new_pred(7)
        # C = np.transpose(np.array([[1,-1,0,0,0,0],[0,0,1,-1,0,0], [0,0,0,0,1,-1]]))
        # g = np.array([1,1,1,1,1,1])

        star = StarSet(center, basis, C, g)

        if i == 0:
            initial_set = star.overapprox_rectangle()
        else:
            initial_set = star


        agent.set_initial(
            initial_set
            ,
            (AgentMode.Default, )
        )
        scenario.add_agent(agent)

        import time

        scenario.config.overwrite = True
        start = time.time()
        traces = scenario.verify(10, 0.04)
        end = time.time()
        tot_time = end - start
        print({"time":tot_time})

        import plotly.graph_objects as go
        from verse.plotter.plotterStar import *

        print(f"{i} run")
        if i == 0:
            diams = time_step_diameter_rect(traces, 10, 0.04)
        else:
            diams = time_step_diameter(traces, 10, 0.04)
        print(len(diams))
        print(sum(diams))
        print(diams[-1])

        if i == 0:
            fig = go.Figure()
            fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
                         'lines', 'trace')
            fig.write_image("laub_rect.png") 
        # else:
        #     plot_reachtube_stars(traces, None, 1, 2,1)

        
