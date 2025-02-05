from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M3
from verse.scenario.scenario import Benchmark, ReachabilityMethod

# from noisy_sensor import NoisyVehicleSensor
from verse.plotter.plotter2D import *

from enum import Enum, auto
import sys
import plotly.graph_objects as go

import pyvista as pv
from verse.plotter.plotter3D import *

from verse.stars.starset import *
from verse.sensor.base_sensor_stars import *
import time 
from verse.utils.star_diams import *
class LaneObjectMode(Enum):
    Vehicle = auto()
    Ped = auto()  # Pedestrians
    Sign = auto()  # Signs, stop signs, merge, yield etc.
    Signal = auto()  # Traffic lights
    Obstacle = auto()  # Static (to road/lane) obstacles


class AgentMode(Enum):
    Normal = auto()
    SwitchLeft = auto()
    SwitchRight = auto()
    Brake = auto()


class TrackMode(Enum):
    T0 = auto()
    T1 = auto()
    T2 = auto()
    M01 = auto()
    M12 = auto()
    M21 = auto()
    M10 = auto()


class State:
    x: float
    y: float
    theta: float
    v: float
    agent_mode: AgentMode
    track_mode: TrackMode

    def __init__(self, x, y, theta, v, agent_mode: AgentMode, track_mode: TrackMode):
        pass


if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "example_controller5.py")

    scenario = Scenario(ScenarioConfig(parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    car = CarAgent("car1", file_name=input_code_name)
    car = NPCAgent("car2")
    car = NPCAgent("car3")
    tmp_map = M3()
    scenario.set_map(tmp_map)
    time_step = 0.1
    C, g = new_pred(4)
    
    car1 = CarAgent("car1", file_name=input_code_name)
    verts = np.array([[5, -0.5, 0, 1.0], [5.5, 0.5, 0, 1.0]])
    center = (verts[0]+verts[1])/2
    basis = np.diag(center-verts[0])
    car1.set_initial(
        StarSet(center, basis, C, g),
        (
                AgentMode.Normal,
                TrackMode.T1,
            )
    )

    car2 = NPCAgent("car2")
    verts = np.array([[20, -0.2, 0, 0.5], [20, 0.2, 0, 0.5]])
    center = (verts[0]+verts[1])/2
    basis = np.diag(center-verts[0])
    car2.set_initial(
        StarSet(center, basis, C, g),
                (
                AgentMode.Normal,
                TrackMode.T1,
            )
    )

    car3 = NPCAgent("car3")
    verts = np.array([[4 - 2.5, 2.8, 0, 1.0], [4.5 - 2.5, 3.2, 0, 1.0]])
    center = (verts[0]+verts[1])/2
    basis = np.diag(center-verts[0])
    car3.set_initial(
        StarSet(center, basis, C, g),
                (
                AgentMode.Normal,
                TrackMode.T0,
            )
    )

    scenario.add_agent(car1)
    scenario.add_agent(car2)
    scenario.add_agent(car3)

    time_step = 0.2
    
    scenario.set_sensor(BaseStarSensor())

    start = time.time()
    trace = scenario.verify(40, time_step)
    runtime = time.time()-start
    print(f'Runtime: {runtime}')

    diams = time_step_diameter(trace, 40, 0.2)

    # plot_reachtube_stars(trace, tmp_map, filter=1)
    print(diams[-1])
    print(len(diams))
    print(sum(diams))
    # fig = pv.Plotter()
    # fig = plot3dReachtube(traces,'car1',1,2,0,'b',fig)
    # fig = plot3dReachtube(traces,'car2',1,2,0,'r',fig)
    # fig = plot3dReachtube(traces,'car3',1,2,0,'g',fig)
    # fig = plot_line_3d([0,0,0],[10,0,0],ax=fig,color='r')
    # fig = plot_line_3d([0,0,0],[0,10,0],ax=fig,color='g')
    # fig = plot_line_3d([0,0,0],[0,0,10],ax=fig,color='b')
    # fig.show()
