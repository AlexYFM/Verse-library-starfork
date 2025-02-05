from verse.agents.example_agent import CarAgent, NPCAgent
from verse.map.example_map.map_tacas import M1
from verse.scenario.scenario import Benchmark
from enum import Enum, auto
from verse.plotter.plotter2D import *
from verse import Scenario, ScenarioConfig
from verse.analysis.verifier import ReachabilityMethod
from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *

import sys
import plotly.graph_objects as go

import time
from verse.plotter.plotterStar import *
from verse.utils.star_diams import *
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


if __name__ == "__main__":
    import os

    script_dir = os.path.realpath(os.path.dirname(__file__))
    input_code_name = os.path.join(script_dir, "example_controller4.py")
    car = CarAgent('car1', file_name=input_code_name)
    car2 = NPCAgent('car2')
    scenario = Scenario(ScenarioConfig(parallel=False))
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS

    scenario.config.model_path = 'highway_diff_center_bench_large'

    center = np.array([0.005, 0, 0, 1])
    basis = np.eye(4)*np.diag([0.005, 0.5, 0, 0])
    C, g = new_pred(4)
    car.set_initial(
        initial_state=StarSet(center, basis, C,g),
        initial_mode=(AgentMode.Normal, TrackMode.T1)
    )

    center2=np.array([15, 0, 0, 0.5])
    basis2 = np.eye(4)*np.diag([0, 0.3, 0, 0])
    car2.set_initial(
        initial_state=StarSet(center2, basis2, C, g),
        initial_mode=(AgentMode.Normal, TrackMode.T1)
    )

    scenario.add_agent(car)
    scenario.add_agent(car2)
    tmp_map = M1()
    scenario.set_map(tmp_map)
    scenario.set_sensor(BaseStarSensor())
    # traces = scenario.simulate(70, 0.05)
    # # traces.dump('./output1.json')
    # fig = go.Figure()
    # fig = simulation_anime(traces, tmp_map, fig, 1,
    #                        2, [1, 2], 'lines', 'trace', anime_mode='trail', full_trace = True)
    # fig.show()

    # traces = scenario.verify(40, 0.05,
    #                          reachability_method='NeuReach',
    #                          params={
    #                              "N_X0": 1,
    #                              "N_x0": 50,
    #                              "N_t": 500,
    #                              "epochs": 50,
    #                              "_lambda": 5,
    #                              "use_cuda": True,
    #                              'r': 0,
    #                          }
    #                         )
    # traces.dump(os.path.join(script_dir, "output6_neureach.json")
    time_step = 0.05
    start = time.time()
    trace = scenario.verify(40,0.1) # increasing ts to 0.1 to increase learning speed, do the same for dryvr
    end = time.time()
    print(f'Time: {end-start}')
    diams = time_step_diameter(trace, 40, 0.1)
    print(f'Initial diameter: {diams[0]}\n Final: {diams[-1]}\n Average: {sum(diams)/len(diams)}')
    # plot_stars_time(trace)
    plot_reachtube_stars(trace,tmp_map, filter=1)