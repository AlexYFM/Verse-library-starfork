from origin_agent import craft_agent
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto

from verse.analysis.verifier import ReachabilityMethod
from verse.stars.starset import *

from verse.sensor.base_sensor_stars import *
class CraftMode(Enum):
    ProxA = auto()
    ProxB = auto()
    Passive = auto()


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/rendezvous_controller.py"
    scenario = Scenario(ScenarioConfig(parallel=False))

    car = craft_agent("test", file_name=input_code_name)
    scenario.set_sensor(BaseStarSensor())
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    # modify mode list input
    scenario.config.model_path = 'spacecraft_small_init_sublinloss_fit_loss'

    scenario.config.model_hparams = {
        "big_initial_set": (np.array([0,-0.5,0,0,0,0]), np.array([15,0.5,0,0,0,0])), # irrelevant for now
        "initial_set_size": 1,
        "lamb": 7,
        "num_epochs": 30,
        "gamma":0.99,
        "lr":1e-4,
        "sublin_loss":True,
        # "num_samples": 100,
        # "Ns": 1
    }

    infin = np.array([-925, -425, 0, 0, 0, 0])
    sup = np.array([-875, -375, 0, 0, 0, 0])

    # off = np.array([20, 20, 0, 0, 0, 0])
    # infin = np.array([-925, -425, 0, 0, 0, 0])+off
    # sup = np.array([-875, -375, 0, 0, 0, 0])-off

    center = (infin+sup)/2
    basis = np.eye(6)*np.diag(center-infin)
    C, g = new_pred(6)

    car.set_initial(
        initial_state=StarSet(center, basis, C, g),
        initial_mode=tuple([CraftMode.ProxA])
    )

    scenario.add_agent(car)
    # scenario.set_init(
    #     [
    #         [[-925, -425, 0, 0, 0, 0], [-875, -375, 0, 0, 0, 0]],
    #     ],
    #     [
    #         tuple([CraftMode.ProxA]),
    #     ],
    # )

    traces = scenario.verify(200, 1)
    # plot_reachtube_stars(traces, filter=2)
    plot_stars_time(traces, 2, scenario_agent=car)
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2], "lines", "trace")
    plt.show()