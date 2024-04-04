from origin_agent import craft_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto


class CraftMode(Enum):
    ProxA = auto()
    ProxB = auto()
    Passive = auto()


if __name__ == "__main__":
    from verse.scenario import Scenario, ScenarioConfig
    from verse.analysis import ReachabilityMethod
    input_code_name = "./demo/dryvr_demo/rendezvous_controller.py"
    config = ScenarioConfig(parallel=False)

    scenario = Scenario(config=config)
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS

    car = craft_agent("test", file_name=input_code_name)
    scenario.add_agent(car)
    scenario.set_sensor(CraftSensor())
    # modify mode list input

    from verse.starsproto.starset import StarSet
    import polytope as pc

    initial_set_polytope_1 = pc.box2poly([[-925,-875], [-425,-375], [0,0], [0,0], [0,0], [0,0]])

    scenario.set_init(
        [
           StarSet.from_polytope(initial_set_polytope_1)
        ],
        [
            tuple([CraftMode.ProxA]),
        ],
    )
    traces = scenario.verify(200, .1)
    #fig = go.Figure()
    #fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2], "lines", "trace")
    #fig.show()

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    plot_reachtube_stars(traces, None, 0 , 1, 1)
