from sleeve_agent import sleeve_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.scenario import ScenarioConfig
import plotly.graph_objects as go
from enum import Enum, auto


class AgentMode(Enum):
    Free = auto()
    Meshed = auto()


if __name__ == "__main__":


    from verse.scenario import Scenario, ScenarioConfig
    from verse.analysis import ReachabilityMethod
    from verse.starsproto.starset import StarSet
    import polytope as pc

    input_code_name = "./demo/dryvr_demo/sleeve_controller.py"
    config = ScenarioConfig(init_seg_length=1, parallel=False)
    scenario = Scenario(config=config, )

    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS


    car = sleeve_agent("sleeve", file_name=input_code_name)
    initial_set_polytope_1 = pc.box2poly([[-0.0168,-0.0166], [0.0029,0.0031], [0,0], [0,0], [0,0]])

    car.set_initial(StarSet.from_polytope(initial_set_polytope_1), tuple([AgentMode.Free]))
    scenario.add_agent(car)

    traces = scenario.verify(0.2, 0.01)
    #traces = scenario.verify(0.2, 0.01)

    
    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    plot_reachtube_stars(traces, None, 0 , 1, filter = 1)

    #traces.dump("./demo/gearbox/output.json")
    #fig = go.Figure()
    #fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], "lines", "trace", sample_rate=1)
    #fig.show()
