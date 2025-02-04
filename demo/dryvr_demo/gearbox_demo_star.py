from sleeve_agent import sleeve_agent
from verse import Scenario
from verse.plotter.plotter2D import *
from verse.scenario import ScenarioConfig
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
    Free = auto()
    Meshed = auto()


if __name__ == "__main__":
    input_code_name = "./demo/dryvr_demo/sleeve_controller.py"
    config = ScenarioConfig(init_seg_length=1, parallel=False)
    scenario = Scenario(config=config)
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.set_sensor(BaseStarSensor())
    car = sleeve_agent("sleeve", file_name=input_code_name)
    
    inf = np.array([-0.0168, 0.0029, 0, 0, 0])
    sup = np.array([-0.0166, 0.0031, 0, 0, 0])

    center = (inf+sup)/2
    basis = np.diag(center-inf)
    C, g = new_pred(5)
    
    car.set_initial(
        StarSet(center, basis, C, g),
        [(AgentMode.Free)]
    )
    scenario.add_agent(car)

    # scenario.set_init(
    #     [
    #         [[-0.0168, 0.0029, 0, 0, 0], [-0.0166, 0.0031, 0, 0, 0]],
    #     ],
    #     [
    #         tuple([AgentMode.Free]),
    #     ],
    # )
    start = time.time()
    traces = scenario.verify(0.2, 0.00001)
    end = time.time()
    print(f'Time: {end-start}')
    diams = time_step_diameter(traces, 0.2, 0.00001)
    print(f'Initial diameter: {diams[0]}\n Final: {diams[-1]}\n Average: {sum(diams)/len(diams)}')
    
    # traces.dump("./demo/gearbox/output.json")
    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2, 3, 4, 5], "lines", "trace", sample_rate=1)
    # fig.show()