from origin_agent import volterra_agent
from verse.scenario import Scenario, ScenarioConfig
from verse.plotter.plotter2D import *
from verse.sensor.example_sensor.craft_sensor import CraftSensor

import plotly.graph_objects as go
from enum import Enum, auto
import time

from verse.analysis import ReachabilityMethod
from verse.stars.starset import *
from verse.sensor.base_sensor_stars import *
import polytope as pc

from verse.utils.star_diams import *

class CraftMode(Enum):
    inside = auto()
    outside = auto()


if __name__ == "__main__":
    input_code_name = './demo/paper_demo/volterra_controller.py'
    #scenario = Scenario()
    scenario = Scenario(ScenarioConfig(parallel=False, init_seg_length=1))


    scenario.config.model_path = 'volterra_big_n'
    car = volterra_agent('test', file_name=input_code_name)
    # scenario.add_agent(car)

    # modify mode list input
    e = .012
    # scenario.set_init(
    #     [
    #         [[1.3 - e, 1, 0], [1.3 + e, 1, 0]],
    #     ],
    #     [
    #         tuple([CraftMode.outside]),
    #     ]
    # )

    # traces = scenario.simulate(3.64, .01)
    # fig = go.Figure()
    # fig = simulation_anime(traces, None, fig, 1, 2, [
    #                        1, 2], 'lines', 'trace', sample_rate=1)
    # fig.show()

    # initial_set_polytope = pc.box2poly([[1.3 - e, 1.3 + e], [1,1.025], [0,0.025]])
    sup = np.array([1.3+e, 1.025, 0.025])
    inf = np.array([1.3-e, 1, 0])
    center = (sup+inf)/2
    C, g = new_pred(3)
    initial_star = StarSet(center, np.diag(center-inf), C, g)
    # car.set_initial(StarSet.from_polytope(initial_set_polytope), (CraftMode.outside,))
    car.set_initial(initial_star, (CraftMode.outside,))

    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.add_agent(car)

    scenario.set_sensor(BaseStarSensor())


    start_time = time.time()
    # scenario.config.overwrite = True # bug with overwrite true making it so that next mode is never reached

    traces = scenario.verify(3.64, .01)
    run_time = time.time() - start_time

    print({
        "tool": "verse",
        "benchmark": "LOVO21",
        "setup": "",
        "result": "1",
        "time": run_time,
        #"metric2": str( abs((traces.nodes[-1].trace['test'][-1][1] - traces.nodes[-1].trace['test'][-2][1] )*(traces.nodes[-1].trace['test'][-1][2] - traces.nodes[-1].trace['test'][-2][2] ) ) ),
        "metric3": "",
    })

    diams = time_step_diameter(traces, 3.64, .01)
    print(len(diams))
    print(sum(diams))
    print(diams[0])
    print(diams[-1])

    import plotly.graph_objects as go
    from verse.plotter.plotterStar import *

    # plot_reachtube_stars(traces, None, 0, 1, 1)
    # sim = scenario.simulate_multi(3.64, 0.01)
    plot_stars_time(traces, 1)
    # points = sample_star()
    #plot_star_reachtubes(traces, 0, 1)


    # fig = go.Figure()
    # fig = reachtube_tree(traces, None, fig, 1, 2, [1, 2],
    #                      'lines', 'trace')
    # fig.add_trace(go.Scatter(x=[
    #     1 + 0.161, 1 + 0.06669, 1 + -0.06669, 1 + -0.161, 1 + -0.161, 1 + -0.06669, 1 + 0.06669, 1 + 0.161, 1 + 0.161
    # ],
    #     y=[
    #         1 + 0.06669, 1 + 0.161, 1 + 0.161, 1 + 0.06669, 1 + -0.06669, 1 + -0.161, 1 + -0.161, 1 + -0.06669,
    #         1 + 0.06669
    #     ]
    # ))
    # fig.add_shape(type="circle",
    #               xref="x", yref="y",
    #               x0=1 - .161, y0=1 - .161, x1=1 + .161, y1=1 + .161,
    #               line_color="LightSeaGreen",
    #               )

    # fig.show()

