from dryvr_agent import LaubLoomisAgent 
from verse import Scenario, ScenarioConfig
from verse.plotter.plotter2D import * 

import plotly.graph_objects as go 
from enum import Enum, auto 

from verse.stars.starset import *
from verse.sensor.base_sensor_stars import *
from verse.analysis.verifier import ReachabilityMethod

class AgentMode(Enum):
    Default = auto()

if __name__ == "__main__":
    scenario = Scenario(ScenarioConfig(parallel=False))
    W = 0.1
    
    agent = LaubLoomisAgent('laub')
    # The initial position of the quadrotor is uncertain in 
    # all directions within [−0.4, 0.4] [m] and also the velocity 
    # is uncertain within [−0.4, 0.4] [m/s] for all directions
    
    # The inertial (north) position x1, the inertial (east) position x2, 
    # the altitude x3, the longitudinal velocity x4, 
    # the lateral velocity x5, the vertical velocity x6, 
    # the roll angle x7, the pitch angle x8, 
    # the yaw angle x9, the roll rate x10, 
    # the pitch rate x11, and the yaw rate x12.
    scenario.config.reachability_method = ReachabilityMethod.STAR_SETS
    scenario.config.model_path = 'laubloomis'
    X = 2*W
    scenario.config.model_hparams = {
        "big_initial_set": (np.array([1.2-X, 1.05-X, 1.5-X, 2.4-X, 1-X, 0.1-X, 0.45-X]), np.array([1.2+X, 1.05+X, 1.5+X, 2.4+X, 1+X, 0.1+X, 0.45+X])),
        "initial_set_size": 1,
    }
    # scenario.config.pca = False
    basis = np.eye(7)*np.diag([W for _ in range(7)])
    center = np.array([1.2,1.05,1.5,2.4,1,0.1,0.45])

    C, g = new_pred(7)
    # C = np.transpose(np.array([[1,-1,0,0,0,0],[0,0,1,-1,0,0], [0,0,0,0,1,-1]]))
    # g = np.array([1,1,1,1,1,1])
    scenario.set_sensor(BaseStarSensor())
    agent.set_initial(
        
            StarSet(center, basis, C, g)
        ,
        
            (AgentMode.Default, )
        
    )
    scenario.add_agent(agent)

    traces = scenario.verify(20, 0.01)

    stars = []
    for node in traces.nodes:
        s_mode = []
        for star in node.trace['laub']:
            s_mode.append(star)
        stars.append(s_mode)
    # plot_stars_points(stars)
    verts = []
    i = 0
    for s_mode in stars:
        v_mode = []
        # star[1].print()
        for star in s_mode:
            v_mode.append([star[0], *star[1].get_max_min(3)])
            # if i==0 and star[0]>=1.0:
            #     break
            # if i==1 and star[0]>=2.5:
            #     break
            # if i==2 and star[0]>=3.5:
            #     break
        v_mode = np.array(v_mode)
        verts.append(v_mode)
        # print([star[0], *star[1].get_max_min(0)])
#    print('Vertices:', verts)
    # verts = np.array(verts)
    #print(np.all(verts[:,2]>verts[:,1]))
    for i in range(len(verts)):
        v_mode = verts[i]
        # plt.plot(v_mode[:, 0], v_mode[:, 1], 'b.')
        # plt.plot(v_mode[:,0], v_mode[:, 2], 'r.')
        plt.fill_between(v_mode[:, 0], v_mode[:, 1], v_mode[:, 2], alpha=0.5)

    plt.title('Laubloomis Example')
    plt.ylabel('u')
    plt.xlabel('Time (s)')
    plt.show()
    # fig = go.Figure() 
    # fig = reachtube_tree(traces, None, fig, 0, 4, [1,3], "lines", "trace")
    # fig.show()