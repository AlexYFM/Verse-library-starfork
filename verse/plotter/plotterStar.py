from __future__ import annotations
import matplotlib.pyplot as plt 

import copy
import numpy as np
import plotly.graph_objects as go
from typing import List, Tuple, Union
from plotly.graph_objs.scatter import Marker
from verse.analysis.analysis_tree import AnalysisTree, AnalysisTreeNode
from verse.map.lane_map import LaneMap

colors = ['orange', 'blue', 'green', 'red', 'yellow']

def reachtube_tree(
    root: Union[AnalysisTree, AnalysisTreeNode],
    map=None,
    x_dim: int = 0,
    y_dim: int = 1
):
    print("graphing")
    if isinstance(root, AnalysisTree):
        root = root.root
    root = sample_trace(root, 1000)
    agent_list = list(root.agent.keys())
    for agent_id in agent_list:
        i = 0
        #print(agent_id)
        reachtube_tree_single(
            root,
            agent_id,
            x_dim,
            y_dim,
            colors[i])
        i = i+1
    plt.show()


def plot_reach_tube(traces, agent_id):
    trace = np.array(traces[agent_id])
    plot_agent_trace(trace)

def plot_agent_trace(trace):
    for i in range(0, len(trace)):
        x, y = np.array(trace[i][1].get_verts())
        plt.plot(x, y, lw = 1, color = 'blue')
    plt.show()
    


def reachtube_tree_single(root,agent_id,x_dim,y_dim, color):
    queue = [root]
    while queue != []:
        node = queue.pop(0)
        traces = node.trace
        trace = np.array(traces[agent_id])
        #plot the trace
        for i in range(0, len(trace)):
            trace[i][1].show()
            #print(trace[i][1])
            x, y = np.array(trace[i][1].get_verts())
            #print(verts)
            #x=[verts[:,0]]
            #print(x)
            #y=[verts[:,1]]
            #print(y)
            plt.plot(x, y, lw = 1, color = color)
            #plt.show()
        queue += node.child



def sample_trace(root, sample_rate: int = 1):
    queue = [root]
    # print(root.trace)
    if root.type == "reachtube":
        sample_rate = sample_rate * 2
        while queue != []:
            node = queue.pop()
            for agent_id in node.trace:
                trace_length = len(node.trace[agent_id])
                tmp = []
                for i in range(0, trace_length, sample_rate):
                    if i + sample_rate - 1 < trace_length:
                        tmp.append(node.trace[agent_id][i])
                        tmp.append(node.trace[agent_id][i + sample_rate - 1])
                node.trace[agent_id] = tmp
            queue += node.child
    else:
        while queue != []:
            node: AnalysisTreeNode = queue.pop()
            for agent_id in node.trace:
                node.trace[agent_id] = [
                    node.trace[agent_id][i]
                    for i in range(0, len(node.trace[agent_id]), sample_rate)
                ]
            queue += node.child
    return root

