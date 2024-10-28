import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from starset import *
from scipy.integrate import ode
from sklearn.decomposition import PCA
import pandas as pd
from syn_sim import *
from star_nn_utils import *

def eval(initial_star: StarSet, model_path: str, sim_test: Callable, T: int = 7, ts: int = 0.05, num_samples: int = 250) -> None: 
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # S_0 = sample_star(initial_star, num_samples*10) ### this is critical step -- this needs to be recomputed per training step
    S = sample_star(initial, num_samples)
    post_points = []
    for point in S:
            post_points.append(sim_test(None, point, T, ts).tolist())
    post_points = np.array(post_points) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]

    test_times = torch.arange(0, T+ts, ts)
    pos = positional_encoding(test_times, d_model)
    test = torch.reshape(test_times, (len(test_times), 1))
    centers = [] ### eventually this should be a NN output too
    for i in range(len(test_times)):
        points = post_points[:, i, 1:]
        new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
        centers.append(torch.tensor(new_center, dtype=torch.float))

    post_points = torch.tensor(post_points).float()

    stars = []
    percent_contained = []
    cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(basis + 1e-6*torch.eye(n))@(p-centers[i])-g)) ### pinv because no longer guaranteed to be non-singular
    bases = []
    C, g, x0 = initial_star.C, initial_star.g, torch.tensor(initial_star.center, dtype=torch.float)
    C = torch.tensor(C, dtype=torch.float)
    g = torch.tensor(g, dtype=torch.float)
    
    for i in range(len(test_times)):
        flat_bases = model(torch.cat((x0, torch.tensor(initial_star.basis, dtype=torch.float).flatten(), pos[i]), dim=-1)) # note that I changed the order here so that it matches the math/pseudocode
        n = int(len(flat_bases) ** 0.5) 
        basis = flat_bases.view(-1, n, n)[0]
        stars.append(StarSet(centers[i], basis.detach().numpy(), C.numpy(), g.numpy()))
        points = torch.tensor(post_points[:, i, 1:]).float()
        contain = torch.sum(torch.stack([cont(point, i) == 0 for point in points]))
        percent_contained.append(contain/(num_samples)*100)
        size_loss = torch.sqrt(torch.sum(torch.norm(basis, dim=1)))
        bases.append(size_loss.detach().numpy())

    percent_contained = np.array(percent_contained)
    plot_stars_points_nonit(stars, post_points)
    # plot_stars_points(stars)
    plt.title(f'Accuracy: {np.mean(np.array(percent_contained))}%')
    plt.show()

    results = pd.DataFrame({
        'time': test.squeeze().numpy(),
        'basis': np.array(bases),
        'percent of points contained': percent_contained
    })

    results.to_csv('./verse/stars/nn_results.csv', index=False)

'Zonotope predicate'
C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
'Triangle predicate'
# C = np.transpose(np.array([[-1,0,1],[0,-1,1]]))
# g = np.array([1/3, 1/3, 1/3])

basis = np.array([[1, 0], [0, 1]]) * np.diag([.025, .025])
center = np.array([1.4,2.3])
initial = StarSet(center, basis, C, g)
d_model = 2*initial.dimension() 

initial_star = StarSet(center, basis, C, g)


input_size = initial.n+basis.flatten().size+d_model
hidden_size = 64     # Number of neurons in the hidden layers -- this may change, I know NeuReach has this at default 64
output_size = basis.flatten().size

model = PostNN(input_size, hidden_size, output_size)
eval(initial_star, "./verse/stars/model_weights_vdp.pth", sim_test)

results = pd.read_csv('./verse/stars/nn_results.csv').to_numpy()
plt.plot(results[:, 0], results[:, 1]/max(results[:,1]), label='Size Loss')
plt.plot(results[:, 0], results[:, 2]/max(results[:,2]), label='Percent Containment')
plt.xlabel('Time (s)')
plt.ylabel(f'Normalized based on {max(results[:,2])}% and {max(results[:,1])}')
plt.legend()
plt.show()