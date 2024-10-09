import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from starset import *
from scipy.integrate import ode
from sklearn.decomposition import PCA
import pandas as pd
from syn_sim import *

class PostNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PostNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc4(x)
        x = self.relu(x)
        # x = self.tanh(x)
        x = self.fc3(x)
        x = self.relu(x)

        return x

def positional_encoding(time: torch.Tensor, d_model: int) -> torch.Tensor:
    pe = torch.zeros(time.size(0), d_model)
    position = time.unsqueeze(1)  # reshape to (batch_size, 1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

'''
Given the hyperrectangle representation using the minima and maxima, some proportional 0<mu<1, and some number of initial sets
Returns a set of initial sets in a list of starsets
'''
def sample_initial_set(mini: np.ndarray, maxa: np.ndarray, mu: float = 0.1, Ns: int = 10) -> List[StarSet]: #
    if mu>1 or mu<0:
        raise Exception('Invalid mu. Please choose a value of mu between 0 and 1')
    if len(mini)!=len(maxa):
        raise Exception('Vertices of hyperrectangle have different dimensions.')

    diff = maxa-mini
    dim = len(mini)
    C, g = new_pred(dim)
    basis = np.eye(dim)*np.diag(mu*diff/2) # each basis vector just e^i weighted with the difference vector in each dimension. Keeping this as fixed for now
    X0 = []
    for _ in range(Ns):
        center = np.random.uniform(mini+mu/2*diff, maxa-mu*diff/2) 
        # the above along with C, g, basis should make a vector centered in middle of hyperrectangle with min extent at minima and max extent at maxima
        # essentially creating a zonotope representation of hyperrectangle
        X0.append(StarSet(center, basis, C, g))
    
    return X0

'''
Returns a random interval between [0, T] that is length at most Nt and spacing ts
'''
def sample_times(T: float = 7, ts: int = 0.05, Nt: int = 100) -> torch.Tensor:
    start: float
    end: float
    if T<=ts*Nt:
        start = 0
        end = T
    else:
        start = (torch.randint(0, int(T/ts)-Nt, (1,))).item()*ts
        end = start+ts*Nt

    return torch.arange(start, end, ts)

C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
basis = np.array([[1, 0], [0, 1]]) * np.diag([.1, .1])
center = np.array([1.40,2.30])
initial = StarSet(center, basis, C, g)
d_model = 2*initial.dimension() # pretty sure I can't load in weights given different model as input size different, but I can try it


# input_size = 1    # Number of input features 
input_size = basis.flatten().size+d_model    # Number of input features 
hidden_size = 64     # Number of neurons in the hidden layers -- this may change, I know NeuReach has this at default 64
# output_size = 1 + center.shape[0] # Number of output neurons -- this should stay 1 until nn outputs V instead of mu, whereupon it should reflect dimensions of starset
# output_size = 1
output_size = basis.flatten().size

model = PostNN(input_size, hidden_size, output_size)

def he_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Apply He Normal Initialization
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Initialize biases to 0 (optional)

# Apply He initialization to the existing model
model.apply(he_init)
# Use SGD as the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

num_epochs = 30 # sample number of epoch -- can play with this/set this as a hyperparameter
num_samples = 100 # number of samples per time step
lamb = 1

T = 7
ts = 0.05

initial_star = StarSet(center, basis, C, g)
# Toy Function to learn: x^2+20

times = torch.arange(0, T+ts, ts) # times to supply, right now this is fixed while S_t is random. can consider making this random as well
pos = positional_encoding(times, d_model)


C = torch.tensor(C, dtype=torch.float)
g = torch.tensor(g, dtype=torch.float)
# Training loop
S_0 = sample_star(initial_star, num_samples*10) # should eventually be a hyperparameter as the second input, 
np.random.seed()

def sample_initial(num_samples: int = num_samples) -> List[List[float]]:
    sample_indices = torch.randint(0, len(S_0), (num_samples,))
    S = torch.tensor(S_0, dtype=float)
    return S[sample_indices].tolist()

for epoch in range(num_epochs):
    # Zero the parameter gradients

    samples = sample_initial()

    post_points = []
    for point in samples:
        post_points.append(sim_test(None, point, T, ts).tolist())
    post_points = np.array(post_points) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]
    
    centers = [] ### eventually this should be a NN output too
    for i in range(len(times)):
        points = post_points[:, i, 1:]
        new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
        centers.append(torch.tensor(new_center, dtype=torch.float))

    post_points = torch.tensor(post_points).float()
    for i in range(len(times)):
        optimizer.zero_grad()
        flat_bases = model(pos[i])
        n = int(len(flat_bases) ** 0.5) 
        basis = flat_bases.view(-1, n, n)
        
        # Compute the loss
        r_basis = basis + 1e-6*torch.eye(n) # so that basis should always be inver
        cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(r_basis)@(p-centers[i])-g)) ### pinv because no longer guaranteed to be non-singular
        cont_loss = torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))/num_samples 
        size_loss = torch.log1p(torch.sum(torch.norm(basis, dim=1))/n)
        loss = lamb*cont_loss + size_loss
        loss.backward()
        optimizer.step()
        
    scheduler.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}] \n_____________\n')
        # print("Gradients of weights and loss", model.fc1.weight.grad, model.fc1.bias.grad)
        for i in range(len(times)):
            flat_bases = model(pos[i])
            n = int(len(flat_bases) ** 0.5) 
            basis = flat_bases.view(-1, n, n)
            r_basis = basis + 1e-6*torch.eye(n) 
            cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(r_basis)@(p-centers[i])-g)) ### pinv because no longer guaranteed to be non-singular
            cont_loss = torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))/num_samples 
            size_loss = torch.log1p(torch.sum(torch.norm(basis, dim=1)))
            loss = lamb*cont_loss + size_loss
            print(f'containment loss: {cont_loss.item():.4f}, size loss: {size_loss.item():.4f}, time: {i*ts:.1f}')


# test the new model

model.eval()
torch.save(model.state_dict(), "./verse/stars/model_weights_org.pth")

# S_0 = sample_star(initial_star, num_samples*10) ### this is critical step -- this needs to be recomputed per training step
S = sample_initial(num_samples*10)
post_points = []
for point in S:
        post_points.append(sim_test(None, point, T, ts).tolist())
post_points = np.array(post_points) ### this has shape N x (T/ts) x (n+1), S_t is equivalent to p_p[:, t, 1:]

test_times = torch.arange(0, T+ts, ts)
test = torch.reshape(test_times, (len(test_times), 1))
centers = [] ### eventually this should be a NN output too
for i in range(len(times)):
    points = post_points[:, i, 1:]
    new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
    centers.append(torch.tensor(new_center, dtype=torch.float))

post_points = torch.tensor(post_points).float()

stars = []
percent_contained = []
cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(basis + 1e-6*torch.eye(n))@(p-centers[i])-g)) ### pinv because no longer guaranteed to be non-singular
bases = []
for i in range(len(times)):
    # mu, center = model(test[i])[0].detach().numpy(), model(test[i])[1:].detach().numpy()
    flat_bases = model(pos[i])
    n = int(len(flat_bases) ** 0.5) 
    basis = flat_bases.view(-1, n, n)[0]
    bases.append(basis.detach().numpy())
    stars.append(StarSet(centers[i], basis.detach().numpy(), C.numpy(), g.numpy()))
    points = torch.tensor(post_points[:, i, 1:]).float()
    contain = torch.sum(torch.stack([cont(point, i) == 0 for point in points]))
    percent_contained.append(contain/(num_samples*10)*100)
    # stars.append(StarSet(center, bases[i], C.numpy(), mu*g.numpy()))
    # stars.append(StarSet(centers[i], bases[i], C.numpy(), np.diag(model(test[i]).detach().numpy())@g.numpy()))

percent_contained = np.array(percent_contained)
plot_stars_points_nonit(stars, post_points)

results = pd.DataFrame({
    'time': test.squeeze().numpy(),
    # 'basis': np.array(basis),
    'percent of points contained': percent_contained
})

results.to_csv('./verse/stars/nn_results.csv', index=False)
plt.show()