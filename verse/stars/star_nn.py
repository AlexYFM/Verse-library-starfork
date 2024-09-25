import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from starset import *
from scipy.integrate import ode
from sklearn.decomposition import PCA
import pandas as pd

### reinstall torch with cuda on this fork if necessary

### synthetic dynamic and simulation function
def dynamic_test(vec, t):
    x, y = t # hack to access right variable, not sure how integrate, ode are supposed to work
    ### vanderpol
    x_dot = y
    y_dot = (1 - x**2) * y - x

    ### cardiac cell
    # x_dot = -0.9*x*x-x*x*x-0.9*x-y+1
    # y_dot = x-2*y

    ### jet engine
    # x_dot = -y-1.5*x*x-0.5*x*x*x-0.5
    # y_dot = 3*x-y

    ### brusselator 
    # x_dot = 1+x**2*y-2.5*x
    # y_dot = 1.5*x-x**2*y

    ### bucking col -- change center to around -0.5 and keep basis size low
    # x_dot = y
    # y_dot = 2*x-x*x*x-0.2*y+0.1

    ###non-descript convergent system
    # x_dot = y
    # y_dot = -5*x-5*x**3-y
    return [x_dot, y_dot]

def sim_test(
    mode: List[str], initialCondition, time_bound, time_step, 
) -> np.ndarray:
    time_bound = float(time_bound)
    number_points = int(np.ceil(time_bound / time_step))
    t = [round(i * time_step, 10) for i in range(0, number_points)]
    # note: digit of time
    init = list(initialCondition)
    trace = [[0] + init]
    for i in range(len(t)):
        r = ode(dynamic_test)
        r.set_initial_value(init)
        res: np.ndarray = r.integrate(r.t + time_step)
        init = res.flatten().tolist()
        trace.append([t[i] + time_step] + init)
    return np.array(trace)

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

C = np.transpose(np.array([[1,-1,0,0],[0,0,1,-1]]))
g = np.array([1,1,1,1])
basis = np.array([[1, 0], [0, 1]]) * np.diag([.1, .1])
center = np.array([1.40,2.30])

input_size = 1    # Number of input features 
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
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

num_epochs = 50 # sample number of epoch -- can play with this/set this as a hyperparameter
num_samples = 100 # number of samples per time step
lamb = 0.5

T = 7
ts = 0.05

initial_star = StarSet(center, basis, C, g)
# Toy Function to learn: x^2+20

times = torch.arange(0, T+ts, ts) # times to supply, right now this is fixed while S_t is random. can consider making this random as well

C = torch.tensor(C, dtype=torch.float)
g = torch.tensor(g, dtype=torch.float)
# Training loop
S_0 = sample_star(initial_star, num_samples*10) # should eventually be a hyperparameter as the second input, 
np.random.seed()

def sample_initial(num_samples: int = num_samples) -> List[List[float]]:
    samples = []
    for _ in range(num_samples):
        samples.append(S_0[np.random.randint(0, len(S_0))])
    return samples

for epoch in range(num_epochs):
    # Zero the parameter gradients
    optimizer.zero_grad()

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
    
    # ### V_t is now always I -- check that mu should go to zero
    # for i in range(len(times)):
    #     points = post_points[:, i, 1:]
    #     new_center = np.mean(points, axis=0) # probably won't be used, delete if unused in final product
    #     bases.append(torch.eye(points.shape[1], dtype=torch.double))
    #     centers.append(torch.tensor(new_center))

    post_points = torch.tensor(post_points).float()
    ### for now, don't worry about batch training, just do single input, makes more sense to me to think of loss function like this
    ### I would really like to be able to do batch training though, figure out a way to make it work
    for i in range(len(times)):
        # Forward pass
        t = torch.tensor([times[i]], dtype=torch.float)
        flat_bases = model(t)
        n = int(len(flat_bases) ** 0.5) 
        basis = reshaped_output = flat_bases.view(-1, n, n)
        
        # Compute the loss
        r_basis = basis + 1e-6*torch.eye(n) # so that basis should always be inver
        cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(r_basis)@(p-centers[i])-g)) ### pinv because no longer guaranteed to be non-singular
        cont_loss = torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))/num_samples 
        size_loss = torch.log1p(torch.sum(torch.norm(basis, dim=1)))
        loss = lamb*cont_loss + size_loss
        loss.backward()
        # if i==50:
        #     print(model.fc1.weight.grad, model.fc1.bias.grad)
        optimizer.step()
        
        # print(f'Loss: {loss.item()}, mu: {mu.item()}, t: {t}')

    scheduler.step()
    # Print loss periodically
    # print(f'Loss: {loss.item():.4f}')
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}] \n_____________\n')
        print("Gradients of weights and loss", model.fc1.weight.grad, model.fc1.bias.grad)
        for i in range(len(times)):
            t = torch.tensor([times[i]], dtype=torch.float32)
            r_basis = basis + 1e-6*torch.eye(n) # so that basis should always be inver
            cont = lambda p, i: torch.linalg.vector_norm(torch.relu(C@torch.linalg.inv(r_basis)@(p-centers[i])-g)) ### pinv because no longer guaranteed to be non-singular
            cont_loss = torch.sum(torch.stack([cont(point, i) for point in post_points[:, i, 1:]]))/num_samples 
            size_loss = torch.log1p(torch.sum(torch.norm(basis, dim=1)))
            loss = lamb*cont_loss + size_loss
            print(f'containment loss: {cont_loss.item():.4f}, size loss: {size_loss.item():.4f}, time: {t.item():.1f}')


# test the new model

model.eval()
torch.save(model.state_dict(), "./verse/stars/model_weights.pth")

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
    flat_bases = model(test[i]).detach()
    n = int(len(flat_bases) ** 0.5) 
    basis = reshaped_output = flat_bases.view(-1, n, n)[0]
    bases.append(basis.numpy())
    stars.append(StarSet(centers[i], basis.numpy(), C.numpy(), g.numpy()))
    points = torch.tensor(post_points[:, i, 1:]).float()
    contain = torch.sum(torch.stack([cont(point, i) == 0 for point in points]))
    percent_contained.append(contain/(num_samples*10)*100)
    # stars.append(StarSet(center, bases[i], C.numpy(), mu*g.numpy()))
    # stars.append(StarSet(centers[i], bases[i], C.numpy(), np.diag(model(test[i]).detach().numpy())@g.numpy()))

percent_contained = np.array(percent_contained)
# for t in test:
#      print(model(t), t)
# for b in bases:
#      print(b)
# plt.plot(test_times, model(test).detach().numpy())
plot_stars_points_nonit(stars, post_points)

results = pd.DataFrame({
    'time': test.squeeze().numpy(),
    # 'basis': np.array(basis),
    'percent of points contained': percent_contained
})

results.to_csv('./verse/stars/nn_results.csv', index=False)
plt.show()