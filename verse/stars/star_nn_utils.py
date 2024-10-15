import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from scipy.integrate import ode
import pandas as pd

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

def create_model(input_size: int, hidden_size: int, output_size: int) -> PostNN:
    return PostNN(input_size, hidden_size, output_size)

def model_he_init(model: PostNN) -> None: 
    def he_init(m) -> None: 
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Apply He Normal Initialization
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)  # Initialize biases to 0 
    
    model.apply(he_init)