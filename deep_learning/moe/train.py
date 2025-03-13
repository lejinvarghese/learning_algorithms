from torch import nn
import torch
import math
from tqdm import tqdm

# Generate the Dataset
def generate_data(n_samples=1000):
  X = torch.zeros(n_samples, 2)
  y = torch.zeros(n_samples, dtype=torch.long)

  # Generate samples from two Gaussian distributions
  X[:n_samples//2] = torch.randn(n_samples//2, 2) + torch.Tensor([3,2])
  X[n_samples//2:] = torch.randn(n_samples//2, 2) + torch.Tensor([-3,2])

  # Labels
  for i in range(X.shape[0]):
    if X[i].norm() > math.sqrt(13):
      y[i] = 1

  X[:, 1] = X[:, 1] - 2

  return X, y

data, labels = generate_data()

class Expert(nn.Module):
  def  __init__(self, input_size, output_size): 
    super(Expert, self).__init__()
    self.linear = nn.Linear(input_size, output_size)
    
  def forward(self, data):
    x = self.linear(data)
    return x

class GatingNetwork(nn.Module):
  def __init__(self, input_size, num_experts):
    super(GatingNetwork, self).__init__()
    self.linear1 = nn.Linear(input_size, 4)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(4, num_experts)
    self.softmax = nn.Softmax(dim=-1)
  
  def forward(self, data): 
    x = self.linear1(data)
    x = self.relu(x)
    x = self.linear2(x)
    x = self.softmax(x)
    return x

class MixtureOfExperts(nn.Module):
  def __init__(self, num_experts=2):
    super(MixtureOfExperts, self).__init__()  
    self.expert1 = Expert(2,1)
    self.expert2 = Expert(2,1)
    self.gating =  GatingNetwork(2, num_experts)
    self.sigmoid = nn.Sigmoid()
      
  def forward(self, data):
    expert1_output = self.expert1(data)
    expert2_output = self.expert2(data)  
    
    gating_output =  self.gating(data)

    mixed_output = gating_output[:,0] * expert1_output.squeeze() + gating_output[:,1] * expert2_output.squeeze()
    
    mixed_output_sigmoid = self.sigmoid(mixed_output)
    
    return mixed_output_sigmoid

  def backward(self, y_hat, labels, criterion, optimizer): 
    optimizer.zero_grad()
    loss = criterion(y_hat, labels)    
    loss.backward()
    optimizer.step()
    return loss.item()

if __name__ == "__main__":
    moe = MixtureOfExperts()
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(moe.parameters(),lr=0.01)

    # Convert data and labels to float tensors
    data_tensor = data.float()
    labels_tensor = labels.view(-1, 1).float()

    # Training loop
    num_epochs = 500 
    for epoch in tqdm(range(num_epochs)):
        # Forward pass
        y_hat = moe.forward(data)

        # Backward pass and optimization
        loss_value = moe.backward(y_hat, labels_tensor, criterion, optimizer)
        print(loss_value)