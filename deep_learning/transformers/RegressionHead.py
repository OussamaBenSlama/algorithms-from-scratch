import torch.nn as nn

class RegressionHead(nn.Module):
    def __init__(self, d_model,ouput_dim=1):
        super().__init__()
        self.fc = nn.Linear(d_model, ouput_dim)
        
    def forward(self, x):
        return self.fc(x)