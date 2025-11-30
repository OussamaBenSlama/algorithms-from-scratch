import torch.nn as nn
import torch.nn.functional as F

class ClassifierHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.fc = nn.Linear(d_model, num_classes)
    def forward(self, x):
        logits = self.fc(x)
        return F.log_softmax(logits, dim=-1)