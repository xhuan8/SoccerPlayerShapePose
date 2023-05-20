import torch.nn.functional as F
from torch import nn

class ClassifyNet(nn.Module):
    def  __init__(self):
        super(ClassifyNet,self).__init__()

        self.fc1 = nn.Linear(1000, 500)
        self.fc2 = nn.Linear(500, 2)
        
    def forward(self, input):
        x = F.relu(self.fc1(input))
        x = self.fc2(x)
        return x