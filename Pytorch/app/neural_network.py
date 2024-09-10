import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_chann, num_class):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=in_chann, out_channels=16, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(in_features=62*62*16, out_features=num_class)
        )
    def forward(self, x):
        out = self.model(x)
        return out
    