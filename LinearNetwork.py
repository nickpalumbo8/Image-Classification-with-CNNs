from torch import nn

class LinearNetwork(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(224*224*3, 512, device=dev),
            nn.ReLU(),
            nn.Linear(512, 512, device=dev),
            nn.ReLU(),
            nn.Linear(512, 2, device=dev),
        )

    def forward(self, x):
        y = self.flatten(x)
        y = self.linear_layers(y)
        return y
        
