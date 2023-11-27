from torch import nn


class AlexNetwork(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.convulution_layers = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4, padding=0, device=dev),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, padding=2, device=dev),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, padding=1, device=dev),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, padding=1, device=dev),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, padding=1, device=dev),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(6400, 4096, device=dev),              ### 6400 for image size = 224
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, device=dev),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2, device=dev),
        )

    def forward(self, x):
        y = self.convulution_layers(x)
        y = self.flatten(y)
        y = self.linear_layers(y)
        return y
        
