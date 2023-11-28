from torch import nn


class AlexNetwork(nn.Module):
    def __init__(self, dev):
        super().__init__()
        self.convulution_layers = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0, device=dev),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, device=dev),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, device=dev),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, device=dev),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, device=dev),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.flatten = nn.Flatten()
        self.linear_layers = nn.Sequential(
            nn.Linear(6400, 4096, device=dev),              ### 6400 for image size = 224
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096, device=dev),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 2, device=dev),
        )

    def forward(self, x):
        y = self.convulution_layers(x)
        y = self.flatten(y)
        y = self.linear_layers(y)
        return y
        
