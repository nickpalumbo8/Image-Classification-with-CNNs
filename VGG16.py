# import torch
from torch import nn

# from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision import transforms
# from torchvision.transforms import ToTensor

# import matplotlib.pyplot as plt

# import os







# Set hyperparameters
# learning_rate = 1e-5
# batch_size = 16
# epochs = 5

# device=torch.device('cuda')
# data_transform = transforms.Compose([
#     transforms.Resize(size=(224, 224)),
#     transforms.ToTensor()
# ])

# dataset = datasets.ImageFolder(root="C:/Users/nickp/OneDrive/Desktop/dogs-vs-cats/data/train", transform = data_transform)

# Save 75% of samples for training, 25% for validation
# train_size = int(0.75 * len(dataset))
# test_size = len(dataset) - train_size

# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
# test_dataloader = DataLoader(test_dataset, batch_size=batch_size)




class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# Create VGG16 model
# vgg_model = VGG16()
# vgg_model.to(device)
# Print the model summary
#print(vgg_model)





# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(vgg_model.parameters(), lr=learning_rate)

# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X = X.to(device)
#         y = y.to(device)
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 1 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss} {current}/{size}")

# def test_loop(dataloader, model, loss_fn):
#     model.eval()
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)

#     test_loss, correct = 0, 0

#     with torch.no_grad():
#         for X, y in dataloader:
#             X = X.to(device)
#             y = y.to(device)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error:\n Accuracy: {(100 * correct):>0.1f}%, Avg. loss: {test_loss:>8f}\n");

# epochs = 10
# for t in range(epochs):
#     print(f"Epoch: {t + 1}")
#     train_loop(train_dataloader, vgg_model, loss_fn, optimizer)
#     test_loop(test_dataloader, vgg_model, loss_fn)