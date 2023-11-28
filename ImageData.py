import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor




class ImageDataLoader():

    def __init__(self, path, trainPercent=0.75, imageDim=224, subset=False):
        
        data_transform = transforms.Compose([
            transforms.Resize(size=(imageDim, imageDim)),
            transforms.ToTensor()
        ])
    
        dataset = datasets.ImageFolder(root=path, transform=data_transform)

        # Save 75% of samples for training, 25% for validation
        length = len(dataset)
        if subset == True:
            length = int(len(dataset) * 0.25)

        train_size = int(trainPercent * length)
        test_size = length - train_size
        excess_size = len(dataset) - train_size - test_size

        self.train_dataset, self.test_dataset, self.excess_dataset = torch.utils.data.random_split(dataset, [train_size, test_size, excess_size])

    def getBatches(self, batchSize):
        
        train_dataloader = DataLoader(self.train_dataset, batch_size=batchSize)
        test_dataloader = DataLoader(self.test_dataset, batch_size=batchSize)
        
        return train_dataloader, test_dataloader
    
