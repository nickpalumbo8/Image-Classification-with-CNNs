import torch
from torch import nn

import AlexNetwork
import LinearNetwork
import VGG16
import ImageData
import DataLogger

# Set hyperparameters
learning_rates = [1e-5]
batch_sizes = [ 32 , 64 , 128 ]
epochs = 2

# Data Location
dataPath = "C:/Users/nickp/OneDrive/Desktop/dogs-vs-cats/data/train"

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print('Warning: No CUDA device available')
    device = torch.device('cpu')


loss_fn = nn.CrossEntropyLoss()


def train_loop(dataloader, model, optimizer):
    
    batchesPerBarBlock = round(len(dataloader) / 20)
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if ((batch + 1) % batchesPerBarBlock == 0):
            print('#', end='', flush=True)
            
        elif (batch == len(dataloader) - 1):
            print(f"  loss = {loss:>8f}")
    

def test_loop(dataloader, model):

    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    correct *= 100
    
    return correct, test_loss




### Initialize Log ###
log = DataLogger.Log()


### Initial Data Loading ###
imageData = ImageData.ImageDataLoader(dataPath)


for currLR in learning_rates:

    log.logCurrentLR(currLR)

    for currBS in batch_sizes:
    
        log.logCurrentBatchSize(currBS)
        
        ### Model ###
        #model = LinearNetwork.LinearNetwork(device)
        #model = AlexNetwork.AlexNetwork(device)
        model = VGG16.VGG16(device)
        
        ### Optimizer ###
        optimizer = torch.optim.Adam(model.parameters(), lr=currLR)
    
    
        ### Data Loaders ###
        trainData, testData = imageData.getBatches(currBS)
        
        print(f"--- LR ({currLR}) --- Batch Size ({currBS})")
    
        for e in range(epochs):

            print(f"Epoch: {e + 1}", end=' ', flush=True)
            
            train_loop(trainData, model, optimizer)
            acc, loss = test_loop(testData, model)
            
            log.logEpochResults(acc, loss)
        
        print()


### Save Results ###

log.saveSummary(r"summary.txt")

