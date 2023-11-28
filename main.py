import torch
from torch import nn

import AlexNetwork
import LinearNetwork
import VGG16
import ImageData
import DataLogger

CUDA_VISIBLE_DEVICES=1

# Set hyperparameters
learning_rates = [1e-5]
batch_sizes = [16] #[ 16 , 32 , 64 ]
epochs = [2, 3, 4]

# Data Location
dataPath = "./dogs-vs-cats/data/train"

# Device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(0))
else:
    print('Warning: No CUDA device available')
    device = torch.device('cpu')


# Disable Debugging APIs
torch.autograd.set_detect_anomaly(enabled=False)
torch.autograd.profiler.emit_nvtx(enabled=False)
torch.autograd.profiler.profile(enabled=False)


loss_fn = nn.CrossEntropyLoss()


def train_loop(dataloader, model, loss_fn, optimizer):
    
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
        #optimizer.zero_grad()
        
        # This method is more efficient than zero_grad()
        # because it does not waste time overwriting
        # the gradients with zeros
        for param in model.parameters():
            param.grad = None

        if ((batch + 1) % batchesPerBarBlock == 0):
            print('#', end='', flush=True)
            
        elif (batch == len(dataloader) - 1):
            print(f"  loss = {loss:>8f}")
    

def test_loop(dataloader, model, loss_fn):

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

for currEpoch in epochs:
    log.logCurrentEpoch(currEpoch)

    for currLR in learning_rates:

        log.logCurrentLR(currLR)

        for currBS in batch_sizes:
        
            log.logCurrentBatchSize(currBS)
            
            ### Model ###
            #model = LinearNetwork.LinearNetwork(device)
            #model = AlexNetwork.AlexNetwork(device)
            model = VGG16.VGG16(device)
            model.cuda()
            
            ### Optimizer ###
            optimizer = torch.optim.Adam(model.parameters(), lr=currLR)
        
        
            ### Data Loaders ###
            trainData, testData = imageData.getBatches(currBS)

            print(f"--- Number of Epochs ({currEpoch}) --- LR ({currLR}) --- Batch Size ({currBS}) ---")
        
            for e in range(currEpoch):

                print(f"Epoch: {e + 1}", end=' ', flush=True)
                
                train_loop(trainData, model, loss_fn, optimizer)
                acc, loss = test_loop(testData, model, loss_fn)
                
                log.logEpochResults(acc, loss)
            
            print()


### Save Results ###

log.saveSummary(r"summary.txt")

