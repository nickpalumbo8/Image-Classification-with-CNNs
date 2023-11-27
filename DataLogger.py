import matplotlib.pyplot as plt


class Log():

    def __init__(self):
        
        self.learningRates = list()
        self.batchSizes = list()
        self.accuracyData = list()
        self.epochs = list()
        
        self.dataIndex = -1
        
        self.allBatchSizesReported = False
        
    
    def logCurrentLR(self, lr):
        
        self.learningRates.append(lr)
        
        if (len(self.learningRates) == 2):
            self.allBatchSizesReported = True
    
    
    def logCurrentBatchSize(self, batchSize):
        
        if (self.allBatchSizesReported == False):
            self.batchSizes.append(batchSize)
        
        self.accuracyData.append(list())
        self.dataIndex += 1
        
    
    def logCurrentEpoch(self, epochs):
        self.epochs.append(epochs)


    def logEpochResults(self, accuracy, averageLoss):
        
        self.accuracyData[self.dataIndex].append([accuracy, averageLoss])
    
    
    def saveSummary(self, fileName):
        
        file = open(fileName, 'w')
        
        print(f"\nSaving summary to \"{fileName}\"\n")
        
        file.write("______________ Training Summary ______________\n\n")
        file.write(f"Learning Rates : {self.learningRates}\n")
        file.write(f"Batch Sizes    : {self.batchSizes}\n")
        file.write(f"Epochs         : {self.epochs}\n")
        
        file.write("______________________________________________\n\n")
        

        for currEpoch in range(len(self.epochs)):
            file.write(f"\nEpoch = {self.epochs[currEpoch]}\n")

            #for currLR in range(len(self.learningRates)):
                # we need to add in the loss function otherwise we will get the same answer everytime
            file.write(f"\nLearning rate = {self.learningRates[currLR]}\n")
        
            #for currBatch in range(len(self.batchSizes)):
            
            file.write(f"\n    Batch size = {self.batchSizes[currBatch]}\n\n")
            
            data = self.accuracyData[currLR * len(self.batchSizes) + currBatch]
            
            numDifferentEpochs = 3

            if currEpoch == 2:
                for i in range(currEpoch*2):
                    file.write(f"        {i + 1}:  Accuracy = {data[i][0]:>0.2f}%  ,  loss = {data[i][1]:>8f}\n")
            if currEpoch == 3:
                for i in range(currEpoch*2):
                    file.write(f"        {i + 1}:  Accuracy = {data[i+4][0]:>0.2f}%  ,  loss = {data[i+4][1]:>8f}\n")
            if currEpoch == 4:
                for i in range(currEpoch*2):
                    file.write(f"        {i + 1}:  Accuracy = {data[i+10][0]:>0.2f}%  ,  loss = {data[i+10][1]:>8f}\n")

    
    
    def saveGraphs(self):
        
        return

