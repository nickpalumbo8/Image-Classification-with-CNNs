import matplotlib.pyplot as plt


class Log():

    def __init__(self):
        
        self.learningRates = list()
        self.batchSizes = list()
        self.accurracyData = list()
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
        
        self.accurracyData.append(list())
        self.dataIndex += 1
        
    
    def logCurrentEpoch(self, epochs):
        self.epochs.append(epochs)


    def logEpochResults(self, accurracy, averageLoss):
        
        self.accurracyData[self.dataIndex].append([accurracy, averageLoss])
    
    
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

            for currLR in range(len(self.learningRates)):
        
                file.write(f"\nLearning rate = {self.learningRates[currLR]}\n")
        
                for currBatch in range(len(self.batchSizes)):
            
                    file.write(f"\n    Batch size = {self.batchSizes[currBatch]}\n\n")
            
                    dat = self.accurracyData[currLR * len(self.batchSizes) + currBatch]
                    
                    for i in range(len(dat)):
                    
                        file.write(f"        {i + 1}:  Accuracy = {dat[i][0]:>0.2f}%  ,  loss = {dat[i][1]:>8f}\n")
    
    
    def saveGraphs(self):
        
        return

