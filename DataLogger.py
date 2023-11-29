import matplotlib
import matplotlib.pyplot as plt


class Log():
    """
    Stores the accuracy and loss of every epoch and
    saves it in a readable format.
    
    The data is stored in the following manner:
    
        Learing Rates:         LR_1                    LR_2                    LR_3    ...
                             /   |   \               /   |   \               /   |   \
        Batch sizes:     BS_1  BS_2  BS_3 ...    BS_1  BS_2  BS_3 ...    BS_1  BS_2  BS_3 ...
                           |     |     |           |     |     |           |     |     |
        Epochs:           e1    e1    e1          e1    e1    e1          e1    e1    e1
                          e2    e2    e2          e2    e2    e2          e2    e2    e2
                          e3    e3    e3          e3    e3    e3          e3    e3    e3
                           .     .     .           .     .     .           .     .     .
                           .     .     .           .     .     .           .     .     .
                           .     .     .           .     .     .           .     .     .
    """
    
    def __init__(self):
        
        self.learningRates = list()
        self.batchSizes = list()
        self.epochData = list()
        
        self.dataIndex = -1
        
        self.allBatchSizesReported = False
        
    
    def logCurrentLR(self, lr):
        
        self.learningRates.append(lr)
        
        if (len(self.learningRates) == 2):
            self.allBatchSizesReported = True
    
    
    def logCurrentBatchSize(self, batchSize):
        
        if (self.allBatchSizesReported == False):
            self.batchSizes.append(batchSize)
        
        self.epochData.append(list())
        self.dataIndex += 1


    def logEpochResults(self, accuracy, averageLoss):
        
        self.epochData[self.dataIndex].append([accuracy, averageLoss])
    
    
    def saveSummary(self, fileName):
        
        file = open(fileName, 'w')
        
        print(f"\nSaving summary to \"{fileName}\"\n")
        
        file.write("______________ Training Summary ______________\n\n")
        file.write(f"Learning Rates : {self.learningRates}\n")
        file.write(f"Batch Sizes    : {self.batchSizes}\n")
        file.write(f"Epochs         : {len(self.epochData[0])}\n")
        
        file.write("______________________________________________\n\n")
        

        for currLR in range(len(self.learningRates)):
    
            file.write(f"\nLearning rate = {self.learningRates[currLR]}\n")
    
            for currBatch in range(len(self.batchSizes)):
        
                file.write(f"\n    Batch size = {self.batchSizes[currBatch]}\n\n")
        
                data = self.epochData[currLR * len(self.batchSizes) + currBatch]
                
                for i in range(len(data)):
                
                    file.write(f"        {i + 1}:  Accuracy = {data[i][0]:>0.2f}%  ,  loss = {data[i][1]:>8f}\n")
        
        
        
        file.write("\n\n______________________________________________\n")
        file.write(f"Raw Epoch Data:\n\n{self.epochData}\n\n")

    
    
    def saveGraphs(self):
        # Accuracy vs Learning Rate
        # Accuracy vs Batch size

        xs = []
        ys = []

        for currBatch in range(len(self.batchSizes)):
            for currLR in range(len(self.learningRates)):
                data = self.epochData[currLR * len(self.batchSizes) + currBatch]

                xs.append(self.learningRates[currLR])
                ys.append(data[-1][0])

            plt.plot(xs, ys, label=f"Batch size = {self.batchSizes[currBatch]}")
            plt.xticks(xs)
            xs = []; ys = []

        plt.legend()
        plt.xscale('log')
        plt.savefig('test.png')
        return
        
        
    #
    # Creates a bar graph showing the accuracy after training with
    # different Learning Rates with a constant batch size and number of epochs.
    #
    # !!! If multiple batch sizes are used, this method will not do anything !!!
    #
    def saveBarGraph_LR(self, modelName, color='blue', width=0.3):
            # Accuracy vs Learning Rate

            if (len(self.batchSizes) > 1):
                
                print("Learning rate bar graph - skipping")
                return
                
            else:
            
                print("Learning rate bar graph - generating")

            xs = []
            ys = []
            
            for currLR in range(len(self.learningRates)):
            
                data = self.epochData[currLR]

                xs.append(str(self.learningRates[currLR]))
                ys.append(data[-1][0])
            
            plt.title(f"{modelName} - Learning Rate Influence\n({len(self.epochData[0])} epochs)(batch size {self.batchSizes[0]})")
            plt.ylabel("Accuracy (%)")
            plt.xlabel("Learning Rate")
            
            
            plt.bar(xs, ys, color=color, width=width)
            
            plt.savefig(f"{modelName}_lr_barplot.png")


    #
    # Creates a bar graph showing the accuracy after training with
    # different Batch Sizes with a constant learning rate and number of epochs.
    #
    # !!! If multiple learning rates are used, this method will not do anything !!!
    #
    def saveBarGraph_BS(self, modelName, color='blue', width=0.3):
            # Accuracy vs Batch size
            
            if (len(self.learningRates) > 1):
                
                print("Batch size bar graph - skipping")
                return
                
            else:
            
                print("Batch size bar graph - generating")

            xs = []
            ys = []
            
            for currBS in range(len(self.batchSizes)):
            
                data = self.epochData[currBS]

                xs.append(str(self.batchSizes[currBS]))
                ys.append(data[-1][0])
            
            plt.title(f"{modelName} - Batch Size Influence\n({len(self.epochData[0])} epochs)(learning rate {self.learningRates[0]})")
            plt.ylabel("Accuracy (%)")
            plt.xlabel("Batch Size")
            
            
            plt.bar(xs, ys, color=color, width=width)
            
            plt.savefig(f"{modelName}_bs_barplot.png")
