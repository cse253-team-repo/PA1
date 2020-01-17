import numpy as np 

class BinaryCrossEntropyLoss():
    def __init__(self, outputs=None, targets=None):
        self.outputs = outputs
        self.targets = targets
        loss = np.mean(targets * np.log(outputs) + (1-targets) * np.log(1-outputs))
        return loss[0]
    
    def backward(self, x):
        grad = - np.sum((self.targets * (1 - self.outputs)).reshape(-1,1) * x + ((1 - self.targets) * ( (-1) *self.outputs)).reshape(-1,1) * x, axis=0)
        
        return grad
    