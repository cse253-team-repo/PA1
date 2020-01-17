import numpy as np 

class CrossEntropyLoss():
    def __init__(self, outputs=None, targets=None):
        self.outputs = outputs
        self.targets = targets

        loss = np.mean(targets * np.log(outputs) + (1-targets) * np.log(1-outputs))
    
    def backward(self, x):
        self.grad = - np.sum((self.targets * (1 - self.outputs)).reshape(-1,1) * x + ((1 - self.targets) * (- self.outputs)).reshape(-1,1) * x, axis=0)

