import numpy as np 

class BinaryCrossEntropyLoss():
    def __init__(self):
        pass
    
    def compute(self, outputs=None, targets=None):
        self.outputs = outputs
        self.targets = targets
        # print("tyepe: ", (targets * np.log(outputs)))
        loss = -np.mean(targets * np.log(outputs) + (1-targets) * np.log(1-outputs))
        # print("loss type: ", type(loss))
        return loss

    def backward(self, inputs):
        grad = - np.sum((self.targets * (1 - self.outputs)).reshape(-1,1) * inputs + ((1 - self.targets) * ( (-1) *self.outputs)).reshape(-1,1) * inputs, axis=0)
        # grad = np.mean((self.targets - self.outputs).reshape(-1,1) * inputs, axis=0)
        # print("grad shape ", grad.shape)
        return grad
    