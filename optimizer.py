import numpy as np 

class Optimizer():
    def __init__(self, learning_rate=0.001):
        self.learning_rate = learning_rate
    
    def update(self, model_para=None, grad=None):
        model_para -= self.learning_rate * grad
        return model_para

        
        