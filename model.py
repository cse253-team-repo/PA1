import numpy as np 

class LR_Classifier():
    def __init__(self, inputs_dim=None):
        self.Linear1 = np.random.normal(size=inputs_dim+1)

    def forward(self, x):
        print("x shape: ", x.shape)
        batch_size = x.shape[0]
        bias = np.ones(batch_size).reshape(-1,1)
        print("bias shape: ", bias.shape)
        x = np.concatenate((x,bias), axis=1)
        print("x shape:, ", x.shape)
        print("fc shape: ", self.Linear1.shape)
        x = np.dot(x,self.Linear1)
        outputs = 1/(1+np.exp(x))
        predictions = outputs > 0.5

        return outputs 
    
    def backward(self, x, outputs, targets):
        self.grad = - np.sum(((targets * (1 - outputs)).reshape(-1,1) * x + ((1 - targets) * (- outputs)).reshape(-1,1) * x), axis=1)

    def update(self, )


    