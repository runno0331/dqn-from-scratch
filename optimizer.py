import numpy as np

#################################################################################
# optimizer
#################################################################################

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    
    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
            
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.t = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m = []
            self.v = []
            for i in range(len(params)):
                self.m.append(np.zeros_like(params[i]))
                self.v.append(np.zeros_like(params[i]))
        
        self.t += 1
        lr = self.lr * np.sqrt(1.0 - self.beta2**self.t) / (1.0 - self.beta1**self.t)
        for i in range(len(params)):       
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * grads[i] * grads[i]
            params[i] -= lr * self.m[i] / np.sqrt(self.v[i] + 1e-8)