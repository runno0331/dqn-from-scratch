import numpy as np
import sys

class Dense:
    def __init__(self, input_size, output_size, weight_std=0.01):
        W = weight_std * np.random.randn(input_size, output_size)
        b = np.zeros((1, output_size))
        
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
    def forward(self, x):
        W, b = self.params
        self.x = x
        return np.dot(x, W) + b

    # input_shape : (batch_size, forward_output_size)
    # output_shape : (batch_size, forward_input_size)
    def backward(self, dout):
        W, b = self.params
        # print("W: {}, b: {}, x: {}, dout: {}".format(W.shape, b.shape, self.x.shape, dout.shape))
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        dx = np.dot(dout, W.T)
        
        return dx

##########################################################
# activation layer
##########################################################

class ReLU:
    def __init__(self):
        self.params = []
        self.grads = []
        self.mask = None
        
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class HuberLoss:
    def __init__(self, delta=0.5):
        self.delta = delta
        self.params = []
        self.grads = []
        self.y = None
        self.err = None
        self.mask = None
        
    def forward(self, y, t):
        self.y = y
        err = y - t
        self.err =  err
        self.mask = (abs(err) < self.delta)
        loss = np.where(self.mask, 0.5*err*err, self.delta*abs(err) - 0.5*self.delta*self.delta)
        
        batch_size = err.shape[0]
        loss /= batch_size
        loss = np.sum(loss)
        if loss is None:
            print(err)
            sys.exit()
        return loss 
        
    def backward(self, dout=1.0):
        batch_size = self.err.shape[0]
        
        dx = self.err.copy()
        for i in range(len(dx)):
            for j in range(len(dx[0])):
                if not self.mask[i][j]:
                    if dx[i][j] > 0:
                        dx[i][j] = self.delta
                    else:
                        dx[i][j] = -self.delta
        
        dx *= dout
        dx /= batch_size
        return dx