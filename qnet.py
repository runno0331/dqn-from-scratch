from layer import *

class QNetwork:
    def __init__(self, input_size, output_size, hidden_size, optimizer):
        self.input_size = input_size
        self.output_size = output_size
        self.optimizer = optimizer
        self.layers = [
            Dense(input_size, hidden_size),
            ReLU(),
            Dense(hidden_size, hidden_size),
            ReLU(),
            Dense(hidden_size, output_size)
        ]
        self.loss_func = HuberLoss()
        
        self.params = []
        self.grads = []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, dout=1.0):
        dout = self.loss_func.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def loss(self, x, t):
        y = self.predict(x)
        loss = self.loss_func.forward(y, t)
        if loss is None:
            print(y)
            sys.exit("loss is None")
        return loss
    
    def train(self, batch_size, gamma, memory, targetQNet):
        inputs = np.zeros((batch_size, self.input_size))
        targets = np.zeros((batch_size, self.output_size))
        mini_batch = memory.sample(batch_size)
        
        # reshape memory data for trainng
        for i, (observation, action, reward, next_observation) in enumerate(mini_batch):
            inputs[i:i+1] = observation
            target = reward
            
            # not terminal
            if not (next_observation == np.zeros(observation.shape)).all():
                next_action_probiility = self.predict(next_observation)
                next_action = np.argmax(next_action_probiility)
                target += gamma * targetQNet.predict(next_observation)[0][next_action] 
            targets[i] = self.predict(observation)
            targets[i][action] = target

        # training network
        loss = self.loss(inputs, targets)
        self.backward()
        self.optimizer.update(self.params, self.grads)
        return loss