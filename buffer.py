import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, maxlen=10000):
        self.buffer = deque(maxlen=maxlen)
        
    def add(self, data):
        self.buffer.append(data)
        
    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def length(self):
        return len(self.buffer)