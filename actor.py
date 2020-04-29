import numpy as np

class Actor:
    def __init__(self, output_size):
        self.output_size = output_size
    
    # take action with epsilon-greedy
    def get_action(self, observation, episode, model):
        epsilon = 0.01 + 0.99 / (1.0+episode)
        
        if np.random.rand() > epsilon:
            action_probility = model.predict(observation)
#             print(action_probility)
            action = np.argmax(action_probility)

        else:
            action = np.random.choice(np.arange(self.output_size))
        
        return action