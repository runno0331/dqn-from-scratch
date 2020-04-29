from actor import Actor
from buffer import ReplayBuffer
from optimizer import Adam
from qnet import QNetwork

import numpy as np
import sys
import time
import gym
import matplotlib.pyplot as plt
from collections import deque

# hyper parameters
env = gym.make('CartPole-v0')
max_episodes = 1000
max_steps = 200
gamma = 0.99
np.random.seed(1)

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
hidden_size = 10
memory_size = 10000
batch_size = 32
learning_rate = 2e-3
judge_range = 20

# for recording
learned_judge = deque(maxlen=judge_range)
score_record = []
loss_record = []

mainQNet = QNetwork(input_size=input_size, output_size=output_size, hidden_size=hidden_size, optimizer=Adam(lr=learning_rate))
targetQNet = QNetwork(input_size=input_size, output_size=output_size, hidden_size=hidden_size, optimizer=Adam(lr=learning_rate))
# targetQNet = mainQNet # comment if ddqn
memory = ReplayBuffer(maxlen=memory_size)
actor = Actor(output_size)

for episode in range(max_episodes):
    # initialize environment
    observation = env.reset()
    observation = np.reshape(observation, (1, input_size))
    score = 0
    loss = []
    
    for step in range(max_steps+1):
        # transition
        action = actor.get_action(observation, episode, mainQNet)
        next_observation, reward, done, _ = env.step(action)
        next_observation = np.reshape(next_observation, (1, input_size))
        
        # if terminal
        if done:
            next_observation = np.zeros_like(observation)
            if step < 195: # failure
                reward = -1
            else: #success
                reward = 1 
            memory.add((observation, action, reward, next_observation))
            break
        else:
            reward = 0
            
        score += 1
            
        memory.add((observation, action, reward, next_observation))
        observation = next_observation
        
        if memory.length() > batch_size:
            loss_value = mainQNet.train(batch_size, gamma, memory, targetQNet)
            loss.append(loss_value)
    
    # record
    score_record.append(score)
    learned_judge.append(score)
    if len(loss) == 0:
        loss.append(0)
    loss_record.append(np.mean(loss))
    if episode % 10 == 0:
        print("{}: score={}".format(episode, score))
        
    if np.mean(learned_judge) > 195:
        print("Learning Completed in {} episodes!!".format(episode+1))
        break
        
    targetQNet, mainQNet = mainQNet, targetQNet

# show record graph
plt.plot(score_record)
plt.show()
plt.plot(loss_record)
plt.show()

# render learned result
try:
    actor = Actor(output_size)
    observation = env.reset()
    for i in range(200):
        time.sleep(0.01)
        
        env.render()
        
        action = actor.get_action(observation, 100000000, mainQNet)
        next_observation, reward, done, _ = env.step(action)
        if done:
            break
        
        observation = next_observation
        
finally:
    env.close()