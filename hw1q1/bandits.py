# -*- coding: utf-8 -*-
"""
Created on Sun Sep 19 11:22:10 2021

@author: Dergel
"""

import numpy as np
import matplotlib.pyplot as plt

NUM_RUNS = 20
ITERS = 1000

class Bandit:
    def __init__(self):
        self.avg_rew = np.random.normal(1,1,10)       
    
    def pull_arm(self, arm_num):
        return np.random.normal(self.avg_rew[arm_num])
    

class Agent:
    def __init__(self, initialization=0):
        self.estimated_avg = np.zeros(10) + initialization
        self.pull_n = np.zeros(10)
        
        self.action_pref = np.zeros(10)
    
    def pull_and_update(self, arm_num, bandit):
        rew = bandit.pull_arm(arm_num)
        self.pull_n[arm_num] += 1
        self.estimated_avg[arm_num] += (rew - self.estimated_avg[arm_num]) \
                                            / self.pull_n[arm_num]
       
        return rew
    
    def eps_greedy_step(self, bandit, eps):
        random_prob = np.random.uniform()
        if random_prob < eps:
            action = np.random.randint(10)
        else:
            action = np.argmax(self.estimated_avg)
        
        self.pull_and_update(action, bandit)
        
        expected_reward = eps / 10 * np.sum(self.estimated_avg) \
                            + (1 - eps) * np.max(self.estimated_avg)
        
        return expected_reward
    
    def eps_greedy_run(self, bandit, eps=0.1, iters=ITERS):
        expected_rewards = []
        
        for i in range(iters):
            expected_reward = self.eps_greedy_step(bandit, eps)
            expected_rewards.append(expected_reward)
            
        return expected_rewards
    
    
    def ucb_step(self, bandit, c):
        
        if 0 in self.pull_n:
            action = np.where(self.pull_n == 0)[0][0]
        else:
            action = np.argmax(self.estimated_avg 
                               + c * (np.log(np.sum(self.pull_n)) 
                                           / self.pull_n)**0.5)
        self.pull_and_update(action, bandit)
        
        expected_reward = self.estimated_avg[action]
        
        return expected_reward
    
    def ucb_run(self, bandit, c=1, iters=ITERS):
        expected_rewards = []
        
        for i in range(iters):
            expected_reward = self.ucb_step(bandit, c=c)
            expected_rewards.append(expected_reward)
            
        return expected_rewards
    
    def grad_step(self, bandit, step=1):
        action = np.argmax(self.action_pref)
        rew = self.pull_and_update(action, bandit)
        
        baseline_rew = np.mean(self.estimated_avg)
        
        self.probs = np.exp(self.action_pref) / np.sum(np.exp(self.action_pref))
        self.action_pref += step * (rew - baseline_rew) \
                            * ((np.array([i for i in range(10)]) == action) 
                               - self.probs)
        expected_reward = np.sum(self.estimated_avg * self.probs)
        return expected_reward         
        
    def grad_run(self, bandit, step=1, iters=ITERS):
        expected_rewards = []
        
        for i in range(iters):
            expected_reward = self.grad_step(bandit, step=step)
            expected_rewards.append(expected_reward)
            
        return expected_rewards

bandit = Bandit()


# ====================================================
# EPSILON GREEDY

temp_for_avg = []

eps = 0.1

for i in range(NUM_RUNS):
    agent = Agent()
    temp_for_avg.append(agent.eps_greedy_run(bandit, eps))

expected_rewards = np.mean(temp_for_avg, axis=0)

plt.plot([i for i in range(ITERS)], expected_rewards)
plt.title(f'Eps-greedy with {eps}')
plt.xlabel('Iteration')
plt.ylabel('Expected reward')
plt.show()

# ====================================================
# OPTIMISTIC INITIALIZATION

initialization = 5

for i in range(NUM_RUNS):
    agent = Agent(initialization=initialization)
    temp_for_avg.append(agent.eps_greedy_run(bandit, eps=0))

expected_rewards = np.mean(temp_for_avg, axis=0)

plt.plot([i for i in range(ITERS)], expected_rewards)
plt.title(f'Optimistic with {initialization}')
plt.xlabel('Iteration')
plt.ylabel('Expected reward')
plt.show()

# ====================================================
# UCB

c = 2

for i in range(NUM_RUNS):
    agent = Agent()
    temp_for_avg.append(agent.ucb_run(bandit, c=c))

expected_rewards = np.mean(temp_for_avg, axis=0)

plt.plot([i for i in range(ITERS)], expected_rewards)
plt.title(f'UCB with {c}')
plt.xlabel('Iteration')
plt.ylabel('Expected reward')
plt.show()

# ====================================================
# GRAD

step = 1

for i in range(NUM_RUNS):
    agent = Agent()
    temp_for_avg.append(agent.grad_run(bandit, step=step))

expected_rewards = np.mean(temp_for_avg, axis=0)

plt.plot([i for i in range(ITERS)], expected_rewards)
plt.title(f'Grad with {step}')
plt.xlabel('Iteration')
plt.ylabel('Expected reward')
plt.show()