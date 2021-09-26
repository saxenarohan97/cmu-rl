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
    def __init__(self, initialization=0, n_init=0):
        self.estimated_avg = np.zeros(10) + initialization
        self.pull_n = np.zeros(10) + n_init
        
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
        
        expected_reward = eps * np.mean(bandit.avg_rew) \
                            + (1 - eps) * bandit.avg_rew[np.argmax(self.estimated_avg)]
        
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
        
        expected_reward = bandit.avg_rew[action]
        return expected_reward
    
    def ucb_run(self, bandit, c=1, iters=ITERS):
        expected_rewards = []
        
        for i in range(iters):
            expected_reward = self.ucb_step(bandit, c=c)
            expected_rewards.append(expected_reward)
            
        return expected_rewards
    
    def grad_step(self, bandit, temp, step=0.1):
        self.probs = np.exp(self.estimated_avg * temp) \
                        / np.sum(np.exp(self.estimated_avg * temp))
        action = np.random.choice(np.arange(0, 10), p=self.probs)
        rew = self.pull_and_update(action, bandit)
        
        baseline_rew = np.mean(self.estimated_avg)
        
        
        expected_reward = np.sum(bandit.avg_rew * self.probs)
        return expected_reward         
        
    def grad_run(self, bandit, temp, step=0.1, iters=ITERS):
        expected_rewards = []
        
        for i in range(iters):
            expected_reward = self.grad_step(bandit, temp, step=step)
            expected_rewards.append(expected_reward)
            
        return expected_rewards

bandit = Bandit()

# ====================================================
# EPSILON GREEDY



epss = [0, 0.001, 0.01, 0.1, 1]

for eps in epss:
    temp_for_avg = []
    for i in range(NUM_RUNS):
        agent = Agent()
        temp_for_avg.append(agent.eps_greedy_run(bandit, eps))
    
    expected_rewards = np.mean(temp_for_avg, axis=0)

    plt.plot([i for i in range(ITERS)], expected_rewards, label=f'eps={eps}')
plt.legend()
plt.title('Eps-greedy')
plt.xlabel('Iteration')
plt.ylabel('Expected reward')
plt.show()

# ====================================================
# OPTIMISTIC INITIALIZATION


initializations = [0, 1, 2, 5, 10]

for initialization in initializations:
    temp_for_avg = []
    for i in range(NUM_RUNS):
        agent = Agent(initialization=initialization, n_init=1)
        temp_for_avg.append(agent.eps_greedy_run(bandit, eps=0))
    expected_rewards = np.mean(temp_for_avg, axis=0)

    plt.plot([i for i in range(ITERS)], expected_rewards, label=f'Init={initialization}')
plt.legend()
plt.title('Optimistic')
plt.xlabel('Iteration')
plt.ylabel('Expected reward')
plt.show()

# ====================================================
# UCB


cs = [0, 1, 2, 5]

for c in cs:
    temp_for_avg = []
    for i in range(NUM_RUNS):
        agent = Agent()
        temp_for_avg.append(agent.ucb_run(bandit, c=c))
    expected_rewards = np.mean(temp_for_avg, axis=0)

    plt.plot([i for i in range(ITERS)], expected_rewards, label=f'c={c}')
plt.legend()
plt.title('UCB')
plt.xlabel('Iteration')
plt.ylabel('Expected reward')
plt.show()

# ====================================================
# GRAD

temps = [1, 3, 10, 30, 100]

for temp in temps:
    temp_for_avg = []
    for i in range(NUM_RUNS):
        agent = Agent()
        temp_for_avg.append(agent.grad_run(bandit, temp=temp))
    expected_rewards = np.mean(temp_for_avg, axis=0)
    plt.plot([i for i in range(ITERS)], expected_rewards, label=f'temp={temp}')
plt.legend()
plt.title('Boltzmann')
plt.xlabel('Iteration')
plt.ylabel('Expected reward')
plt.show()

# ====================================================
# BEST

temp_for_avg=[]
eps = 0.1
for i in range(NUM_RUNS):
    agent = Agent()
    temp_for_avg.append(agent.eps_greedy_run(bandit, eps))

expected_rewards = np.mean(temp_for_avg, axis=0)

plt.plot([i for i in range(ITERS)], expected_rewards,
         label=f'Eps-greedy with eps={eps}')

temp_for_avg=[]
initialization = 10
for i in range(NUM_RUNS):
    agent = Agent(initialization=initialization, n_init=1)
    temp_for_avg.append(agent.eps_greedy_run(bandit, eps=0))
expected_rewards = np.mean(temp_for_avg, axis=0)
plt.plot([i for i in range(ITERS)], expected_rewards,
         label=f'Optimistic initialization with {initialization}')

temp_for_avg=[]
c = 2
for i in range(NUM_RUNS):
    agent = Agent()
    temp_for_avg.append(agent.ucb_run(bandit, c=c))
expected_rewards = np.mean(temp_for_avg, axis=0)
plt.plot([i for i in range(ITERS)], expected_rewards,
         label=f'UCB with c={c}')

temp_for_avg=[]
temp = 3
for i in range(NUM_RUNS):
    agent = Agent()
    temp_for_avg.append(agent.grad_run(bandit, temp=temp))
expected_rewards = np.mean(temp_for_avg, axis=0)
plt.plot([i for i in range(ITERS)], expected_rewards,
         label=f'Boltzmann with temp={temp}')

plt.legend()
plt.title('Best')
plt.xlabel('Iteration')
plt.ylabel('Expected reward')
plt.show()