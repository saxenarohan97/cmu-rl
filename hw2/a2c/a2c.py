import sys
import argparse
import numpy as np

import torch
import time


def sample_action(env, probs):
    return np.random.choice(range(env.action_space.n), p=probs)


def compute_returns(gamma, rewards):
    gammas = torch.cuda.FloatTensor([gamma ** i for i in range(len(rewards))][::-1])
    rewards = rewards[::-1]
    rewards = torch.cuda.FloatTensor(rewards)
    returns = torch.cumsum(gammas * rewards, dim=0) / gammas
    returns = torch.flip(returns, dims=(0,))
    return returns


class Reinforce(object):

    def __init__(self, policy, lr):
        self.type = 'Reinforce'
        self.policy = policy
        self.policy_optim = torch.optim.Adam(policy.parameters(), lr)
    

    def evaluate_policy(self, env, render=False, delay=False):
        done = False
        state = env.reset()
        total_reward = 0.
        while not done:
            if render:
                env.render()
                if delay:
                    time.sleep(0.1)
            with torch.no_grad():
                prob = self.policy(torch.cuda.FloatTensor(state))
            action = sample_action(env, prob.cpu().detach().numpy())
            state, reward, done, info = env.step(action)
            total_reward += reward
        return total_reward


    def generate_episode(self, env, render=True):
        # Generates an episode by executing the current policy in the given env.
        # Returns:
        # - a list of states, indexed by time step
        # - a list of actions, indexed by time step
        # - a list of rewards, indexed by time step
        states = []
        actions = []
        rewards = []
        probs = []

        done = False
        state = env.reset()
        while not done:
            if render:
                env.render()
            states.append(state)
            prob = self.policy(torch.cuda.FloatTensor(state))
            probs.append(prob)
            action = sample_action(env, prob.cpu().detach().numpy())
            actions.append(action)
            state, reward, done, info = env.step(action)
            rewards.append(reward)
        
        return states, actions, rewards, torch.stack(probs)


    def train(self, env, gamma=0.99):
        self.policy_optim.zero_grad()
        states, actions, rewards, probs = self.generate_episode(env)
        returns = compute_returns(gamma, rewards)
        prob_actions = probs[range(len(probs)), actions]
        loss = - torch.mean(returns * torch.log(prob_actions))
        loss.backward()
        self.policy_optim.step()


class Baseline(Reinforce):
    
    def __init__(self, policy, lr, baseline, baseline_lr,):
        super(Baseline, self).__init__(policy, lr)
        self.type = 'Baseline'
        self.baseline = baseline
        self.baseline_optim = torch.optim.Adam(baseline.parameters(), baseline_lr)
    

    def train(self, env, gamma=0.99):
        self.policy_optim.zero_grad()
        self.baseline_optim.zero_grad()
        states, actions, rewards, probs = self.generate_episode(env)
        returns = compute_returns(gamma, rewards)
        prob_actions = probs[range(len(probs)), actions]
        baselines = self.baseline(torch.cuda.FloatTensor(states))
        policy_loss = - torch.mean((returns - baselines.detach()) * torch.log(prob_actions))
        baseline_loss = torch.mean((returns - baselines) ** 2.)
        policy_loss.backward()
        baseline_loss.backward()
        self.policy_optim.step()
        self.baseline_optim.step()


class A2C(Reinforce):
    # Implementation of N-step Advantage Actor Critic.
    # This class inherits the Reinforce class, so for example, you can reuse
    # generate_episode() here for different methods.

    def __init__(self, actor, actor_lr, N, critic, critic_lr):
        # Note: baseline is true if we use reinforce with baseline
        #       a2c is true if we use a2c else reinforce
        super(A2C, self).__init__(actor, actor_lr)
        self.type = 'A2C'  # Pick one of: "A2C", "Baseline", "Reinforce"
        self.N = N
        self.critic = critic
        self.critic_optim = torch.optim.Adam(critic.parameters(), critic_lr)


    def train(self, env, gamma=0.99, n=10):
        # Trains the model on a single episode using REINFORCE or A2C/A3C.
        # Implement this method. It may be helpful to call the class
        # method generate_episode() to generate training data.
        self.policy_optim.zero_grad()
        self.critic_optim.zero_grad()
        states, actions, rewards, probs = self.generate_episode(env)

        T = len(states)
        states = np.array(states)
        values = self.critic(torch.cuda.FloatTensor(states))

        if self.N >= T:
            v_ends = 0.
        else:
            next_states = np.arange(self.N, T)
            v_ends = values[next_states]
            v_ends = torch.squeeze(v_ends, dim=-1)
            v_ends = torch.cat([v_ends, torch.cuda.FloatTensor([0. for _ in range(self.N)])])
        
        gammas = np.array([gamma ** i for i in range(self.N)])
        returns = np.array([np.sum(gammas[:min(t + self.N, T) - t] * rewards[t : min(t + self.N, T)]) for t in range(T)])

        if type(v_ends) is float:
            v_ends_value = v_ends
        
        else:
            v_ends_value = v_ends.detach()
            returns = torch.cuda.FloatTensor(returns) + (gamma ** self.N) * v_ends_value
            critic_loss = torch.mean((torch.cuda.FloatTensor(returns) - values) ** 2.)
            critic_loss.backward()
            self.critic_optim.step()
        
        prob_actions = probs[range(len(probs)), actions]
        policy_loss = - torch.mean((torch.cuda.FloatTensor(returns) - values.detach()) * torch.log(prob_actions))
        policy_loss.backward()
        self.policy_optim.step()
