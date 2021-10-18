#!/usr/bin/env python
import numpy as np, gym, sys, copy, argparse
import os
import torch
import collections
import tqdm
import matplotlib.pyplot as plt
import random

def compute_returns(gamma, rewards):
    gammas = torch.cuda.FloatTensor([gamma ** i for i in range(len(rewards))][::-1])
    rewards = rewards[::-1]
    rewards = torch.cuda.FloatTensor(rewards)
    returns = torch.cumsum(gammas * rewards, dim=0) / gammas
    returns = torch.flip(returns, dims=(0,))
    return returns

def sample_action(env, probs):
    return np.random.choice(range(env.action_space.n), p=probs)

class FullyConnectedModel(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 16)
        self.activation1 = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(16, 16)
        self.activation2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(16, 16)
        self.activation3 = torch.nn.ReLU()

        self.output_layer = torch.nn.Linear(16, output_size)
        #no activation output layer

        #initialization
        torch.nn.init.xavier_uniform_(self.linear1.weight)
        torch.nn.init.xavier_uniform_(self.linear2.weight)
        torch.nn.init.xavier_uniform_(self.linear3.weight)
        torch.nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, inputs):
        x = self.activation1(self.linear1(inputs))
        x = self.activation2(self.linear2(x))
        x = self.activation3(self.linear3(x))
        x = self.output_layer(x)
        return x


class QNetwork():

    # This class essentially defines the network architecture.
    # The network should take in state of the world as an input,
    # and output Q values of the actions available to the agent as the output.

    def __init__(self, env, lr=5e-4, logdir=None):
    # Define your network architecture here. It is also a good idea to define any training operations
    # and optimizers here, initialize your variables, or alternately compile your model here.
        self.env = env
        self.nS = 4
        self.model = FullyConnectedModel(self.nS + 1, 1).to('cuda:0')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr)
        self.logdir = logdir
        self.gamma = 0.99

    def save_model_weights(self, suffix):
    # Helper function to save your model / weights.
        path = os.path.join(self.logdir, "model")
        torch.save(self.model.state_dict(), path)
        return path

    def load_model(self, model_file):
    # Helper function to load an existing model.
        return self.model.load_state_dict(torch.load(model_file))

    def load_model_weights(self, weight_file):
    # Optional Helper function to load model weights.
        pass
    
    def copy_weights(self, q_net):
        self.model.load_state_dict(q_net.model.state_dict())
    
    
    def predict(self, inp):
        return torch.squeeze(self.model.forward(torch.cuda.FloatTensor(inp)))
    
    def q_values(self, state):
        return self.predict([state.tolist() + [0], state.tolist() + [1]])
    
    def train(self, batch, model, target_model):
        self.optimizer.zero_grad()
        loss = torch.cuda.FloatTensor([0.0])
        for sample in batch:
            
            state, action, reward, new_state, done = sample
            target_q_vals = target_model.q_values(new_state)
            q_val = model.q_values(state)[action]
            if done:
                y = reward
            else:
                y = reward + self.gamma * torch.max(target_q_vals)
            
            loss += (y - q_val) ** 2
        loss /= len(batch)
        loss.backward()
        self.optimizer.step()


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):

    # The memory essentially stores transitions recorder from the agent
    # taking actions in the environment.

    # Burn in episodes define the number of episodes that are written into the memory from the
    # randomly initialized agent. Memory size is the maximum size after which old elements in the memory are replaced.
    # A simple (if not the most efficient) was to implement the memory is as a list of transitions.

    # Hint: you might find this useful:
    # 		collections.deque(maxlen=memory_size)
        self.memory = collections.deque(maxlen=memory_size)
        self.burn_in = burn_in

    def sample_batch(self, batch_size=32):
    # This function returns a batch of randomly sampled transitions - i.e. state, action, reward, next state, terminal flag tuples.
    # You will feed this to your model to train.
        batch = random.sample(self.memory, batch_size)
        return batch

    def append(self, transition):
    # Appends transition to the memory.
        self.memory.append(transition)

def random_action():
    return np.random.randint(0,2)

class DQN_Agent():

    # In this class, we will implement functions to do the following.
    # (1) Create an instance of the Q Network class.
    # (2) Create a function that constructs a policy from the Q values predicted by the Q Network.
    #		(a) Epsilon Greedy Policy.
    # 		(b) Greedy Policy.
    # (3) Create a function to train the Q Network, by interacting with the environment.
    # (4) Create a function to test the Q Network's performance on the environment.
    # (5) Create a function for Experience Replay.

    def __init__(self, environment_name, render=False):

        # Create an instance of the network itself, as well as the memory.
        # Here is also a good place to set environmental parameters,
        # as well as training parameters - number of episodes / iterations, etc.
        self.env = gym.make(environment_name)
        self.render = render
        self.eps = 0.05
        self.model = QNetwork(self.env)
        self.target_model = QNetwork(self.env)
        self.memory = Replay_Memory()
        self.epochs = 200
        self.test_hist = []
        
        
    def epsilon_greedy_policy(self, q_values):
        # Creating epsilon greedy probabilities to sample from.
        if np.random.uniform() < self.eps:
            return random_action()
        return np.argmax(q_values.detach().cpu().numpy())

    def greedy_policy(self, q_values):
        # Creating greedy policy for test time.
        return np.argmax(q_values.detach().cpu().numpy())

    def train(self):
        # In this function, we will train our network.

        # When use replay memory, you should interact with environment here, and store these
        # transitions to memory, while also updating your model.
        self.burn_in_memory()
        c = 0
        for epoch in tqdm.tqdm(range(self.epochs)):
            if epoch % 10 == 0:
                self.test()
            done = False
            state = self.env.reset()
            while not done:
                
                if c % 50 == 0:
                    self.target_model.copy_weights(self.model)
                
                q_values = self.model.q_values(state)
                action = self.epsilon_greedy_policy(q_values)
                new_state, reward, done, info = self.env.step(action)
                
                self.memory.append((state, action, reward, new_state, done))
                state = new_state
                batch = self.memory.sample_batch()
                self.model.train(batch, self.model, self.target_model)
                
                c += 1

    def test(self, model_file=None):
        # Evaluate the performance of your agent over 100 episodes, by calculating average cummulative rewards (returns) for the 100 episodes.
        # Here you need to interact with the environment, irrespective of whether you are using a memory.
        rewards = []
        for epoch in range(20):
            state = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                q_values = self.model.q_values(state).detach()
                action = self.greedy_policy(q_values)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
            rewards.append(episode_reward)
        self.test_hist.append(np.mean(rewards))
                

    def burn_in_memory(self):
        # Initialize your replay memory with a burn_in number of episodes / transitions.
        while len(self.memory.memory) <= self.memory.burn_in:
            state = self.env.reset()
            done = False
            while not done:
                action = random_action()
                new_state, reward, done, info = self.env.step(action)
                self.memory.append((state, action, reward, new_state, done))
                state = new_state


# Note: if you have problems creating video captures on servers without GUI,
#       you could save and relaod model to create videos on your laptop.
def test_video(agent, env, epi):
    # Usage:
    # 	you can pass the arguments within agent.train() as:
    # 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network Argument Parser')
    parser.add_argument('--env',dest='env',type=str,default='CartPole-v0')
    parser.add_argument('--render',dest='render',type=int,default=0)
    parser.add_argument('--train',dest='train',type=int,default=1)
    parser.add_argument('--model',dest='model_file',type=str)
    # parser.add_argument('--lr', dest='lr', type=float, default=5e-4)
    return parser.parse_args()


def main(args):

    args = parse_arguments()
    environment_name = args.env
    
    avg_test_hist = []
    
    for _ in range(5):
        agent = DQN_Agent(environment_name)
        agent.train()
    
        avg_test_hist.append(agent.test_hist)
        
        print(agent.test_hist)
        
    plt.plot([i * 10 for i in range(20)], np.stack(avg_test_hist).mean(axis=0))
    plt.xlabel('epoch')
    plt.ylabel('reward') 
    plt.title('DQN')
    plt.savefig("./dqn.png")
    
if __name__ == '__main__':
    main(sys.argv)
