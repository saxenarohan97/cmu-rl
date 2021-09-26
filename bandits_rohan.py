import numpy as np
from matplotlib import pyplot as plt


class Bandit:

    def __init__(self):
        self.mean_reward = np.random.normal(1., 1.)
    
    def pull_arm(self):
        return np.random.normal(self.mean_reward, 1.)


class Testbed:

    def __init__(self, arms, q_initialization=0.):
        self.arms = arms
        self.n = np.zeros(arms)
        self.q = np.zeros(arms) + q_initialization
        self.bandits = []
        for i in range(arms):
            self.bandits.append(Bandit())
        self.mean_rewards = [bandit.mean_reward for bandit in self.bandits]
        
    def pull_and_update(self, action, epsilon):
        reward = self.bandits[action].pull_arm()
        self.n[action] += 1
        self.q[action] += (reward - self.q[action]) / self.n[action]
        avg_reward = epsilon / self.arms * np.sum(self.mean_rewards) \
                        + (1. - epsilon) * self.mean_rewards[np.argmax(self.q)]
        return avg_reward


def epsilon_greedy(epsilon, arms=10, max_iters=1000, total_runs=20, q_initialization=0.):

    all_rewards = []

    for run in range(total_runs):
        test = Testbed(arms, q_initialization)
        run_rewards = []

        for iter in range(max_iters):
            action = np.random.choice([np.argmax(test.q), np.random.randint(0, test.arms)],
                                      p=[1 - epsilon, epsilon])

            avg_reward = test.pull_and_update(action, epsilon)
            run_rewards.append(avg_reward)
        
        all_rewards.append(run_rewards)

    return np.mean(all_rewards, axis=0)


def ucb(c, arms=10, max_iters=1000, total_runs=20):

    all_rewards = []

    for run in range(total_runs):
        test = Testbed(arms)
        run_rewards = []

        for iter in range(max_iters):
            if 0 in test.n:
                found = False
                while not found:
                    action = np.random.randint(0, arms)
                    if test.n[action] == 0:
                        found = True
            else:
                t = np.sum(test.n)
                action = np.argmax(test.q + c * np.sqrt(np.log(t) / test.n))
            avg_reward = test.pull_and_update(action, 0.)
            run_rewards.append(avg_reward)
        
        all_rewards.append(run_rewards)
    
    return np.mean(all_rewards, axis=0)


def q1():

    epsilons = [0., 0.001, 0.01, 0.1, 1.]

    for epsilon in epsilons:
        rewards = epsilon_greedy(epsilon)
        plt.plot(list(range(1, 1001)), rewards, label='epsilon = {}'.format(epsilon))
    
    plt.legend()
    plt.show()


def q2():

    initializations = [0., 1., 2., 5., 10.]

    for initialization in initializations:
        rewards = epsilon_greedy(0., q_initialization=initialization)
        plt.plot(list(range(1, 1001)), rewards, label='init = {}'.format(initialization))

    plt.legend()
    plt.show()


def q3():

    cs = [0., 1., 2., 5.]

    for c in cs:
        rewards = ucb(c)
        plt.plot(list(range(1, 1001)), rewards, label='c = {}'.format(c))
    
    plt.legend()
    plt.show()


if __name__ == '__main__':
    q3()