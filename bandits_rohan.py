import numpy as np
from matplotlib import pyplot as plt


class Bandit:

    def __init__(self):
        self.mean_reward = np.random.normal(1., 1.)
    
    def pull_arm(self):
        return np.random.normal(self.mean_reward, 1.)


class Testbed:

    def __init__(self, arms, q_initialization=None):
        self.arms = arms
        self.n = np.zeros(arms)
        if q_initialization is None:
            self.q = np.zeros(arms)
        else:
            self.q = np.array(q_initialization)
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


def epsilon_greedy(epsilon, arms=10, max_iters=1000, total_runs=20):

    test = Testbed(arms)
    all_rewards = []

    for run in range(total_runs):
        run_rewards = []

        for iter in range(max_iters):
            # action = np.random.choice([np.argmax(test.q), np.random.randint(0, test.arms)],
            #                           p=[1 - epsilon, epsilon])

            prob = np.random.uniform()
            if prob < epsilon:
                action = np.random.randint(10)
            else:
                action = np.argmax(test.q)

            avg_reward = test.pull_and_update(action, epsilon)
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


if __name__ == '__main__':
    q1()