import numpy as np
import gym
from lqr import controllers_ref as controllers
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

seed = 10703
np.random.seed(seed)

def plot_state_and_controls(states, us, goal, episode):
    labels = ['q[0]', 'q[1]', 'dq[0]', 'dq[1]', 'u[0]', 'u[1]']
    data = np.hstack((states, us))
    fig, axes = plt.subplots(6, 1)
    fig.set_size_inches(9, 5)
    for i, ax in enumerate(axes):
        ax.plot(data[:, i])
        ax.set_ylabel(labels[i])
        ax.set_yticks(np.linspace(start=min(data[:, i]), stop=max(data[:, i]), num=5))

    plt.suptitle("start: %s, goal: %s" % (
        np.array2string(states[0], precision=2),
        np.array2string(goal, precision=2),
    ))
    plt.savefig("lqr_output_%d.jpg" % episode)
    # plt.show()



def main():
    """
    Model Predictive Control using LQR with known dynamics model.
    """
    parser = argparse.ArgumentParser(description='Test LQR controller.')
    parser.add_argument(
        '--env',
        default='TwoLinkArm-random-goal-v0',
        type=str,
        help='Environment name')
    args = parser.parse_args()

    env = gym.make(args.env)
    sim_env = gym.make(args.env)
    av_ret = 0
    step_dt = 1e-3
    NUM_EPISODES = 5

    for episode in tqdm(range(NUM_EPISODES)):
        # ___ WRITE CODE HERE ___
        states = []
        us = []
        done = False
        state = env.reset()
        while not done:
            u = controllers.calc_lqr_input(env, sim_env)
            # TODO: ... = env.step(u, dt=step_dt)
            # TODO: Store x_t, u_t
            # TODO: update average return

        print('Success! Total reward: ', total_reward)

        # plot the trajectory
        states = np.array(states)
        us = np.array(us)
        plot_state_and_controls(states, us, goal=env.goal, episode=episode)

    # We expect around -61.378163357148736
    print('Average return: ', av_ret)


if __name__ == '__main__':
    main()

