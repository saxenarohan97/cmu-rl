#!/usr/bin/env python
# coding: utf-8

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from builtins import input

import gym
import lake_envs
from pi_vi import *
import time


def run_random_policy(env):
    """Run a random policy for the given environment.

    Logs the total reward and the number of steps until the terminal
    state was reached.

    Parameters
    ----------
    env: gym.envs.Environment
      Instance of an OpenAI gym.

    Returns
    -------
    (float, int)
      First number is the total undiscounted reward received. The
      second number is the total number of actions taken before the
      episode finished.
    """
    initial_state = env.reset()
    env.render()
    time.sleep(1)  # just pauses so you can see the output

    total_reward = 0
    num_steps = 0
    while True:
        nextstate, reward, is_terminal, debug_info = env.step(
            env.action_space.sample())
        env.render()

        total_reward += reward
        num_steps += 1

        if is_terminal:
            break

        time.sleep(1)

    return total_reward, num_steps


def print_env_info(env):
    print('Environment has %d states and %d actions.' % (env.nS, env.nA))


def print_model_info(env, state, action):
    transition_table_row = env.P[state][action]
    print(
        ('According to transition function, '
         'taking action %s(%d) in state %d leads to'
         ' %d possible outcomes') % (lake_env.action_names[action],
                                     action, state, len(transition_table_row)))
    for prob, nextstate, reward, is_terminal in transition_table_row:
        state_type = 'terminal' if is_terminal else 'non-terminal'
        print(
            '\tTransitioning to %s state %d with probability %f and reward %f'
            % (state_type, nextstate, prob, reward))


def main():
    # create the environment
    env = gym.make('Deterministic-8x8-FrozenLake-v0')

    print_env_info(env)
    # print_model_info(env, 0, lake_env.DOWN)
    # print_model_info(env, 1, lake_env.DOWN)
    # print_model_info(env, 14, lake_env.RIGHT)
    
    improv_li = []
    value_li = []
    iters_li = []
    for i in range(10):
    
        # policy, value_func, num_improv_iter, total_value_iter = policy_iteration_sync(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        # policy, value_func, num_improv_iter, total_value_iter = policy_iteration_async_ordered(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        policy, value_func, num_improv_iter, total_value_iter = policy_iteration_async_randperm(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        
        improv_li.append(num_improv_iter)
        value_li.append(total_value_iter)
        
        # policy, value_func, num_iters = value_iteration_sync(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        # policy, value_func, num_iters = value_iteration_async_ordered(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        # policy, value_func, num_iters = value_iteration_async_randperm(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        # policy, value_func, num_iters = value_iteration_async_custom(env, gamma=0.9, max_iterations=int(1e3), tol=1e-3)
        # iters_li.append(num_iters)
        
    print(f'Improvement steps: {np.mean(improv_li)}.')
    print(f'Value steps: {np.mean(value_li)}.')
    
    # print(f'Number of iterations: {np.mean(iters_li)}.')
    
    
    value_func_heatmap(env, value_func)
    display_policy_letters(env, policy)

if __name__ == '__main__':
    main()
