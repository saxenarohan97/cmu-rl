# coding: utf-8
from __future__ import division, absolute_import
from __future__ import print_function, unicode_literals

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import gym
try:
	import hw1q1.lake_envs as lake_env
except:
	import lake_envs as lake_env


def print_policy(policy, action_names):
		"""Print the policy in human-readable format.

		Parameters
		----------
		policy: np.ndarray
			Array of state to action number mappings
		action_names: dict
			Mapping of action numbers to characters representing the action.
		"""
		str_policy = policy.astype('str')
		for action_num, action_name in action_names.items():
				np.place(str_policy, policy == action_num, action_name)

		print(str_policy)


def value_function_to_policy(env, gamma, value_function):
		"""Output action numbers for each state in value_function.

		Parameters
		----------
		env: gym.core.Environment
			Environment to compute policy for. Must have nS, nA, and P as
			attributes.
		gamma: float
			Discount factor. Number in range [0, 1)
		value_function: np.ndarray
			Value of each state.

		Returns
		-------
		np.ndarray
			An array of integers. Each integer is the optimal action to take
			in that state according to the environment dynamics and the
			given value function.
		"""
		# Hint: You might want to first calculate Q value,
		#       and then take the argmax.
		return policy


def evaluate_policy_sync(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
		"""Performs policy evaluation.

		Evaluates the value of a given policy.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		value_func: np.array
			The current value functione estimate
		gamma: float
			Discount factor, must be in range [0, 1)
		policy: np.array
			The policy to evaluate. Maps states to actions.
		max_iterations: int
			The maximum number of iterations to run before stopping.
		tol: float
			Determines when value function has converged.

		Returns
		-------
		np.ndarray, int
			The value for the given policy and the number of iterations till
			the value function converged.
		"""

		num_iterations = 0

		while num_iterations < max_iterations:
			delta = 0
			updated_value = np.copy(value_func)

			for state in range(env.nS):
					action = policy[state]
					prob, nextstate, reward, is_terminal = env.P[state][action][0]
					updated_value[state] = reward + gamma * value_func[nextstate]
					delta = max(delta, abs(value_func[state] - updated_value[state]))
			
			value_func = updated_value
			num_iterations += 1

			if delta < tol:
					break

		return value_func, num_iterations


def evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
		"""Performs policy evaluation.

		Evaluates the value of a given policy by asynchronous DP.  Updates states in
		their 1-N order.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		value_func: np.array
			The current value functione estimate
		gamma: float
			Discount factor, must be in range [0, 1)
		policy: np.array
			The policy to evaluate. Maps states to actions.
		max_iterations: int
			The maximum number of iterations to run before stopping.
		tol: float
			Determines when value function has converged.

		Returns
		-------
		np.ndarray, int
			The value for the given policy and the number of iterations till
			the value function converged.
		"""

		num_iterations = 0
		value_func = np.copy(value_func)

		while num_iterations < max_iterations:
			delta = 0

			for state in range(env.nS):
				action = policy[state]
				prob, nextstate, reward, is_terminal = env.P[state][action][0]
				old_value = value_func[state]
				value_func[state] = reward + gamma * value_func[nextstate]
				delta = max(delta, abs(value_func[state] - old_value))

			num_iterations += 1

			if delta < tol:
				break

		return value_func, num_iterations


def evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iterations=int(1e3), tol=1e-3):
		"""Performs policy evaluation.

		Evaluates the value of a policy.  Updates states by randomly sampling index
		order permutations.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		value_func: np.array
			The current value functione estimate
		gamma: float
			Discount factor, must be in range [0, 1)
		policy: np.array
			The policy to evaluate. Maps states to actions.
		max_iterations: int
			The maximum number of iterations to run before stopping.
		tol: float
			Determines when value function has converged.

		Returns
		-------
		np.ndarray, int
			The value for the given policy and the number of iterations till
			the value function converged.
		"""

		num_iterations = 0
		value_func = np.copy(value_func)

		while num_iterations < max_iterations:
			delta = 0

			for state in np.random.permutation(range(env.nS)):
				action = policy[state]
				prob, nextstate, reward, is_terminal = env.P[state][action][0]
				old_value = value_func[state]
				value_func[state] = reward + gamma * value_func[nextstate]
				delta = max(delta, abs(value_func[state] - old_value))

			num_iterations += 1

			if delta < tol:
				break

		return value_func, num_iterations


def improve_policy(env, gamma, value_func, policy):
		"""Performs policy improvement.

		Given a policy and value function, improves the policy.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		gamma: float
			Discount factor, must be in range [0, 1)
		value_func: np.ndarray
			Value function for the given policy.
		policy: dict or np.array
			The policy to improve. Maps states to actions.

		Returns
		-------
		bool, np.ndarray
			Returns true if policy changed. Also returns the new policy.
		"""

		policy_stable = True
		policy = np.copy(policy)

		for state in range(env.nS):
			old_action = policy[state]
			action_values = []
			for action in range(env.nA):
				prob, nextstate, reward, is_terminal = env.P[state][action][0]
				action_values.append(reward + gamma * value_func[nextstate])
			
			policy[state] = np.argmax(action_values)
			
			if old_action != policy[state]:
				policy_stable = False

		return policy_stable, policy


def policy_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
		"""Runs policy iteration.

		See page 85 of the Sutton & Barto Second Edition book.

		You should use the improve_policy() and evaluate_policy_sync() methods to
		implement this method.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		gamma: float
			Discount factor, must be in range [0, 1)
		max_iterations: int
			The maximum number of iterations to run before stopping.
		tol: float
			Determines when value function has converged.

		Returns
		-------
		(np.ndarray, np.ndarray, int, int)
			 Returns optimal policy, value function, number of policy
			 improvement iterations, and number of value iterations.
		"""
		policy = np.zeros(env.nS, dtype='int')
		value_func = np.zeros(env.nS)

		policy_stable = False
		num_improv_iter = 0
		total_value_iter = 0

		while not policy_stable:
			value_func, num_eval_iter = evaluate_policy_sync(env, value_func, gamma, policy, max_iterations, tol)
			total_value_iter += num_eval_iter
			policy_stable, policy = improve_policy(env, gamma, value_func, policy)
			num_improv_iter += 1

		return policy, value_func, num_improv_iter, total_value_iter


def policy_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
		"""Runs policy iteration.

		You should use the improve_policy and evaluate_policy_async_ordered methods
		to implement this method.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		gamma: float
			Discount factor, must be in range [0, 1)
		max_iterations: int
			The maximum number of iterations to run before stopping.
		tol: float
			Determines when value function has converged.

		Returns
		-------
		(np.ndarray, np.ndarray, int, int)
			 Returns optimal policy, value function, number of policy
			 improvement iterations, and number of value iterations.
		"""
		policy = np.zeros(env.nS, dtype='int')
		value_func = np.zeros(env.nS)

		policy_stable = False
		num_improv_iter = 0
		total_value_iter = 0

		while not policy_stable:
			value_func, num_eval_iter = evaluate_policy_async_ordered(env, value_func, gamma, policy, max_iterations, tol)
			total_value_iter += num_eval_iter
			policy_stable, policy = improve_policy(env, gamma, value_func, policy)
			num_improv_iter += 1

		return policy, value_func, num_improv_iter, total_value_iter


def policy_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
																		tol=1e-3):
		"""Runs policy iteration.

		You should use the improve_policy and evaluate_policy_async_randperm methods
		to implement this method.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		gamma: float
			Discount factor, must be in range [0, 1)
		max_iterations: int
			The maximum number of iterations to run before stopping.
		tol: float
			Determines when value function has converged.

		Returns
		-------
		(np.ndarray, np.ndarray, int, int)
			 Returns optimal policy, value function, number of policy
			 improvement iterations, and number of value iterations.
		"""
		policy = np.zeros(env.nS, dtype='int')
		value_func = np.zeros(env.nS)

		policy_stable = False
		num_improv_iter = 0
		total_value_iter = 0

		while not policy_stable:
			value_func, num_eval_iter = evaluate_policy_async_randperm(env, value_func, gamma, policy, max_iterations, tol)
			total_value_iter += num_eval_iter
			policy_stable, policy = improve_policy(env, gamma, value_func, policy)
			num_improv_iter += 1

		return policy, value_func, num_improv_iter, total_value_iter


def value_iteration_sync(env, gamma, max_iterations=int(1e3), tol=1e-3):
	"""Runs value iteration for a given gamma and environment.

	Parameters
	----------
	env: gym.core.Environment
		The environment to compute value iteration for. Must have nS,
		nA, and P as attributes.
	gamma: float
		Discount factor, must be in range [0, 1)
	max_iterations: int
		The maximum number of iterations to run before stopping.
	tol: float
		Determines when value function has converged.

	Returns
	-------
	np.ndarray, iteration
		The value function and the number of iterations it took to converge.
	"""
	value_func = np.zeros(env.nS)  # initialize value function
	num_iters = 0

	while num_iters < max_iterations:
		delta = 0
		updated_value_func = np.copy(value_func)

		for state in range(env.nS):
			action_values = []
			for action in range(env.nA):
				prob, nextstate, reward, is_terminal = env.P[state][action][0]
				action_values.append(reward + gamma * value_func[nextstate])

			updated_value_func[state] = np.argmax(action_values)
			delta = max(delta, abs(updated_value_func[state] - value_func[state]))

		value_func = updated_value_func
		num_iters += 1

		if delta < tol:
			break

	return value_func, num_iters


def value_iteration_async_ordered(env, gamma, max_iterations=int(1e3), tol=1e-3):
		"""Runs value iteration for a given gamma and environment.
		Updates states in their 1-N order.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		gamma: float
			Discount factor, must be in range [0, 1)
		max_iterations: int
			The maximum number of iterations to run before stopping.
		tol: float
			Determines when value function has converged.

		Returns
		-------
		np.ndarray, iteration
			The value function and the number of iterations it took to converge.
		"""
		value_func = np.zeros(env.nS)  # initialize value function
		return value_func, 0


def value_iteration_async_randperm(env, gamma, max_iterations=int(1e3),
																	 tol=1e-3):
		"""Runs value iteration for a given gamma and environment.
		Updates states by randomly sampling index order permutations.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		gamma: float
			Discount factor, must be in range [0, 1)
		max_iterations: int
			The maximum number of iterations to run before stopping.
		tol: float
			Determines when value function has converged.

		Returns
		-------
		np.ndarray, iteration
			The value function and the number of iterations it took to converge.
		"""
		value_func = np.zeros(env.nS)  # initialize value function
		return value_func, 0


def value_iteration_async_custom(env, gamma, max_iterations=int(1e3), tol=1e-3):
		"""Runs value iteration for a given gamma and environment.
		Updates states by student-defined heuristic.

		Parameters
		----------
		env: gym.core.Environment
			The environment to compute value iteration for. Must have nS,
			nA, and P as attributes.
		gamma: float
			Discount factor, must be in range [0, 1)
		max_iterations: int
			The maximum number of iterations to run before stopping.
		tol: float
			Determines when value function has converged.

		Returns
		-------
		np.ndarray, iteration
			The value function and the number of iterations it took to converge.
		"""
		value_func = np.zeros(env.nS)  # initialize value function
		return value_func, 0


######################
#  Optional Helpers  #
######################

# Here we provide some helper functions simply for your convinience.
# You DON'T necessarily need them, especially "env_wrapper" if
# you want to deal with it in your different ways.

# Feel FREE to change/delete these helper functions.

def display_policy_letters(env, policy):
		"""Displays a policy as letters, as required by problem 1.2 & 1.3

		Parameters
		----------
		env: gym.core.Environment
		policy: np.ndarray, with shape (env.nS)
		"""
		policy_letters = []
		for l in policy:
				policy_letters.append(lake_env.action_names[l][0])

		policy_letters = np.array(policy_letters).reshape(env.nrow, env.ncol)


		for row in range(env.nrow):
				print(''.join(policy_letters[row, :]))


def env_wrapper(env_name):
		"""Create a convinent wrapper for the loaded environment

		Parameters
		----------
		env: gym.core.Environment

		Usage e.g.:
		----------
				envd4 = env_load('Deterministic-4x4-FrozenLake-v0')
				envd8 = env_load('Deterministic-8x8-FrozenLake-v0')
		"""
		env = gym.make(env_name)

		# T : the transition probability from s to s’ via action a
		# R : the reward you get when moving from s to s' via action a
		env.T = np.zeros((env.nS, env.nA, env.nS))
		env.R = np.zeros((env.nS, env.nA, env.nS))

		for state in range(env.nS):
			for action in range(env.nA):
				for prob, nextstate, reward, is_terminal in env.P[state][action]:
						env.T[state, action, nextstate] = prob
						env.R[state, action, nextstate] = reward
		return env


def value_func_heatmap(env, value_func):
		"""Visualize a policy as a heatmap, as required by problem 1.2 & 1.3

		Note that you might need:
				import matplotlib.pyplot as plt
				import seaborn as sns

		Parameters
		----------
		env: gym.core.Environment
		value_func: np.ndarray, with shape (env.nS)
		"""
		fig, ax = plt.subplots(figsize=(7,6))
		sns.heatmap(np.reshape(value_func, [env.nrow, env.ncol]),
								annot=False, linewidths=.5, cmap="GnBu_r", ax=ax,
								yticklabels = np.arange(1, env.nrow+1)[::-1],
								xticklabels = np.arange(1, env.nrow+1))
		# Other choices of cmap: YlGnBu
		# More: https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
		return None


def q1_2(gamma):
	envs = ['Deterministic-4x4-FrozenLake-v0', 'Deterministic-8x8-FrozenLake-v0']
	
	for env_name in envs:
		env = gym.make(env_name)		
		policy, value_func, num_improv_iter, num_value_iter = policy_iteration_sync(env, gamma)
		print('Optimal policy for {}:'.format(env_name))
		display_policy_letters(env, policy)
		print()
		value_func_heatmap(env, value_func)
	
	# Show the generated heatmaps
	plt.show()


def q1_3(gamma):
	envs = ['Deterministic-4x4-FrozenLake-v0', 'Deterministic-8x8-FrozenLake-v0']

	for env_name in envs:
		env = gym.make(env_name)
		value_func, num_iters = value_iteration_sync(env, gamma, max_iterations=1e10)
		print(env_name + ':')
		print('Number of iterations = {}'.format(num_iters))
		print('Display policy:')
		display_policy_letters(env, policy)
		print()
		value_func_heatmap(env, value_func)
	
	# Show the generated heatmaps
	plt.show()
		


def q1_4(gamma):
	env = gym.make('Deterministic-8x8-FrozenLake-v0')

	# One trial for async_ordered heuristic
	policy, value_func, num_improv_iter, num_value_iter = policy_iteration_async_ordered(env, gamma)
	print('1 trial of async_ordered:')
	print('Policy improvement steps: {}'.format(num_improv_iter))
	print('Total policy evaluation steps: {}\n'.format(num_value_iter))

	# Ten trials for async_randperm heuristic
	randperm_improv_iter = []
	randperm_value_iter = []
	for i in range(10):
		policy, value_func, num_improv_iter, num_value_iter = policy_iteration_async_ordered(env, gamma)
		randperm_improv_iter.append(num_improv_iter)
		randperm_value_iter.append(num_value_iter)
	
	print('10 trials of async_randperm:')
	print('Policy improvement steps: {}'.format(np.mean(randperm_improv_iter)))
	print('Total policy evaluation steps: {}\n'.format(np.mean(randperm_value_iter)))


if __name__ == "__main__":
	# Define num_trials, gamma and whatever variables you need below.
	gamma = 0.9
	
	# q1_2(gamma)

	# q1_4(gamma)

	q1_3(gamma)