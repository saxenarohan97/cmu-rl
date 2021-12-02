import numpy as np
import scipy.linalg


def simulate_dynamics(sim_env, x, u, dt=1e-5):
    """Step simulator to see how state changes.

    Parameters
    ----------
    sim_env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. When approximating A you will need to perturb
      this.
    u: np.array
      The command to test. When approximating B you will need to
      perturb this.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    xdot: np.array
      This is the **CHANGE** in x. i.e. (x[1] - x[0]) / dt
      If you return x you will need to solve a different equation in
      your LQR controller.
    """
    # ___ WRITE CODE HERE ___
    # sim_env.reset()
    sim_env.state = np.copy(x)
    sim_env.dt = dt
    xdot = (sim_env.step(u)[0]  - x) / dt
    # ^^^ WRITE CODE HERE ^^^
    return xdot


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate A matrix using finite differences. Note that you are
    approximating the Jacobian of xdot = f(x, u) with respect to x.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test. You will need to perturb this.
    u: np.array
      The command to test.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    A: np.array
      The A matrix for the dynamics at state x and command u.
    """
    # ___ WRITE CODE HERE ___
    A = np.zeros((x.shape[0], x.shape[0]))
    for i in range(len(x)):
      x_pert_plus = np.copy(x)
      x_pert_plus[i] += delta
      x_pert_minus = np.copy(x)
      x_pert_minus[i] -= delta
      df_xi = (simulate_dynamics(env, x_pert_plus, u, dt)
                - simulate_dynamics(env, x_pert_minus, u, dt)) / 2 / delta
      A[:,i] = np.copy(df_xi)
    # ^^^ WRITE CODE HERE ^^^
    return A


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Approximate B matrix using finite differences.

    Parameters
    ----------
    env: gym.core.Env
      The environment you are try to control. In this homework the 2
      link arm.
    x: np.array
      The state to test.
    u: np.array
      The command to test. You will ned to perturb this.
    delta: float
      How much to perturb the state by.
    dt: float, optional
      The time step to simulate. In general the smaller the time step
      the more accurate the gradient approximation.

    Returns
    -------
    B: np.array
      The B matrix for the dynamics at state x and command u.
    """
    # ___ WRITE CODE HERE ___
    B = np.zeros((x.shape[0], u.shape[0]))
    for i in range(len(u)):
      u_pert_plus = np.copy(u)
      u_pert_plus[i] += delta
      u_pert_minus = np.copy(u)
      u_pert_minus[i] -= delta
      df_ui = (simulate_dynamics(env, x, u_pert_plus, dt)
                - simulate_dynamics(env, x, u_pert_minus, dt)) / 2 / delta
      B[:,i] = np.copy(df_ui)
    # ^^^ WRITE CODE HERE ^^^
    return B


def calc_lqr_input(env, sim_env):
    """Calculate the optimal control input for the given state.

    Approximate A and B with zero action u=0, then solve the continuous ARE
    to help compute the optimal action u.

    Parameters
    ----------
    env: gym.core.Env
      This is the true environment you will execute the computed
      commands on. Use this environment to get the Q and R values as
      well as the state.
    sim_env: gym.core.Env
      A copy of the env class. Use this to simulate the dynamics when
      doing finite differences.

    Returns
    -------
    u: np.array
      The command to execute at this point.
    """
    # ___ WRITE CODE HERE ___
    # approximate the dynamics for current state x, not x-x*!
    # x-x* is only used to generate the optimal control u!
    Q = env.Q
    R = env.R
    x = env.state
    u = np.zeros(env.action_space.shape)
    A = approximate_A(sim_env, x, u)
    B = approximate_B(sim_env, x, u)
    S = scipy.linalg.solve_continuous_are(A, B, Q, R, balanced=False)
    x_star = env.goal
    K = np.linalg.inv(R) @ B.T @ S
    u = - K @ (x - x_star)
    return u
