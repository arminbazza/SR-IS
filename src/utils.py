import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.animation as manimation
import random


def update_terminal_reward(agent, loc, r):
    """
    Update the reward for the terminal state of the agent according to loc

    Args:
        agent (LinearRL class) : The SR-IS agent
        loc (int) : The terminal location to change the reward of [0->n] n= number of terminal locations - 1
        r (float) : The new reward to change r[loc] to
    """
    # Get location of reward and change
    r_loc = np.argwhere(agent.terminals)[loc]
    agent.r[r_loc] = r
    # Update expr_t inside of the agent
    agent.expr_t = np.exp(agent.r[agent.terminals] / agent._lambda)

def update_terminal_reward_SR(agent, loc, r):
    """
    Update the reward for the terminal state of the agent according to loc

    Args:
        agent (SR-IS class) : The SR agent
        loc (int) : The terminal location to change the reward of [0->n] n= number of terminal locations - 1
        r (float) : The new reward to change r[loc] to
    """
    # Get location of reward and change
    r_loc = np.argwhere(agent.terminals)[loc]
    agent.r[r_loc] = r

def exponential_decay(initial_learning_rate, decay_rate, global_step, decay_steps):
    """
    Applies exponential decay to the learning rate.
    """
    learning_rate = initial_learning_rate * decay_rate ** (global_step / decay_steps)
    return learning_rate

def woodbury(agent, T, inv=False):
    """
    Applies the woodbury update to the DR of the agent, accomodating for new transition structure (T)
    
    Args:
        agent (SR-IS class) : The SR-IS agent 
        T (array) : The transition matrix of the new environment
        inv (bool) : Whether or not to use the inverse matrix for an absolute solution (used for debugging/sanity check)
    """
    differences = agent.T != T
    different_rows, _ = np.where(differences)
    delta_locs = np.unique(different_rows)

    L0 = np.diag(np.exp(-agent.r)/agent._lambda) - agent.T 
    L = np.diag(np.exp(-agent.r)/agent._lambda) - T

    if inv:
        D0 = np.linalg.inv(L0)
    else:
        D0 = agent.gamma * agent.DR

    R = L[delta_locs, :] - L0[delta_locs, :]
    m0 = D0[:,delta_locs]

    C = np.zeros_like(m0)
    C[delta_locs[0], 0] = 1
    C[delta_locs[1], 1] = 1

    A = np.linalg.inv(np.eye(len(delta_locs)) + R @ D0 @ C)

    D = D0 - (D0 @ C @ A @ R @ D0)

    return D

def woodbury_SR(agent, T, T_pi, inv=False):
    """
    Applies the woodbury update to the SR of the agent, accomodating for new transition structure (T)
    
    Args:
        agent (SR-IS class) : The SR-IS agent 
        T (array) : The transition matrix of the new environment
        T_pi (array) : The biased SR transition matrix
        inv (bool) : Whether or not to use the inverse matrix for an absolute solution (used for debugging/sanity check)
    """
    differences = T_pi != T
    different_rows, _ = np.where(differences)
    delta_locs = np.unique(different_rows)

    L0 = np.diag(np.full(agent.size, agent.gamma)) - T_pi
    L = np.diag(np.full(agent.size, agent.gamma)) - T

    if inv:
        D0 = np.linalg.inv(L0)
    else:
        D0 = agent.SR

    R = L[delta_locs, :] - L0[delta_locs, :]
    m0 = D0[:,delta_locs]

    C = np.zeros_like(m0)
    C[delta_locs[0], 0] = 1
    C[delta_locs[1], 1] = 1

    A = np.linalg.inv(np.eye(len(delta_locs)) + R @ D0 @ C)

    D = D0 - (D0 @ C @ A @ R @ D0)

    return D

def woodbury_V(agent, D, term_reward):
    Z, V = np.zeros(agent.size), np.zeros(agent.size)
    Z[~agent.terminals] = D[~agent.terminals][:,~agent.terminals] @ agent.P @ np.array([np.exp(term_reward)])
    Z[agent.terminals] = np.exp(term_reward)
    
    min_z = np.min(Z)
    Z += (np.abs(min_z) + 0.1)
    V = np.round(np.log(Z), 5)

    return Z, V

def policy_reval(agent):
    """
    Performs replanning when the reward structure of the environment has changed
    
    Args:
        agent (SR-IS class) : The SR-IS agent

    Returns:
        V_new (array) : New value of each state
    """
    r_new = agent.r
    expr_new = np.exp(r_new[agent.terminals] / agent._lambda)
    Z_new = np.zeros(len(r_new))

    Z_new[~agent.terminals] = agent.DR[~agent.terminals][:,~agent.terminals] @ agent.P @ expr_new
    Z_new[agent.terminals] = expr_new
    V_new = np.log(Z_new) * agent._lambda

    return V_new, Z_new

def new_goal(agent, T, loc):
    """
    New Environment is the same as the old one, with the inclusion of a new goal state that we want to use the old DR to
    plan towards
    
    Args:
    agent (SR-IS class): The SR-IS agent 
    T (array): The transition matrix of the new environment
    loc (tuple): Location of the new goal state
    """
    D0 = agent.DR
    L0 = np.diag(np.exp(-agent.r)) - agent.T
    L = np.diag(np.exp(-agent.r)) - T

    idx = agent.mapping[loc]

    d = L[idx, :] - L0[idx, :]
    m0 = D0[:,idx]

    d = d.reshape(1, -1)

    m0 = m0.reshape(-1, 1)

    alpha = (np.dot(m0,d)) / (1 + (np.dot(d,m0)))
    change = np.dot(alpha,D0)

    D = np.copy(D0)
    D -= change

    agent.DR = D

    agent.terminals = np.diag(T) == 1
    agent.P = T[~agent.terminals][:,agent.terminals]
    agent.r = np.full(len(T), -1)
    agent.r[agent.terminals] = 20
    agent.expr = np.exp(agent.r[agent.terminals] / agent._lambda)

def create_mapping(maze):
    """
    Creates a mapping from maze state indices to transition matrix indices
    """
    n = len(maze)  # Size of the maze (N)

    mapping = {}
    matrix_idx = 0

    for i in range(n):
        for j in range(n):
            mapping[(i,j)] = matrix_idx
            matrix_idx += 1

    return mapping

def create_mapping_nb(maze, walls):
    """
    Creates a mapping from maze state indices to transition matrix indices
    This mapping *excludes* blocks that are inacessible, hence the nb stands for "not blocked"
    """
    n = len(maze)  # Size of the maze (N)

    # Create a mapping from maze state indices to transition matrix indices
    mapping = {}
    matrix_idx = 0

    for i in range(n):
        for j in range(n):
            if (i, j) not in walls:
                mapping[(i, j)] = matrix_idx
                matrix_idx += 1

    return mapping

def get_transition_matrix_nb(env, size, mapping):
    """
    Creates a state -> state transition matrix. This means the transition matrix *excludes* blocks that are inacessible, hence the nb stands for "not blocked"
    """
    barriers = []
    maze = env.unwrapped.maze

    T = np.zeros(shape=(size, size))
    # loop through the maze
    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):  
            # if we hit a barrier
            if maze[row,col] == '1':
                barriers.append(mapping[row, col])
                continue

            idx_cur = mapping[row, col]

            # check if current state is terminal
            if maze[row,col] == 'G':
                T[idx_cur, idx_cur] = 1
                continue

            state = (row,col)
            successor_states = env.unwrapped.get_successor_states(state)
            for successor_state in successor_states:
                idx_new = mapping[successor_state[0][0], successor_state[0][1]]
                T[idx_cur, idx_new] = 1/len(successor_states)
    
    return T, barriers

def get_transition_matrix(env, mapping):
    """
    Creates a state -> state transition matrix.
    """
    maze = env.unwrapped.maze

    T = np.zeros(shape=(len(mapping), len(mapping)))
    # loop through the maze
    for row in range(maze.shape[0]):
        for col in range(maze.shape[1]):            
            # if we hit a barrier
            if maze[row,col] == '1':
                continue

            idx_cur = mapping[row, col]

            # check if current state is terminal
            if maze[row,col] == 'G':
                T[idx_cur, idx_cur] = 1
                continue

            state = (row,col)
            successor_states = env.unwrapped.get_successor_states(state)
            for successor_state in successor_states:
                idx_new = mapping[successor_state[0][0], successor_state[0][1]]
                T[idx_cur, idx_new] = 1/len(successor_states)
    
    return T

def get_map(agent):
    # Replace 'S' and 'G' with 0
    m = np.where(np.isin(agent.maze, ['S', 'G']), '0', agent.maze)

    # Convert the array to int
    m = m.astype(int)
    
    return m

def get_full_maze_values(agent):
    """
    Function that prints out the values of each state and labels blocked states
    """
    v_maze = np.zeros_like(agent.maze, dtype=np.float64)
    for row in range(v_maze.shape[0]):
        for col in range(v_maze.shape[1]):
            if agent.maze[row, col] == "1":
                v_maze[row,col] = -np.inf
                continue
            v_maze[row,col] = np.round(agent.V[agent.mapping[(row,col)]], 2)
    
    return v_maze

def decision_policy(agent, Z):
    """
    Performs matrix version of equation 6 from the LinearRL paper

    Args:
        agent (SR-IS class) : The SR-IS agent
        Z (array) : The Z-Values to operate on (usually just agent.Z)

    Returns:
        pii (array) : The decision policy
    """
    G = np.dot(agent.T, Z)

    expv_tiled = np.tile(Z, (len(Z), 1))
    G = G.reshape(-1, 1)
    
    zg = expv_tiled / G
    pii = agent.T * zg

    return pii

def softmax(x, temperature=1.0):
    """
    Compute the softmax of a vector with inverse temperature scaling.
    """
    x_temp = x / temperature
    # Numerical stability
    x_shifted = x_temp - np.max(x_temp)
    exp_x = np.exp(x_shifted)

    return exp_x / np.sum(exp_x)

def decision_policy_SR(agent):
    """
    Computes the SR decision policy, which acts as T^pi
    """
    T_pi = np.zeros_like(agent.T)
    for state in agent.mapping:
        state_idx = agent.mapping[state]
        successor_states = agent.env.unwrapped.get_successor_states(state)

        # Fixed softmax calculation in one line
        exp_values = np.exp([agent.V[agent.mapping[(s[0][0],s[0][1])]] / agent.beta for s in successor_states])
        v_sum = np.sum(exp_values)

        for i, action in enumerate(agent.env.unwrapped.get_available_actions(state)):
            direction = agent.env.unwrapped._action_to_direction[action]
            new_state = state + direction
            new_state_idx = agent.mapping[(new_state[0], new_state[1])]
            prob = exp_values[i] / v_sum  # Use the corresponding exp value
            T_pi[state_idx, new_state_idx] += prob

    T_pi[agent.terminals, agent.terminals] = 1

    return T_pi
        
def gen_nhb_exp():
    """
    Defines the environment for the Nature Human Behavior experiment introduced by Momennejad et al.
    """
    envstep=[]
    for s in range(6):
        # actions 0=left, 1=right
        envstep.append([[0,0], [0,0]])  # [s', done]
    envstep = np.array(envstep)

    # State 0 -> 1, 2
    envstep[0,0] = [1,0]
    envstep[0,1] = [2,0]

    # State 1 -> 3, 4
    envstep[1,0] = [3,0]
    envstep[1,1] = [4,0]

    # State 2 -> 4, 5
    envstep[2,0] = [4,0]
    envstep[2,1] = [5,0]

    # State 3 -> 3'
    envstep[3,0] = [6,1]
    envstep[3,1] = [6,1]

    # State 4 -> 4'
    envstep[4,0] = [7,1]
    envstep[4,1] = [7,1]

    # State 5 -> 5'
    envstep[5,0] = [8,1]
    envstep[5,1] = [8,1]
    
    return envstep

def gen_nhb_exp_SR():
    """
    Defines the environment for the Nature Human Behavior experiment introduced by Momennejad et al.
    """
    envstep=[]
    for s in range(6):
        # actions 0=left, 1=right
        envstep.append([[0,0], [0,0]])  # [s', done]
    envstep = np.array(envstep)

    # State 0 -> 1, 2
    envstep[0,0] = [1,0]
    envstep[0,1] = [2,0]

    # State 1 -> 3, 4
    envstep[1,0] = [3,1]
    envstep[1,1] = [4,1]

    # State 2 -> 4, 5
    envstep[2,0] = [4,1]
    envstep[2,1] = [5,1]
    
    return envstep

def gen_two_step():
    """
    Defines the environment for the two-step task. Each step returns the next state and if we are in a terminal state
    """
    envstep=[]
    for s in range(3):
        # actions 0=left, 1=right
        envstep.append([[0,0], [0,0]])  # [s', done]
    envstep = np.array(envstep)

    # State 0 -> 1, 2
    envstep[0,0] = [1,0]
    envstep[0,1] = [2,0]

    # State 1 -> 3, 4
    envstep[1,0] = [3,1]
    envstep[1,1] = [4,1]

    # State 2 -> 5, 6
    envstep[2,0] = [5,1]
    envstep[2,1] = [6,1]
    
    return envstep

class TwoStepStochastic:
    def __init__(self, size=7, prob_common=0.5, seed=None, stoch_states={1}):
        self.size = size
        self.prob_common = prob_common
        self.rng = np.random.RandomState(seed)
        
        self.envstep = self._build_transition_table()
        self.stochastic_states = stoch_states
        
    def _build_transition_table(self):
        """Build the base transition lookup table."""
        envstep = []
        for s in range(self.size):
            envstep.append([[0, 0], [0, 0]])
        envstep = np.array(envstep)
        
        # State 0 -> 1, 2 (deterministic)
        envstep[0, 0] = [1, 0]
        envstep[0, 1] = [2, 0]
        
        # State 1 -> 3, 4 (stochastic)
        envstep[1, 0] = [3, 1]  # common for action 0
        envstep[1, 1] = [4, 1]  # common for action 1
        
        # State 2 -> 5, 6
        envstep[2, 0] = [5, 1]
        envstep[2, 1] = [6, 1]
        
        return envstep
        
    def step_deterministic(self, state, action):
        state, done = self.envstep[state, action]
        return state, done

    def step(self, state, action):
        # Get the "common" transition for this action
        common_state, done = self.envstep[state, action]
        
        # Handle stochastic transitions
        if state in self.stochastic_states:
            if self.rng.random() < self.prob_common:
                # Common transition
                next_state = common_state
            else:
                # Rare transition (flip to the other action's common state)
                rare_action = 1 - action
                next_state = self.envstep[state, rare_action][0]
        else:
            # Deterministic transition
            next_state = common_state
            
        return next_state, done
    
    def reset(self):
        """Reset to initial state."""
        return 0
    
    def get_transition_type(self, state, action, next_state):
        if state not in self.stochastic_states:
            return 'deterministic'
        
        common_state = self.envstep[state, action][0]
        if next_state == common_state:
            return 'common'
        else:
            return 'rare'

def test_agent(agent, policy="greedy", state=None, seed=None, term_state=None):
    """
    Function to test the agent

    Args:
        agent (SR-IS class) : The SR-IS agent
        policy (string) : Which policy to use, default is greedy
        state (tuple) : The starting state, default is none which sets the agent at the starting state of the maze
        term_state (tuple) : Check if we reach a specific terminal state
    
    Returns:
        traj (list) : The list of states the agent chooses
    """
    # Set the policy to testing
    agent.policy = policy

    traj = []
    
    if seed is not None:
        agent.env.reset(seed=seed, options={})
        np.random.seed(seed)
    
    if state is None:
        state = agent.start_loc

    # set the start and agent location
    agent.env.unwrapped.start_loc, agent.env.unwrapped.agent_loc = state, state

    # If we are checking for a specific terminal state
    if term_state is not None:
        while True:
            if policy == "softmax":
                action, _ = agent.select_action(state)
            else:
                action = agent.select_action(state)

            obs, _, done, _, _ = agent.env.step(action)
            next_state = obs["agent"]
            traj.append(next_state)
            state = next_state
            if np.all(next_state == term_state):
                break
    else:
        while True:
            if policy == "softmax":
                action, _ = agent.select_action(state)
            else:
                action = agent.select_action(state)

            obs, _, done, _, _ = agent.env.step(action)
            next_state = obs["agent"]
            traj.append(next_state)
            state = next_state

            if done:
                break

    return traj