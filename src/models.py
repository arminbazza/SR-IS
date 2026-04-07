import numpy as np
import gymnasium as gym

import gym_env
from utils import get_transition_matrix, create_mapping_nb, gen_nhb_exp, gen_nhb_exp_SR, exponential_decay, TwoStepStochastic


class SR_IS:
    """
    SR-IS agent for maze environments

    Args:
        reward (float) : The reward to make states (has effect on gamma)
        term_reward (float) : The reward of the terminal state
        alpha (float) : Learning rate
        beta (float) : Temperature parameter
        _lambda (float) : Lambda value that downweights the rewards
        num_steps (int) : Number of training steps
        policy (string) : Decision policy
        imp_samp (bool) : Whether or not to use importance sampling
        decay (bool) : Whether or not to use learning rate decay
        decay_params (list) : decay parameters (decay_rate, decay_steps)
    """
    def __init__(self, env_name, reward=-0.2, term_reward=-0.2, alpha=0.1, beta=1, _lambda=1.0, num_steps=25000, policy="random", imp_samp=False, decay=False, decay_params=None):
        self.env = gym.make(env_name)
        self.start_loc = self.env.unwrapped.start_loc
        self.target_locs = self.env.unwrapped.target_locs
        self.maze = self.env.unwrapped.maze
        self.walls = self.env.unwrapped.get_walls()
        self.size = self.maze.size - len(self.walls)   # Size of the state space is the = size of maze - number of blocked states
        self.height, self.width = self.maze.shape

        # Create mapping and Transition matrix
        self.mapping = create_mapping_nb(self.maze, self.walls)
        self.reverse_mapping = {index: (i, j) for (i, j), index in self.mapping.items()}
        self.T = get_transition_matrix(self.env, self.mapping)

        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        # Calculate P = T_{NT}
        self.P = self.T[~self.terminals][:,self.terminals]
        # Set reward
        self.reward_nt = reward   # Non-terminal state reward
        self.reward_t = term_reward    # Terminal state reward
        self.r = np.full(len(self.T), self.reward_nt)
        self.r[self.terminals] = self.reward_t
        self.expr_t = np.exp(self.r[self.terminals] / _lambda)
        # Precalculate exp(r) for use with LinearRL equations
        self.expr_nt = np.exp(self.reward_nt / _lambda)

        # Params
        self.alpha = alpha
        self.beta = beta
        self.gamma = self.expr_nt
        self._lambda = _lambda
        self.num_steps = num_steps
        self.policy = policy
        self.imp_samp = imp_samp
        self.decay = decay

        if decay:
            self.initial_lr = alpha
            self.decay_rate = decay_params[0]
            self.decay_steps = decay_params[1]

        # Model
        self.DR = self.get_DR()
        self.Z = np.full(self.size, 0.01)

        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def get_DR(self):
        """
        Returns the DR initialization based on what decision policy we are using, values are filled with 0.01 if using softmax to avoid div by zero
        """
        if self.policy == "random":
            DR = np.eye(self.size)
            DR[np.where(self.terminals)[0], np.where(self.terminals)[0]] = (1/(1-self.gamma))
        
        elif self.policy == "softmax":
            DR = np.full((self.size, self.size), 0.01)
            np.fill_diagonal(DR, 1)
            DR[np.where(self.terminals)[0], np.where(self.terminals)[0]] = (1/(1-self.gamma))

        return DR
    
    def get_D_inv(self):
        """
        Calculates the DR directly using matrix inversion, used for testing
        """
        I = np.eye(self.size)
        D_inv = np.linalg.inv(I-self.gamma*self.T)

        return D_inv

    def update_V(self):
        self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t
        self.Z[self.terminals] = self.expr_t
        self.V = np.round(np.log(self.Z), 5)
    
    def importance_sampling(self, state, s_prob):
        """
        Performs importance sampling P(x'|x)/u(x'|x). P(.) is the default policy, u(.) us the decision policy
        """
        successor_states = self.env.unwrapped.get_successor_states(state)
        p = 1/len(successor_states)
        w = p/s_prob

        return w

    def select_action(self, state):
        """
        Action selection based on our policy
        Options are: [random, softmax, greedy]
        """
        if self.policy == "random":
            return self.env.unwrapped.random_action()
        
        elif self.policy == "softmax":
            successor_states = self.env.unwrapped.get_successor_states(state)
            action_probs = np.full(self.env.action_space.n, 0.0)

            v_sum = sum(
                        np.exp((np.log(self.Z[self.mapping[(s[0][0],s[0][1])]] + 1e-20)) / self.beta) for s in successor_states
                        )

            # if we don't have enough info, random action
            if v_sum == 0:
                return self.env.unwrapped.random_action(), 1

            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction
                # print(self.Z[self.mapping[(new_state[0], new_state[1])]])
                action_probs[action] = np.exp( (np.log( self.Z[self.mapping[(new_state[0], new_state[1])]] + 1e-20 )) / self.beta ) / v_sum

            if np.any(np.isnan(action_probs)):
                print("NaNs detected, recalculating with safer log operations")
                
                # Recalculate with the same logic but safer operations
                v_sum = sum(
                    np.exp((np.log(np.maximum(self.Z[self.mapping[(s[0][0],s[0][1])]], 1e-10))) / self.beta) 
                    for s in successor_states
                )
                
                if v_sum == 0:
                    return self.env.unwrapped.random_action(), 1
                
                for action in self.env.unwrapped.get_available_actions(state):
                    direction = self.env.unwrapped._action_to_direction[action]
                    new_state = state + direction
                    z_val = np.maximum(self.Z[self.mapping[(new_state[0], new_state[1])]], 1e-10)
                    action_probs[action] = np.exp(np.log(z_val) / self.beta) / v_sum
                
                # Final safety check
                if np.any(np.isnan(action_probs)) or np.sum(action_probs) == 0:
                    action_probs = np.ones_like(action_probs) / len(action_probs)

            # NaN handler
            # if np.any(np.isnan(action_probs)):
            #     action_probs = np.nan_to_num(action_probs)  # NaNs become 0s
            #     total_sum = np.sum(action_probs)
            #     if total_sum == 0:  # All were NaN, so now all are 0
            #         action_probs = np.ones_like(action_probs) / len(action_probs)  # Uniform distribution
            #     else:
            #         action_probs = action_probs / total_sum

            action = np.random.choice(self.env.action_space.n, p=action_probs)
            s_prob = action_probs[action]

            return action, s_prob
            
        elif self.policy == "greedy":
            action_values = np.full(self.env.action_space.n, -np.inf)
            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction

                if np.any(np.all(self.target_locs == new_state, axis=1)):
                    return action

                if self.maze[new_state[0], new_state[1]] == "1":
                    continue
                action_values[action] = self.V[self.mapping[(new_state[0], new_state[1])]]

            return np.nanargmax(action_values)

    def take_action(self, state):
        """
        Select an action according to policy and take it. 
        """
        # Choose action
        if self.policy == "softmax":
            action, s_prob = self.select_action(state)
        else:
            action = self.select_action(state)

        # Take action
        obs, _, done, _, _ = self.env.step(action)

        # Unpack observation to get new state
        next_state = obs["agent"]
        next_state_idx = self.mapping[(next_state[0], next_state[1])]

        # Importance sampling
        if self.imp_samp:
            w = self.importance_sampling(state, s_prob)
            w = 1 if np.isnan(w) or w == 0 else w
        else:
            w = 1
        
        return next_state, next_state_idx, w, done

    def update(self, state_idx, next_state_idx, w, step):
        """
        Update the DR
        """
        # If we are using lr decay
        if self.decay:
            self.alpha = exponential_decay(self.initial_lr, self.decay_rate, step, self.decay_steps)

        # Update default representation
        target = self.one_hot[state_idx] + self.gamma * self.DR[next_state_idx]
        self.DR[state_idx] = (1 - self.alpha) * self.DR[state_idx] + self.alpha * target * w

        # Update Z-Values
        self.Z = self.DR[:,~self.terminals] @ self.P @ self.expr_t

    def learn(self, seed=None):
        """
        Agent explores the maze according to its decision policy and and updates its DR as it goes
        """
        # Set the seeds
        self.env.reset(seed=seed, options={})
        np.random.seed(seed)
        self.Z[self.terminals] = self.expr_t
        
        # Iterate through number of steps
        for i in range(self.num_steps):
            # Current state
            state = self.env.unwrapped.agent_loc
            state_idx = self.mapping[(state[0], state[1])]

            # Take action
            next_state, next_state_idx, w, done = self.take_action(state)

            # Update  DR
            self.update(state_idx, next_state_idx, w, i)

            if done:
                self.env.reset(seed=seed, options={})
                continue
            
            # Update state
            state = next_state

        # Update DR at terminal state
        self.update_V()


class SR_IS_NHB:
    def __init__(self, alpha=0.25, beta=1.0, _lambda=10, epsilon=0.4, num_steps=250, policy="softmax", imp_samp=True, exp_type="policy_reval"):
        # Hard code start and end locations as well as size
        self.start_loc = 0
        self.target_locs = [3,4,5]
        self.start_locs = [0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 1, 2, 1, 2, 3, 4, 5, 0, 0, 1, 0, 0, 2, 3, 1, 4, 2, 5, 0, 1, 2, 0]
        self.size = 9
        self.agent_loc = self.start_loc
        self.exp_type = exp_type

        # Construct the transition probability matrix and envstep functions
        self.T = self.construct_T()
        self.envstep = gen_nhb_exp()
        
        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        # Calculate P = T_{NT}
        self.P = self.T[~self.terminals][:,self.terminals]

        # Set reward
        self.reward_nt = -0.1
        self.r = np.full(len(self.T), self.reward_nt)
        # Reward of terminal states depends on if we are replicating reward revaluation or policy revaluation
        if self.exp_type == "policy_reval":
            self.r_term_1 = [0, 15, 30]
            self.r_term_2 = [45, 15, 30]
        elif self.exp_type == "reward_reval":
            self.r_term_1 = [15, 0, 30]
            self.r_term_2 = [45, 0, 30]
        elif self.exp_type == "trans_reval":
            self.r_term_1 = [15, 0, 30]
        else:
            print("Incorrect experiment type (exp_type)")
            return(0)
        self.r[self.terminals] = self.r_term_1

        # Precalculate exp(r) for use with LinearRL equations
        self.expr_t = np.exp(self.r[self.terminals] / _lambda)
        self.expr_nt = np.exp(self.reward_nt / _lambda)

        # Params
        self.alpha = alpha
        self.beta = beta
        self.gamma = self.expr_nt
        self._lambda = _lambda
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.policy = policy
        self.imp_samp = imp_samp

        # Model
        self.DR = self.get_DR()
        self.Z = np.full(self.size, 0.01)

        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def construct_T(self):
        """
        Manually construct the transition matrix
        """
        # For NHB two-step task
        T = np.zeros((9, 9))
        T[0, 1:3] = 0.5
        T[1, 3:5] = 0.5
        T[2, 4:6] = 0.5
        T[3, 6] = 1
        T[4, 7] = 1
        T[5, 8] = 1
        T[6:9, 6:9] = np.eye(3)

        return T
    
    def construct_T_new(self):
        """
        Manually construct a new transition matrix for the transition revaluation problem
        """
        T = np.zeros((9, 9))
        T[0, 1:3] = 0.5
        T[1, 4:6] = 0.5
        T[2, 3:5] = 0.5
        T[3, 6] = 1
        T[4, 7] = 1
        T[5, 8] = 1
        T[6:9, 6:9] = np.eye(3)

        return T
    
    def update_exp(self):
        """
        Update the terminal state values or transition matrix (experiment dependent)
        """
        if self.exp_type in ["reward_reval", "policy_reval"]:
            self.r[self.terminals] = self.r_term_2

    def get_DR(self):
        """
        Returns the DR initialization based on what decision policy we are using, values are filled with 0.01 if using softmax to avoid div by zero
        """
        if self.policy == "softmax":
            DR = np.full((self.size, self.size), 0.01)
            np.fill_diagonal(DR, 1)
            DR[np.where(self.terminals)[0], np.where(self.terminals)[0]] = (1/(self.gamma))
        else:
            DR = np.eye(self.size)

        return DR

    def update_Z(self):
        self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t
        self.Z[self.terminals] = self.expr_t

    def update_V(self):
        self.V = np.log(self.Z) * self._lambda
    
    def get_successor_states(self, state):
        """
        Manually define the successor states based on which state we are in
        """
        return np.where(self.T[state, :] != 0)[0]

    def importance_sampling(self, state, s_prob):
        """
        Performs importance sampling P(x'|x)/u(x'|x). P(.) is the default policy, u(.) is the decision policy
        """
        successor_states = self.get_successor_states(state)
        p = 1/len(successor_states)
        w = p/s_prob
                
        return w

    def select_action(self, state):
        """
        Action selection based on our policy
        Options are: [random, softmax]
        """
        if self.policy == "random":
            action = np.random.choice([0,1])

            return action
        
        elif self.policy == "softmax":
            successor_states = self.get_successor_states(state)
            action_probs = np.full(2, 0.0)   # We can hardcode this because every state has 2 actions

            v_sum = sum(np.exp((np.log(self.Z[s] + 1e-20) * self._lambda) / self.beta) for s in successor_states)

            # if we don't have enough info, random action
            if v_sum == 0:
                return  np.random.choice([0,1])

            for action in [0,1]:
                new_state, done = self.envstep[state, action]

                # If we hit a done state our action doesn't matter
                if done:
                    action = np.random.choice([0,1])
                    return action, 1
                action_probs[action] = np.exp((np.log(self.Z[new_state] + 1e-20) * self._lambda) / self.beta ) / v_sum
                
            action = np.random.choice([0,1], p=action_probs)
            s_prob = action_probs[action]

            return action, s_prob

    def get_D_inv(self):
        """
        Calculates the DR directly using matrix inversion, used for testing
        """
        I = np.eye(self.size)
        D_inv = np.linalg.inv(I-self.gamma*self.T)
        
        return D_inv
    
    def learn_with_start_locs(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)

        self.Z[self.terminals] = self.expr_t

        for start_loc in self.start_locs:
            self.agent_loc = start_loc

            while True:
                state = self.agent_loc
                # Choose action
                if self.policy == "softmax":
                    action, s_prob = self.select_action(state)
                else:
                    action = self.select_action(state)
            
                # Take action
                next_state, done = self.envstep[state, action]
                
                # Importance sampling
                if self.imp_samp:
                    w = self.importance_sampling(state, s_prob)
                    w = 1 if np.isnan(w) or w == 0 else w
                else:
                    w = 1
                
                # Update default representation
                target = self.one_hot[state] + self.gamma * self.DR[next_state]
                self.DR[state] = (1 - self.alpha) * self.DR[state] + self.alpha * target * w

                # Update Z-Values
                self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t
                
                if done:
                    break

                # Update state
                state = next_state
                self.agent_loc = state

        # Update Z and V values
        self.update_Z()
        self.update_V()


    def learn(self, seed=None):
        """
        Agent explores the maze according to its decision policy and and updates its DR as it goes
        """
        if seed is not None:
            np.random.seed(seed=seed)

        # Iterate through number of steps
        for i in range(self.num_steps):
            # Agent gets some knowledge of terminal state values
            if i == 2:
                self.Z[self.terminals] = self.expr_t
            # Current state
            state = self.agent_loc

            # Choose action
            if self.policy == "softmax":
                action, s_prob = self.select_action(state)
            else:
                action = self.select_action(state)
        
            # Take action
            next_state, done = self.envstep[state, action]
            
            # Importance sampling
            if self.imp_samp:
                w = self.importance_sampling(state, s_prob)
                w = 1 if np.isnan(w) or w == 0 else w
            else:
                w = 1
            
            # Update default representation
            target = self.one_hot[state] + self.gamma * self.DR[next_state]
            self.DR[state] = (1 - self.alpha) * self.DR[state] + self.alpha * target * w

            # Update Z-Values
            self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t
            
            if done:
                self.agent_loc = self.start_loc
                continue
            
            # Update state
            state = next_state
            self.agent_loc = state

        # Update DR at terminal state
        self.update_Z()
        self.update_V()


class SR_TD:
    """
    SR-TD agent for maze environments

    Args:
        reward (float) : The reward to make states
        term_reward (float) : The reward at the terminal state
        alpha (float) : Learning rate
        gamma (float) : Discount parameter
        beta (float) : Temperature parameter
        _lambda (float) : Lambda value that downweights the rewards
        num_steps (int) : Number of training steps
        policy (string) : Decision policy
    """
    def __init__(self, env_name, reward=1, term_reward=10, gamma=0.82, alpha=0.1, beta=1, epsilon=0.1, num_steps=25000, policy="random", diag=False):
        self.env = gym.make(env_name)
        self.start_loc = self.env.unwrapped.start_loc
        self.target_locs = self.env.unwrapped.target_locs
        self.maze = self.env.unwrapped.maze
        self.walls = self.env.unwrapped.get_walls()
        self.size = self.maze.size - len(self.walls)   # Size of the state space is the = size of maze - number of blocked states
        self.height, self.width = self.maze.shape

        # Create mapping and Transition matrix
        self.mapping = create_mapping_nb(self.maze, self.walls)
        self.reverse_mapping = {index: (i, j) for (i, j), index in self.mapping.items()}
        self.T = get_transition_matrix(self.env, self.mapping)

        # Get terminal states
        self.terminals = np.diag(self.T) == 1

        # Set reward
        self.reward_nt = reward            # Non-terminal state reward
        self.reward_t = term_reward        # Terminal state reward
        self.r = np.full(self.size, self.reward_nt)
        self.r[self.terminals] = self.reward_t

        # Params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.policy = policy

        # Model
        if diag:
            self.SR = np.eye(self.size)
        else:
            self.SR = np.random.uniform(0, 0.1, (self.size, self.size))
            np.fill_diagonal(self.SR, 1)
        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def update_V(self):
        self.V = self.SR @ self.r

    def select_action(self, state):
        """
        Action selection based on our policy
        Options are: [random, softmax, greedy]
        """
        if self.policy == "random":
            return self.env.unwrapped.random_action()
        
        elif self.policy == "softmax":
            successor_states = self.env.unwrapped.get_successor_states(state)
            action_probs = np.full(self.env.action_space.n, 0.0)

            # Simple softmax without overflow protection
            exp_values = np.exp([self.V[self.mapping[(s[0][0],s[0][1])]] / self.beta for s in successor_states])
            v_sum = np.sum(exp_values)

            # if we don't have enough info, random action
            if v_sum == 0:
                return self.env.unwrapped.random_action(), None

            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction
                action_probs[action] = np.exp(self.V[self.mapping[(new_state[0], new_state[1])]] / self.beta) / v_sum
            action = np.random.choice(self.env.action_space.n, p=action_probs)
            return action, None
        
        elif self.policy == "egreedy":
            # Explore: random action with probability epsilon
            if np.random.random() < self.epsilon:
                return self.env.unwrapped.random_action()

            # Exploit: greedy action with probability 1 - epsilon
            action_values = np.full(self.env.action_space.n, -np.inf)
            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction
                if np.any(np.all(self.target_locs == new_state, axis=1)):
                    return action
                action_values[action] = self.V[self.mapping[(new_state[0], new_state[1])]]
            return int(np.nanargmax(action_values))

        elif self.policy == "greedy":
            action_values = np.full(self.env.action_space.n, -np.inf)
            for action in self.env.unwrapped.get_available_actions(state):
                direction = self.env.unwrapped._action_to_direction[action]
                new_state = state + direction
                if np.any(np.all(self.target_locs == new_state, axis=1)):
                    return action

                if self.maze[new_state[0], new_state[1]] == "1":
                    continue
                action_values[action] = self.V[self.mapping[(new_state[0], new_state[1])]]

            return np.nanargmax(action_values)

    def take_action(self, state):
        """
        Select an action according to policy and take it. 
        """
        # Choose action
        if self.policy == "softmax":
            action, _ = self.select_action(state)
        else:
            action = self.select_action(state)

        # Take action
        obs, _, done, _, _ = self.env.step(action)

        # Unpack observation to get new state
        next_state = obs["agent"]
        next_state_idx = self.mapping[(next_state[0], next_state[1])]
        
        return next_state, next_state_idx, done

    def update(self, state_idx, next_state_idx):
        """
        Update the SR
        """
        # Update successor representation
        self.SR[state_idx] = (1 - self.alpha) * self.SR[state_idx] + self.alpha * (self.one_hot[state_idx] + self.gamma * self.SR[next_state_idx])

        # Ensure SR values stay reasonable
        self.SR = np.clip(self.SR, 0, 1/(1-self.gamma))

        # Update Values
        self.update_V()

    def learn(self, seed=None):
        """
        Agent explores the maze according to its decision policy and and updates its DR as it goes
        """
        # print(f"Decision Policy: {self.policy}, Number of Iterations: {self.num_steps}, lr={self.alpha}, temperature={self.beta}, importance sampling={self.imp_samp}")
        self.env.reset(seed=seed, options={})

        # Iterate through number of steps
        for i in range(self.num_steps):
            # Current state
            state = self.env.unwrapped.agent_loc
            state_idx = self.mapping[(state[0], state[1])]

            # Take action
            next_state, next_state_idx, done = self.take_action(state)

            # Update  SR
            self.update(state_idx, next_state_idx)

            if done:
                self.env.reset(seed=seed, options={})
                continue
            
            # Update state
            state = next_state


class SR_NHB:
    def __init__(self, alpha=0.1, beta=1, gamma=0.904, num_steps=25000, policy="random", exp_type="policy_reval"):
        # Hard code start and end locations as well as size
        self.start_loc = 0
        self.target_locs = [3,4,5]
        self.start_locs = [0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 0, 0, 1, 2, 1, 2, 0, 0, 1, 0, 0, 2, 1, 2, 0, 1, 2, 0]
        self.size = 6
        self.agent_loc = self.start_loc
        self.exp_type = exp_type

        # Construct the transition probability matrix and envstep functions
        self.T = self.construct_T()
        self.envstep = gen_nhb_exp_SR()

        # Get terminal states
        self.terminals = np.diag(self.T) == 1

        # Set reward
        self.reward_nt = 1
        self.r = np.full(len(self.T), self.reward_nt)
        # Reward of terminal states depends on if we are replicating reward revaluation or policy revaluation
        if self.exp_type == "policy_reval":
            self.r_term_1 = [0, 15, 30]
            self.r_term_2 = [45, 15, 30]
        elif self.exp_type == "reward_reval":
            self.r_term_1 = [15, 0, 30]
            self.r_term_2 = [45, 0, 30]
        else:
            print("Incorrect experiment type (exp_type)")
            return(0)
        self.r[self.terminals] = self.r_term_1

        # Params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_steps = num_steps
        self.policy = policy

        # Model
        self.SR = np.eye(self.size)
        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def construct_T(self):
        """
        Manually construct a T that is biased by a decision policy
        """
        T = np.zeros((6, 6))
        T[0,1] = 0.5
        T[0,2] = 0.5
        T[1,3] = 0.5
        T[1,4] = 0.5
        T[2,4] = 0.5
        T[2,5] = 0.5
        T[3:6, 3:6] = np.eye(3)

        return T
    
    def construct_T_new(self):
        """
        Manually construct a new transition matrix for the transition revaluation problem
        """
        T = np.zeros((6, 6))
        T[0, 1:3] = 0.5
        T[1, 4:6] = 0.5
        T[2, 3:5] = 0.5
        T[3:6, 3:6] = np.eye(3)

        return T
    
    def gen_new_envstep(self):
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

        # State 1 -> 4, 5
        envstep[1,0] = [4,1]
        envstep[1,1] = [5,1]

        # State 2 -> 3, 4
        envstep[2,0] = [3,1]
        envstep[2,1] = [4,1]
        
        return envstep
    
    def update_exp(self):
        """
        Update the terminal state values or transition matrix (experiment dependent)
        """
        if self.exp_type in ["reward_reval", "policy_reval"]:
            self.r[self.terminals] = self.r_term_2

    def update_V(self):
        self.V = self.SR @ self.r

    def get_successor_states(self, state):
        """
        Manually define the successor states based on which state we are in
        """
        return np.where(self.T[state, :] != 0)[0]

    def select_action(self, state):
        """
        Action selection based on our policy
        Options are: [random, softmax, egreedy, test]
        """
        if self.policy == "random":
            successor_states = self.get_successor_states(state)
            return np.random.choice([0,1])

        elif self.policy == "softmax":
            successor_states = self.get_successor_states(state)
            action_probs = np.full(2, 0.0)   # We can hardcode this because every state has 2 actions

            V = self.V[successor_states]
            exp_V = np.exp(V / self.beta)
            action_probs = exp_V / exp_V.sum()

            action = np.random.choice([0,1], p=action_probs)

            return action
    
    def decision_policy_SR(self):
        """
        Computes the SR decision policy, which acts as T^pi
        """
        T_pi = np.zeros_like(self.T)

        for state in [0,1,2]:
            # Get next states
            next_states = self.envstep[state][:,0]
            # Calculate softmax probabilities
            exp_values = np.exp(self.V[next_states] / self.beta)
            probs = exp_values / np.sum(exp_values)
            # Use probs to fill T_pi
            for i, next_state in enumerate(next_states):
                T_pi[state,next_state] += probs[i]

        T_pi[self.terminals, self.terminals] = 1
        
        return T_pi

    def learn_with_start_locs(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)
        
        for start_loc in self.start_locs:
            self.agent_loc = start_loc
            done = False

            while True:
                state = self.agent_loc
                action = self.select_action(state)

                # Take action
                next_state, done = self.envstep[state, action]

                self.SR[state] = (1-self.alpha)* self.SR[state] + self.alpha * ( self.one_hot[state] + self.gamma * self.SR[next_state]  )

                # Update Values
                self.update_V()

                if done:
                    break

                # Update state
                state = next_state
                self.agent_loc = state

        self.update_V()
    
    def learn_trans_reval(self, seed=None):
        # update T and envstep
        T_new = self.construct_T_new()
        new_envstep = self.gen_new_envstep()

        self.envstep = new_envstep
        self.T = T_new

        if seed is not None:
            np.random.seed(seed=seed)

        # 3 times from state 1 and state 2
        for start_loc in [1, 2]:
            self.agent_loc = start_loc
            state = self.agent_loc
            action = self.select_action(state)

            # Take action
            next_state, done = new_envstep[state, action]

            # Update SR
            self.SR[state] = (1-self.alpha) * self.SR[state] + self.alpha * ( self.one_hot[state] + self.gamma * self.SR[next_state]  )

            # Update Values
            self.update_V()

    def learn(self, seed=None):
        """
        Agent explores the maze according to its decision policy and and updates its DR as it goes
        """
        if seed is not None:
            np.random.seed(seed=seed)

        # Iterate through number of steps
        for i in range(self.num_steps):
            # Current state
            state = self.agent_loc
            action = self.select_action(state)
            # print(state, action)

            # Take action
            next_state, done = self.envstep[state, action]

            # Update SR
            self.SR[state] = (1-self.alpha)* self.SR[state] + self.alpha * ( self.one_hot[state] + self.gamma * self.SR[next_state]  )

            # Update Values
            self.update_V()

            if done:
                self.agent_loc = self.start_loc
                continue

            # Update state
            state = next_state
            self.agent_loc = state


class MB_NHB:
    def __init__(self, beta=1, policy="random", exp_type="policy_reval"):
        self.exp_type = exp_type
        self.size = 6
        
        # Construct the transition probability matrix and envstep functions
        self.T = self.construct_T()
        self.envstep = gen_nhb_exp_SR()

        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        self.nonterminals = ~self.terminals

        # Set reward
        self.reward_nt = 1
        self.r = np.full(len(self.T), self.reward_nt)
        # Reward of terminal states depends on if we are replicating reward revaluation or policy revaluation
        if self.exp_type == "policy_reval":
            self.r_term_1 = [0, 15, 30]
            self.r_term_2 = [45, 15, 30]
        elif self.exp_type == "reward_reval":
            self.r_term_1 = [15, 0, 30]
            self.r_term_2 = [45, 0, 30]
        else:
            print("Incorrect experiment type (exp_type)")
            return(0)
        self.r[self.terminals] = self.r_term_1
        
        # Params
        self.beta = beta
        self.policy = policy

        # Model
        self.V = np.zeros(self.size)

    def construct_T(self):
        T = np.zeros((6, 6))
        T[0,1] = 0.5
        T[0,2] = 0.5
        T[1,3] = 0.5
        T[1,4] = 0.5
        T[2,4] = 0.5
        T[2,5] = 0.5
        T[3:6, 3:6] = np.eye(3)

        return T
    
    def construct_T_new(self):
        T = np.zeros((6, 6))
        T[0, 1:3] = 0.5
        T[1, 4:6] = 0.5
        T[2, 3:5] = 0.5
        T[3:6, 3:6] = np.eye(3)

        return T
    
    def update_exp(self):
        if self.exp_type in ["reward_reval", "policy_reval"]:
            self.r[self.terminals] = self.r_term_2
            
    def gen_new_envstep(self):
        envstep=[]
        for s in range(6):
            # actions 0=left, 1=right
            envstep.append([[0,0], [0,0]])  # [s', done]
        envstep = np.array(envstep)

        # State 0 -> 1, 2
        envstep[0,0] = [1,0]
        envstep[0,1] = [2,0]

        # State 1 -> 4, 5
        envstep[1,0] = [4,1]
        envstep[1,1] = [5,1]

        # State 2 -> 3, 4
        envstep[2,0] = [3,1]
        envstep[2,1] = [4,1]
        
        return envstep
    
    def _get_succ_states(self, state):
        return np.where(self.T[state, :] != 0)[0]
    
    def select_action(self, state):
        if self.policy == "random":
            successor_states = self._get_succ_states(state)
            return np.random.choice([0,1])

        elif self.policy == "softmax":
            successor_states = self._get_succ_states(state)
            action_probs = np.full(2, 0.0)   # We can hardcode this because every state has 2 actions

            V = self.V[successor_states]
            exp_V = np.exp(V / self.beta)
            action_probs = exp_V / exp_V.sum()

            action = np.random.choice([0,1], p=action_probs)

            return action
    
    def update_V(self):
        self.V[self.terminals] = self.r[self.terminals]

        # Nonterminals: propagate backward
        for _ in range(self.size):
            for s in range(self.size):
                if self.nonterminals[s]:
                    # Find neighbors
                    neighbors = self._get_succ_states(s)
                    if len(neighbors) > 0:
                        # V[s] = max value of neighbors
                        self.V[s] = np.max(self.V[neighbors])
                        
    def learn_trans_reval(self, seed=None):
        # update T and envstep
        T_new = self.construct_T_new()
        new_envstep = self.gen_new_envstep()

        self.envstep = new_envstep
        self.T = T_new

        if seed is not None:
            np.random.seed(seed=seed)

        self.update_V()


class Hybrid_NHB:
    """
    Hybrid model combining SR (Successor Representation) and MB (Model-Based) approaches.

    The SR model is trained for num_steps, the MB model calculates values directly,
    and the final value is a weighted combination: V = w_sr * V_SR + w_mb * V_MB

    Args:
        alpha (float): Learning rate for SR model
        beta (float): Temperature parameter for softmax action selection
        gamma (float): Discount factor for SR model
        num_steps (int): Number of training steps for SR model
        policy (str): Decision policy ("random" or "softmax")
        exp_type (str): Experiment type ("policy_reval" or "reward_reval")
        w_sr (float): Weight for SR model values (default 0.5)
        w_mb (float): Weight for MB model values (default 0.5)
    """
    def __init__(self, alpha=0.1, beta=1, gamma=0.904, num_steps=25000, policy="random", exp_type="policy_reval", w_sr=0.5, w_mb=0.5):
        # Initialize SR model
        self.sr_model = SR_NHB(alpha=alpha, beta=beta, gamma=gamma, num_steps=num_steps, policy=policy, exp_type=exp_type)

        # Initialize MB model
        self.mb_model = MB_NHB(beta=beta, policy=policy, exp_type=exp_type)

        # Hybrid weights
        self.w_sr = w_sr
        self.w_mb = w_mb

        # Copy common attributes for convenience
        self.size = self.sr_model.size
        self.terminals = self.sr_model.terminals
        self.exp_type = exp_type
        self.T = self.sr_model.T
        self.r = self.sr_model.r
        self.envstep = self.sr_model.envstep

        # Combined value
        self.V = np.zeros(self.size)

    def update_exp(self):
        """Update both models for revaluation experiments."""
        self.sr_model.update_exp()
        self.mb_model.update_exp()
        # Update local reward reference
        self.r = self.sr_model.r

    def update_V(self):
        """Combine SR and MB values using weighted average."""
        self.sr_model.update_V()
        self.mb_model.update_V()
        self.V = self.w_sr * self.sr_model.V + self.w_mb * self.mb_model.V

    def learn(self, seed=None):
        """
        Train the hybrid model:
        1. Train SR model for num_steps
        2. Calculate MB values (no training needed)
        3. Combine values with weights
        """
        # Train the SR model
        self.sr_model.learn(seed=seed)

        # Calculate MB values (no training needed)
        self.mb_model.update_V()

        # Combine values
        self.update_V()

    def learn_with_start_locs(self, seed=None):
        """
        Train using predefined start locations:
        1. Train SR model with start locations
        2. Calculate MB values
        3. Combine values with weights
        """
        # Train the SR model
        self.sr_model.learn_with_start_locs(seed=seed)

        # Calculate MB values (no training needed)
        self.mb_model.update_V()

        # Combine values
        self.update_V()

    def learn_trans_reval(self, seed=None):
        """
        Handle transition revaluation:
        1. Update SR model with new transitions
        2. Update MB model with new transitions
        3. Combine values
        """
        # Update SR model for transition revaluation
        self.sr_model.learn_trans_reval(seed=seed)

        # Update MB model for transition revaluation
        self.mb_model.learn_trans_reval(seed=seed)

        # Update local references
        self.T = self.sr_model.T
        self.envstep = self.sr_model.envstep

        # Combine values
        self.update_V()

    def get_successor_states(self, state):
        """Get successor states (delegates to SR model)."""
        return self.sr_model.get_successor_states(state)

    def select_action(self, state):
        """Select action based on hybrid values."""
        if self.sr_model.policy == "random":
            return np.random.choice([0, 1])
        elif self.sr_model.policy == "softmax":
            successor_states = self.get_successor_states(state)
            V = self.V[successor_states]
            exp_V = np.exp(V / self.sr_model.beta)
            action_probs = exp_V / exp_V.sum()
            return np.random.choice([0, 1], p=action_probs)


class SR_IS_TwoStep:
    def __init__(self, alpha=0.25, beta=1.0, _lambda=1, num_steps=250, policy="softmax", imp_samp=True, seed=None):
        # Hard code start and end locations as well as size
        # Uses 11 states with auxiliary terminal states (like SR_IS_NHB)
        # States 0-6 are non-terminal, states 7-10 are terminal
        self.start_loc = 0
        self.target_locs = [7,8,9,10]
        self.start_locs = [0]
        self.size = 11
        self.agent_loc = self.start_loc
        self.seed = seed
        self.prob_common = 0.5
        self.rng = np.random.RandomState(seed)

        # Construct the transition probability matrix and envstep
        self.T = self.construct_T()
        self.envstep = self._build_envstep()

        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        # Calculate P = T_{NT}
        self.P = self.T[~self.terminals][:,self.terminals]

        # Set reward
        self.reward_nt = -0.1
        self.r = np.full(len(self.T), self.reward_nt)
        # Rewards on auxiliary terminal states 7,8,9,10
        self.r[self.terminals] = [10,-10,0,1]

        # Precalculate exp(r) for use with LinearRL equations
        self.expr_t = np.exp(self.r[self.terminals] / _lambda)
        self.expr_nt = np.exp(self.reward_nt / _lambda)

        # Params
        self.alpha = alpha
        self.beta = beta
        self.gamma = self.expr_nt
        self._lambda = _lambda
        self.num_steps = num_steps
        self.policy = policy
        self.imp_samp = imp_samp

        # Model
        self.DR = self.get_DR()
        self.Z = np.full(self.size, 0.01)

        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def construct_T(self):
        # For two-step task with auxiliary terminal states
        # State 0 -> (1,2), State 1 -> (3,4), State 2 -> (5,6)
        # State 3 -> 7, State 4 -> 8, State 5 -> 9, State 6 -> 10
        # States 7,8,9,10 are terminal
        T = np.zeros((self.size, self.size))
        T[0, 1:3] = 0.5
        T[1, 3:5] = 0.5
        T[2, 5:7] = 0.5
        T[3, 7] = 1
        T[4, 8] = 1
        T[5, 9] = 1
        T[6, 10] = 1
        T[7:11, 7:11] = np.eye(4)

        return T

    def _build_envstep(self):
        """Build the transition lookup table for 11 states."""
        envstep = []
        for s in range(self.size):
            envstep.append([[0, 0], [0, 0]])  # [next_state, done]
        envstep = np.array(envstep)

        # State 0 -> 1, 2 (deterministic)
        envstep[0, 0] = [1, 0]
        envstep[0, 1] = [2, 0]

        # State 1 -> 3, 4 (stochastic, handled in step())
        envstep[1, 0] = [3, 0]  # common for action 0
        envstep[1, 1] = [4, 0]  # common for action 1

        # State 2 -> 5, 6 (deterministic)
        envstep[2, 0] = [5, 0]
        envstep[2, 1] = [6, 0]

        # States 3-6 -> auxiliary terminal states (deterministic)
        envstep[3, 0] = [7, 1]
        envstep[3, 1] = [7, 1]
        envstep[4, 0] = [8, 1]
        envstep[4, 1] = [8, 1]
        envstep[5, 0] = [9, 1]
        envstep[5, 1] = [9, 1]
        envstep[6, 0] = [10, 1]
        envstep[6, 1] = [10, 1]

        return envstep

    def step_deterministic(self, state, action):
        """Get the deterministic (common) transition."""
        next_state, done = self.envstep[state, action]
        return next_state, done

    def step(self, state, action):
        """Take a step, handling stochastic transitions for state 1."""
        common_state, done = self.envstep[state, action]

        # Handle stochastic transitions for state 1
        if state == 1:
            if self.rng.random() < self.prob_common:
                next_state = common_state
            else:
                # Rare transition (flip to the other action's common state)
                rare_action = 1 - action
                next_state = self.envstep[state, rare_action][0]
        else:
            next_state = common_state

        return next_state, done

    def get_DR(self):
        if self.policy == "softmax":
            DR = np.full((self.size, self.size), 0.01)
            np.fill_diagonal(DR, 1)
            DR[np.where(self.terminals)[0], np.where(self.terminals)[0]] = (1/(self.gamma))
        else:
            DR = np.eye(self.size)

        return DR
    
    def update_exp(self):
        self.r[self.terminals] = [14,-10,0,1]

    def update_Z(self):
        self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t
        self.Z[self.terminals] = self.expr_t

    def update_V(self):
        self.V = np.log(self.Z) * self._lambda
    
    def get_successor_states(self, state):
        return np.where(self.T[state, :] != 0)[0]

    def importance_sampling(self, state, s_prob):
        successor_states = self.get_successor_states(state)
        p = 1/len(successor_states)
        w = p/s_prob
                
        return w

    def select_action(self, state):
        if self.policy == "random":
            action = np.random.choice([0,1])

            return action

        elif self.policy == "softmax":
            successor_states = self.get_successor_states(state)
            action_probs = np.full(2, 0.0)   # We can hardcode this because every state has 2 actions

            v_sum = sum(np.exp((np.log(self.Z[s] + 1e-20) * self._lambda) / self.beta) for s in successor_states)

            # if we don't have enough info, random action
            if v_sum == 0:
                return np.random.choice([0,1]), 1

            for action in [0,1]:
                new_state, done = self.step_deterministic(state, action)

                # If we hit a done state our action doesn't matter
                if done:
                    action = np.random.choice([0,1])
                    return action, 1
                action_probs[action] = np.exp((np.log(self.Z[new_state] + 1e-20) * self._lambda) / self.beta ) / v_sum

            action = np.random.choice([0,1], p=action_probs)
            s_prob = action_probs[action]

            return action, s_prob

    def get_D_inv(self):
        I = np.eye(self.size)
        D_inv = np.linalg.inv(I-self.gamma*self.T)
        
        return D_inv

    def learn(self):
        if self.seed is not None:
            np.random.seed(seed=self.seed)

        for i in range(self.num_steps):
            # Agent gets some knowledge of terminal state values
            if i == 2:
                self.Z[self.terminals] = self.expr_t
            state = self.agent_loc

            if self.policy == "softmax":
                action, s_prob = self.select_action(state)
            else:
                action = self.select_action(state)

            next_state, done = self.step(state, action)

            if self.imp_samp:
                w = self.importance_sampling(state, s_prob)
                w = 1 if np.isnan(w) or w == 0 else w
            else:
                w = 1

            target = self.one_hot[state] + self.gamma * self.DR[next_state]
            self.DR[state] = (1 - self.alpha) * self.DR[state] + self.alpha * target * w

            self.Z[~self.terminals] = self.DR[~self.terminals][:,~self.terminals] @ self.P @ self.expr_t

            if done:
                self.agent_loc = self.start_loc
                continue

            # Update state
            state = next_state
            self.agent_loc = state

        # Update DR at terminal state
        self.update_Z()
        self.update_V()


class SR_TwoStep:
    def __init__(self, alpha=0.25, beta=1.0, gamma=0.904, num_steps=250, policy="softmax", seed=None):
        # Hard code start and end locations as well as size
        self.start_loc = 0
        self.target_locs = [3,4,5,6]
        self.start_locs = [0]
        self.size = 7
        self.agent_loc = self.start_loc
        self.seed = seed

        # Construct the transition probability matrix and env
        self.T = self.construct_T()
        self.env = TwoStepStochastic(size=7, prob_common=0.5, seed=self.seed)
        
        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        # Calculate P = T_{NT}
        self.P = self.T[~self.terminals][:,self.terminals]

        # Set reward
        self.reward_nt = 1
        self.r = np.full(len(self.T), self.reward_nt)
        # Reward of terminal states depends on if we are replicating reward revaluation or policy revaluation
        self.r[self.terminals] = [10,-10,0,1]

        # Params
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_steps = num_steps
        self.policy = policy

        # Model
        self.SR = np.eye(self.size)
        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def construct_T(self):
        # For two-step task
        T = np.zeros((self.size, self.size))
        T[0, 1:3] = 0.5
        T[1, 3:5] = 0.5
        T[2, 5:7] = 0.5
        T[3:7, 3:7] = np.eye(4)

        return T

    def update_exp(self):
        self.r[self.terminals] = [14,-10,0,1]
    
    def update_V(self):
        self.V = self.SR @ self.r
    
    def get_successor_states(self, state):
        return np.where(self.T[state, :] != 0)[0]

    def select_action(self, state):
        if self.policy == "random":
            successor_states = self.get_successor_states(state)
            return np.random.choice([0,1])
        
        elif self.policy == "softmax":
            successor_states = self.get_successor_states(state)
            action_probs = np.full(2, 0.0)   # We can hardcode this because every state has 2 actions

            V = self.V[successor_states]
            exp_V = np.exp(V / self.beta)
            action_probs = exp_V / exp_V.sum()

            action = np.random.choice([0,1], p=action_probs)

            return action

    def learn(self, seed=None):
        if seed is not None:
            np.random.seed(seed=seed)

        for i in range(self.num_steps):
            state = self.agent_loc
            action = self.select_action(state)

            next_state, done = next_state, done = self.env.step(state, action)

            self.SR[state] = (1-self.alpha)* self.SR[state] + self.alpha * ( self.one_hot[state] + self.gamma * self.SR[next_state]  )

            self.update_V()

            if done:
                self.agent_loc = self.start_loc
                continue

            state = next_state
            self.agent_loc = state


class MB_TwoStep:
    def __init__(self, beta=1.0, gamma=1, policy="softmax", seed=None):
        # Hard code start and end locations as well as size
        self.start_loc = 0
        self.target_locs = [3,4,5,6]
        self.size = 7
        self.seed = seed

        # Construct the transition probability matrix and env
        self.T = self.construct_T()
        self.env = TwoStepStochastic(size=7, prob_common=0.5, seed=self.seed)
        
        # Get terminal states
        self.terminals = np.diag(self.T) == 1
        self.nonterminals = ~self.terminals
        # Calculate P = T_{NT}
        self.P = self.T[~self.terminals][:,self.terminals]

        # Set reward
        self.reward_nt = 1
        self.r = np.full(len(self.T), self.reward_nt)
        # Reward of terminal states depends on if we are replicating reward revaluation or policy revaluation
        self.r[self.terminals] = [10,-10,0,1]

        # Params
        self.beta = beta
        self.gamma = gamma
        self.policy = policy

        # Model
        self.SR = np.eye(self.size)
        self.V = np.zeros(self.size)
        self.one_hot = np.eye(self.size)

    def construct_T(self):
        # For two-step task
        T = np.zeros((self.size, self.size))
        T[0, 1:3] = 1.0
        T[1, 3:5] = 0.5
        T[2, 5] = 1.0
        T[2, 6] = 1.0
        T[3:7, 3:7] = np.eye(4)

        return T

    def update_exp(self):
        self.r[self.terminals] = [14,-10,0,1]
    
    def _get_succ_states(self, state):
        return np.where(self.T[state, :] != 0)[0]
    
    def update_V(self):
        self.V[self.terminals] = self.r[self.terminals]
        
        for s in range(self.size):
            if self.nonterminals[s]:
                neighbors = self._get_succ_states(s)
                
                if len(neighbors) > 0:
                    if s == 1:  # State 2 is stochastic
                        self.V[s] = 0.5 * self.V[3] + 0.5 * self.V[4]
                    else:  # States 0 and 2 are deterministic
                        self.V[s] = np.max(self.V[neighbors])

    def select_action(self, state):
        if self.policy == "random":
            successor_states = self._get_succ_states(state)
            return np.random.choice([0,1])
        
        elif self.policy == "softmax":
            successor_states = self._get_succ_states(state)
            action_probs = np.full(2, 0.0)   # We can hardcode this because every state has 2 actions

            V = self.V[successor_states]
            exp_V = np.exp(V / self.beta)
            action_probs = exp_V / exp_V.sum()

            action = np.random.choice([0,1], p=action_probs)

            return action


class Hybrid_TwoStep:
    def __init__(self, alpha=0.25, beta=1.0, gamma_sr=0.904, gamma_mb=1, num_steps=250, 
                 policy="softmax", w_sr=0.5, w_mb=0.5, seed=None):
        # SR model
        self.sr_model = SR_TwoStep(alpha=alpha, beta=beta, gamma=gamma_sr, 
                                   num_steps=num_steps, policy=policy, seed=seed)

        # MB model
        self.mb_model = MB_TwoStep(beta=beta, gamma=gamma_mb, policy=policy, seed=seed)

        self.w_sr = w_sr
        self.w_mb = w_mb

        self.size = self.sr_model.size
        self.terminals = self.sr_model.terminals
        self.T = self.sr_model.T
        self.r = self.sr_model.r
        self.env = self.sr_model.env
        self.seed = seed

        self.V = np.zeros(self.size)

    def update_exp(self):
        self.sr_model.update_exp()
        self.mb_model.update_exp()
        self.r = self.sr_model.r

    def update_V(self):
        self.sr_model.update_V()
        self.mb_model.update_V()
        self.V = self.w_sr * self.sr_model.V + self.w_mb * self.mb_model.V

    def learn(self, seed=None):
        self.sr_model.learn(seed=seed)

        self.mb_model.update_V()

        self.update_V()

    def get_successor_states(self, state):
        return self.sr_model.get_successor_states(state)

    def select_action(self, state):
        if self.sr_model.policy == "random":
            return np.random.choice([0, 1])
        
        elif self.sr_model.policy == "softmax":
            successor_states = self.get_successor_states(state)
            V = self.V[successor_states]
            exp_V = np.exp(V / self.sr_model.beta)
            action_probs = exp_V / exp_V.sum()
            return np.random.choice([0, 1], p=action_probs)