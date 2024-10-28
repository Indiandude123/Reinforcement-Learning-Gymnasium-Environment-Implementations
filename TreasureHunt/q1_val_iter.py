import numpy as np
from tqdm import tqdm
from env import TreasureHunt

class ValueIteration:
    def __init__(self, env, discount_factor=1.0, theta=0.105):
        self.env = env
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.theta = theta  # A small threshold for stopping the iteration
        self.value_function = np.zeros(self.env.num_states)  # Initialize value function

    def run_value_iteration(self):
        while True:
            delta = 0
            for state in range(self.env.num_states):
                v = self.value_function[state]
                # Update the value function based on the Bellman equation
                self.value_function[state] = max(
                    sum(
                        self.env.T[state, action, next_state] * 
                        (self.env.reward[next_state] + self.discount_factor * self.value_function[next_state]) 
                        for next_state in range(self.env.num_states)
                    ) 
                    for action in range(self.env.num_actions)
                )
                delta = max(delta, abs(v - self.value_function[state]))

            print(f"Value function: {self.value_function}")  # Debugging print
            print(f"Delta: {delta}")  # Debugging print
            
            if delta < self.theta:
                break

    def extract_policy(self):
        policy = np.zeros((self.env.num_states, self.env.num_actions))  # Initialize policy
        for state in range(self.env.num_states):
            action_values = np.zeros(self.env.num_actions)
            for action in range(self.env.num_actions):
                action_values[action] = sum(
                    self.env.T[state, action, next_state] * 
                    (self.env.reward[next_state] + self.discount_factor * self.value_function[next_state])
                    for next_state in range(self.env.num_states)
                )
            best_action = np.argmax(action_values)
            policy[state][best_action] = 1.0  # Deterministic policy
        return policy

# Usage example
locations = {
    'ship': [(0, 0)],
    'land': [(3, 0), (3, 1), (3, 2), (4, 2), (4, 1), (5, 2), (0, 7), (0, 8), (0, 9), (1, 7), (1, 8), (2, 7)],
    'fort': [(9, 9)],
    'pirate': [(4, 7), (8, 5)],
    'treasure': [(4, 0), (1, 9)]
}

env = TreasureHunt(locations)
value_iteration = ValueIteration(env)
value_iteration.run_value_iteration()

# Extract the resulting policy after value iteration
optimal_policy = value_iteration.extract_policy()

env.visualize_policy(optimal_policy, path='value_iteration_viz.png')
env.visualize_policy_execution(optimal_policy, path='value_iteration_viz.gif')
