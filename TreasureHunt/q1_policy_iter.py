import numpy as np
from tqdm import tqdm
from env import TreasureHunt


class PolicyIteration:
    def __init__(self, env, discount_factor=1.0, theta=0.105):
        self.env = env
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.theta = theta  # A small threshold for stopping the iteration
        self.policy = np.zeros((self.env.num_states, self.env.num_actions))  # Initialize policy to random actions
        self.value_function = np.zeros(self.env.num_states)  # Initialize value function

    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.env.num_states):
                v = self.value_function[state]
                # Calculate the value function for the current policy
                self.value_function[state] = sum(
                    self.policy[state, action] * 
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


    def policy_improvement(self):
        is_policy_stable = True
        for state in range(self.env.num_states):
            old_action = np.argmax(self.policy[state])
            # Find the best action to take for the current state
            action_values = np.zeros(self.env.num_actions)
            for action in range(self.env.num_actions):
                action_values[action] = sum(
                    self.env.T[state, action, next_state] * 
                    (self.env.reward[next_state] + self.discount_factor * self.value_function[next_state])
                    for next_state in range(self.env.num_states)
                )
            best_action = np.argmax(action_values)
            # Update policy based on the best action
            self.policy[state] = np.zeros(self.env.num_actions)
            self.policy[state][best_action] = 1.0  # Deterministic policy
            if old_action != best_action:
                is_policy_stable = False
        return is_policy_stable

    def run_policy_iteration(self):
        while True:
            self.policy_evaluation()  # Evaluate the current policy
            if self.policy_improvement():  # Improve the policy
                break

# Usage example
# Assuming locations are defined elsewhere as shown in your environment code
locations = {
    'ship': [(0, 0)],
    'land': [(3, 0), (3, 1), (3, 2), (4, 2), (4, 1), (5, 2), (0, 7), (0, 8), (0, 9), (1, 7), (1, 8), (2, 7)],
    'fort': [(9, 9)],
    'pirate': [(4, 7), (8, 5)],
    'treasure': [(4, 0), (1, 9)]
}

env = TreasureHunt(locations)
policy_iteration = PolicyIteration(env)
policy_iteration.run_policy_iteration()

# The resulting policy can be accessed as:
optimal_policy = policy_iteration.policy
env.visualize_policy(optimal_policy, path='policy_iteration_viz.png')

env.visualize_policy_execution(optimal_policy, path='policy_iteration_viz.gif')
