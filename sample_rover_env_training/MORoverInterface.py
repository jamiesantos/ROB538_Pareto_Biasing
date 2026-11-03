import yaml
import torch
import numpy as np

from Policy import Policy
from MORoverEnv import MORoverEnv

class MORoverInterface():
    def __init__(self, rover_config_filename):
        """
        Initialise the MOROverInterface class with its instance of the MOROverEnv Domain.
        Setup an internal reference to the rover config file
        """
        self.rover_env = MORoverEnv(rover_config_filename)
        with open(rover_config_filename, 'r') as config_file:
            self.config = yaml.safe_load(config_file)
    
    # to perform a key-wise sum of two dicts
    def _keywise_sum(self, dict1, dict2):
        return {key: dict1.get(key, 0) + dict2.get(key, 0) for key in set(dict1) | set(dict2)}

    def rollout(self, joint_policy: list):
        """
        Perform a rollout of a given multiheaded actor in the MORoverEnv domain.

        Parameters:
        - joint_policy: list of policies, representing the whole team

        Returns:
        - rollout_trajectory (dict): Complete trajectory of the rollout with position, and action data of each agent.
        - global_reward (list): Reward vector that evaluates this joint_policy on each system-level objective.
        """
        if not (isinstance(joint_policy, list) and isinstance(p, Policy) for p in joint_policy):
            raise ValueError("The supplied joint policy should be a list of Policy type objects")

        ep_length = self.rover_env.get_ep_length()
        agent_locations = self.config['Agents']['starting_locs']  # set each agent to the starting location
        num_sensors = self.config['Agents']['num_sensors']
        observation_radii = self.config['Agents']['observation_radii']
        max_step_sizes = self.config['Agents']['max_step_sizes']
        
        cumulative_global_reward = {}  # Initialize cumulative global reward

        rollout_trajectory = [[] for _ in range(len(joint_policy))] # List of list of dicts

        self.rover_env.reset() # reset the rover env

        for t in range(ep_length):
            # get each agent's observation at the current position
            observations_list = self.rover_env.generate_observations(agent_locations, 
                                                                     num_sensors, 
                                                                     observation_radii, 
                                                                     normalise=True) 
            
            joint_action = [] # Will store the combined joint action

            # get each agent's move based on corresponding observation
            for i, policy in enumerate(joint_policy):
                # extract singe agent's observation
                agent_observation = observations_list[i]
                agent_observation_tensor = torch.FloatTensor(agent_observation)
                # get policy output based on observation
                action_tensor = policy.forward(agent_observation_tensor)
                # Ensure actions are clipped to [-1, 1]
                action_tensor = torch.clamp(action_tensor, -1.0, 1.0)
                 # Convert to a numpy array without tracking gradient
                action = action_tensor.squeeze(0).detach().numpy()

                # Scale the action to comply with the agent's max step size
                norm = np.linalg.norm(action) # get the magnitude of the calculated move
                max_step = max_step_sizes[i]
                scaling_factor = (max_step / norm) if norm > 0 else 0 # the factor by which the moves should be scaled
                scaled_action = action * scaling_factor # multiply each member of the action by the scaling factor

                # Add scaled action to the list of agent moves
                joint_action.append(scaled_action)

                # Add the agent's transition to the trajectory
                rollout_trajectory[i].append(
                    {
                        'state' : agent_observation,
                        'action' : action,
                        'position': agent_locations[i],  # Store the actual position
                    }
                )

            # get updated agent positions based on the joint action
            agent_locations = self.rover_env.update_agent_locations(agent_locations=agent_locations, 
                                                                    agent_deltas=joint_action, 
                                                                    max_step_sizes=max_step_sizes)
            
            # Get the global reward and update the cumulative global reward
            global_reward = self.rover_env.get_global_rewards(rov_locations=agent_locations, timestep=t)
            cumulative_global_reward = self._keywise_sum(cumulative_global_reward, global_reward)

        return rollout_trajectory, cumulative_global_reward

    # Function that evaluates a given trajectory for global rewards (without rollout)
    def evaluate_trajectory(self, traj: dict):
        parsed_trajectory = [
            [agent['position'] for agent in timestep]
            for timestep in zip(*traj)
        ] # NOTE: this only works for 2-dimensional environments

        self.rover_env.reset() # reset the rover env

        cumulative_global_reward = {obj: 0 for obj in range(self.rover_env.num_objs)} # Initialize cumulative global reward
        for t, agent_locations in enumerate(parsed_trajectory):
            global_reward = self.rover_env.get_global_rewards(rov_locations=agent_locations, timestep=t)
            cumulative_global_reward = self._keywise_sum(cumulative_global_reward, global_reward)
        
        return cumulative_global_reward

    # Function to get domain-specific information for the algorithm
    def get_state_size(self):
        '''
        Get the number of inputs to the actor network.
        '''
        return self.config['Agents']['num_sensors'][0] * (self.get_num_objs()+1) + len(self.rover_env.dimensions)*int(self.rover_env.include_location_in_obs)
    
    def get_action_size(self):
        '''
        Get the number of outputs per agent for the actor network.
        '''
        return len(self.rover_env.dimensions)

    def get_team_size(self):
        '''
        Get the number of agents that must be deployed in a rollout. Team size.
        '''
        # team size is implicitly recorded by the length of starting locations
        return len(self.config['Agents']['starting_locs'])
    
    def get_num_objs(self):
        '''
        Get the number of objectives in the problem.
        '''
        return self.rover_env.num_objs