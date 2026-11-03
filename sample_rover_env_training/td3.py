import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from MORoverInterface import MORoverInterface
import ExpUtils.DataLogger

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477
#
# Multi-agent version for the rover domain: every rover has its own actor that acts on its own
# observation, and a single critic shared by the team scores the joint state and joint action
# against the global (team) reward. The critic is only used in training, so the trained rovers
# act independently.

class Actor(nn.Module):
    """
    Policy for a single rover: maps the rover's observation to an action in [-1, 1].
    """

    def __init__(self, obs_dim, action_dim, hidden_size=64):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, action_dim)

    def forward(self, obs):
        a = F.relu(self.l1(obs))
        a = F.relu(self.l2(a))
        return torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3(object):
    def __init__(
        self,
        num_agents,
        obs_dim,
        action_dim,
        state_dim,
        max_action=1.0,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.num_agents = num_agents
        self.obs_dim = obs_dim

        # One actor per rover
        self.actors = [Actor(obs_dim, action_dim) for _ in range(num_agents)]
        self.actor_targets = [copy.deepcopy(actor) for actor in self.actors]
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=3e-4) for actor in self.actors]

        # One critic shared by the team, scoring the joint state and the joint action
        self.critic = Critic(state_dim, num_agents * action_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def joint_action(self, actors, state):
        """
        Concatenate every rover's action, with each rover acting on its own observation in the joint state.
        """
        return torch.cat([actor(state[:, i * self.obs_dim:(i + 1) * self.obs_dim]) for i, actor in enumerate(actors)], dim=1)

    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.joint_action(self.actor_targets, next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute the actor loss with the whole team acting from its current policies. Each
            # actor only produces its own slot of the joint action, so it only gets its own gradient.
            actor_loss = -self.critic.Q1(state, self.joint_action(self.actors, state)).mean()

            # Optimize the actors
            for actor_optimizer in self.actor_optimizers:
                actor_optimizer.zero_grad()
            actor_loss.backward()
            for actor_optimizer in self.actor_optimizers:
                actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for actor, actor_target in zip(self.actors, self.actor_targets):
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

        for i, (actor, actor_optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
            torch.save(actor.state_dict(), filename + f"_actor_{i}")
            torch.save(actor_optimizer.state_dict(), filename + f"_actor_{i}_optimizer")


    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic", weights_only=True))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer", weights_only=True))
        self.critic_target = copy.deepcopy(self.critic)

        for i, (actor, actor_optimizer) in enumerate(zip(self.actors, self.actor_optimizers)):
            actor.load_state_dict(torch.load(filename + f"_actor_{i}", weights_only=True))
            actor_optimizer.load_state_dict(torch.load(filename + f"_actor_{i}_optimizer", weights_only=True))
        self.actor_targets = [copy.deepcopy(actor) for actor in self.actors]

class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        assert self.size >= batch_size, "Batch size larger than experiences in the replay buffer"
        return (
            torch.FloatTensor(self.state[ind]),
            torch.FloatTensor(self.action[ind]),
            torch.FloatTensor(self.next_state[ind]),
            torch.FloatTensor(self.reward[ind]),
            torch.FloatTensor(self.not_done[ind])
        )

class ExplorationPolicy:
    """
    Wraps an actor for MORoverInterface.rollout, adding Gaussian noise to its actions so the team explores.
    """

    def __init__(self, actor, noise_std):
        self.actor = actor
        self.noise_std = noise_std

    def forward(self, obs):
        with torch.no_grad():
            action = self.actor(obs)
        return action + torch.randn_like(action) * self.noise_std

class RandomPolicy:
    """
    Uniformly random actions for MORoverInterface.rollout, used to fill the replay buffer before training.
    """

    def __init__(self, action_dim):
        self.action_dim = action_dim

    def forward(self, obs):
        return torch.rand(self.action_dim) * 2 - 1

def joint_state(observations, t, ep_length):
    """
    State for the shared critic: every rover's observation in team order, followed by the fraction
    of the episode that has elapsed, so the critic can tell how much of the fixed-length episode is left.
    """
    return np.concatenate([np.ravel(observations), [t / ep_length]])

def add_trajectory(replay_buffer, interface, trajectory, episode_reward):
    """
    Add the joint transitions of a trajectory returned by MORoverInterface.rollout to the replay buffer.

    The rollout scores step t on where the rovers are after that step's move, which is the position
    logged at step t + 1, so the global reward of each step is recomputed from the logged positions.
    The positions after the last move are not logged, so the last step gets the rest of the episode reward.
    """
    ep_length = len(trajectory[0])

    interface.rover_env.reset()
    rewards = []
    for t in range(ep_length - 1):
        positions = [agent_traj[t + 1]['position'] for agent_traj in trajectory]
        rewards.append(interface.rover_env.get_global_rewards(rov_locations=positions, timestep=t)[0])
    rewards.append(episode_reward - sum(rewards))

    states = [joint_state([agent_traj[t]['state'] for agent_traj in trajectory], t, ep_length) for t in range(ep_length)]
    for t in range(ep_length):
        done = t == ep_length - 1 # every episode ends after ep_length steps
        replay_buffer.add(
            state=states[t],
            action=np.concatenate([agent_traj[t]['action'] for agent_traj in trajectory]),
            next_state=states[t] if done else states[t + 1], # never bootstrapped from on the last step
            reward=rewards[t],
            done=float(done)
        )

class TD3Trainer:
    """
    Trains a team of rovers with TD3. Each episode is collected with MORoverInterface.rollout and added to
    the replay buffer, then the shared critic and every actor take one gradient step per environment step.
    """

    def __init__(self, alg_config_filename, rover_config_filename, data_filename):
        self.config_filename = alg_config_filename
        self._read_config()

        self.data_logger = ExpUtils.DataLogger.DataLogger(data_fields=['episode',
                                                                       'steps',
                                                                       'noisy_fitness',
                                                                       'eval_fitness',
                                                                       'trajectory'],
                                                                       target_filename=data_filename)

        self.interface = MORoverInterface(rover_config_filename)
        if self.interface.get_num_objs() != 1:
            raise ValueError(f"[TD3] Expected 1 objective, but the environment has {self.interface.get_num_objs()}.")

        self.team_size = self.interface.get_team_size()
        self.action_size = self.interface.get_action_size()
        self.ep_length = self.interface.rover_env.get_ep_length()
        state_size = self.team_size * self.interface.get_state_size() + 1 # every rover's observation and the elapsed fraction of the episode

        self.td3 = TD3(num_agents=self.team_size,
                       obs_dim=self.interface.get_state_size(),
                       action_dim=self.action_size,
                       state_dim=state_size,
                       discount=self.discount,
                       tau=self.tau,
                       policy_noise=self.policy_noise,
                       noise_clip=self.noise_clip,
                       policy_freq=self.policy_freq)
        self.replay_buffer = ReplayBuffer(state_size, self.team_size * self.action_size)

        # Fill the replay buffer with random actions before training
        for _ in range(self.warmup_episodes):
            self._collect_episode([RandomPolicy(self.action_size) for _ in range(self.team_size)])
        self.env_steps = self.warmup_episodes * self.ep_length

    def _read_config(self):
        """Read and load TD3 configuration from the YAML file."""
        with open(self.config_filename, 'r') as config_file:
            self.config_data = yaml.safe_load(config_file)
            print('[TD3]: YAML config read.')
        self._load_config()

    def _load_config(self):
        """Load the training schedule and TD3 hyperparameters."""
        self.num_episodes = self.config_data['TD3']['num_episodes']
        self.warmup_episodes = self.config_data['TD3']['warmup_episodes']
        self.expl_noise = self.config_data['TD3']['expl_noise']
        self.batch_size = self.config_data['TD3']['batch_size']
        self.discount = self.config_data['TD3']['discount']
        self.tau = self.config_data['TD3']['tau']
        self.policy_noise = self.config_data['TD3']['policy_noise']
        self.noise_clip = self.config_data['TD3']['noise_clip']
        self.policy_freq = self.config_data['TD3']['policy_freq']
        self.print_freq = self.config_data['TD3']['print_freq']

    def _collect_episode(self, joint_policy):
        """Roll out one episode with joint_policy, add it to the replay buffer and return its fitness."""
        trajectory, fitness_dict = self.interface.rollout(joint_policy)
        fitness = float(fitness_dict[0])
        add_trajectory(self.replay_buffer, self.interface, trajectory, fitness)
        return fitness

    def train_episode(self, episode, traj_write_freq=100):
        """
        Collect one episode with exploration noise on every actor, then train the shared critic and the
        actors with one gradient step per environment step. Logs the fitness of the collected episode and
        of the team acting without noise after the update.
        """
        fitness = self._collect_episode([ExplorationPolicy(actor, self.expl_noise) for actor in self.td3.actors])
        self.env_steps += self.ep_length

        for _ in range(self.ep_length):
            self.td3.train(self.replay_buffer, batch_size=self.batch_size)

        # Evaluate the team without exploration noise
        with torch.no_grad():
            trajectory, eval_fitness_dict = self.interface.rollout(self.td3.actors)
        eval_fitness = float(eval_fitness_dict[0])

        self.data_logger.add_data(key='episode', value=episode)
        self.data_logger.add_data(key='steps', value=self.env_steps)
        self.data_logger.add_data(key='noisy_fitness', value=[fitness])
        self.data_logger.add_data(key='eval_fitness', value=[eval_fitness])
        if episode == self.num_episodes - 1 or episode % traj_write_freq == 0:
            self.data_logger.add_data(key='trajectory', value=trajectory)
        else:
            self.data_logger.add_data(key='trajectory', value=None)
        self.data_logger.write_data()

        if episode == self.num_episodes - 1 or episode % self.print_freq == 0:
            print(f"[TD3] episode {episode}  steps {self.env_steps}  noisy_fitness {fitness:.3f}  eval_fitness {eval_fitness:.3f}")
