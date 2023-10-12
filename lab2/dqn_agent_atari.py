import torch
import torch.nn as nn
from base_agent import DQNBaseAgent
from select_topic_RL.lab2.models.atari_model import AtariNetDQN
import gym
import random


class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)

		self.env = gym.make(config["env_id"], render_mode="rgb_array")
		self.env = gym.wrappers.AtariPreprocessing(env=self.env, frame_skip=1, terminal_on_life_loss=True)
		self.env = gym.wrappers.FrameStack(env=self.env, num_stack=4)

		self.test_env = gym.make(config["env_id"], render_mode="rgb_array")
		self.test_env = gym.wrappers.AtariPreprocessing(env=self.test_env, frame_skip=1, terminal_on_life_loss=False)
		self.test_env = gym.wrappers.FrameStack(env=self.test_env, num_stack=4)

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())

		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)

	def decide_agent_actions(self, observation, epsilon=0.0, action_space=None) -> int:
		if random.random() <= epsilon:
			return random.randrange(self.env.action_space.n)
		else:
			action = self.behavior_net.forward([observation], self.device).argmax().cpu().item()
			return action

	def choose_batch_actions(self, net, states) -> torch.Tensor:
		if net == "behavior":
			_, actions = self.behavior_net.forward(states, self.device).max(dim=1, keepdim=True)
			return actions
		elif net == "target":
			_, actions = self.target_net.forword(states, self.device).max(dim=1, keepdim=True)
			return actions

	def Q_value(self, net, states, actions):
		# get Q values from actions and target network
		actions = actions.type(torch.int64)   # to solve "gather(): Expected dtype int64 for index" error
		if net == "target":
			next_q_values = self.target_net.forward(states, self.device).gather(dim=1, index=actions)
			return next_q_values
		elif net == "behavior":
			q_values = self.behavior_net.forward(states, self.device).gather(dim=1, index=actions)
			return q_values

	# implement DDQN
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		next_actions = self.choose_batch_actions("behavior", next_state)
		Q_target = reward + self.gamma * self.Q_value("target", next_state, next_actions)
		Q_output = self.Q_value("behavior", state, action)

		criterion = nn.SmoothL1Loss()
		loss = criterion(Q_output, Q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()
	