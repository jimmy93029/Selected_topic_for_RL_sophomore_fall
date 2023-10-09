import torch
import torch.nn as nn
from base_agent import DQNBaseAgent
from select_topic_RL.lab2.models.atari_model import AtariNetDQN
import gym
import random


class AtariDQNAgent(DQNBaseAgent):
	def __init__(self, config):
		super(AtariDQNAgent, self).__init__(config)

		### TODO ###
		# initialize env
		self.env = gym.make(config["env_id"])

		### TODO ###
		# initialize test_env
		self.test_env = gym.make(config["env_id"])

		# initialize behavior network and target network
		self.behavior_net = AtariNetDQN(self.env.action_space.n)
		self.behavior_net.to(self.device)
		self.target_net = AtariNetDQN(self.env.action_space.n)
		self.target_net.to(self.device)
		self.target_net.load_state_dict(self.behavior_net.state_dict())

		# initialize optimizer
		self.lr = config["learning_rate"]
		self.optim = torch.optim.Adam(self.behavior_net.parameters(), lr=self.lr, eps=1.5e-4)
		
	def decide_agent_action(self, observation, epsilon=0.0, action_space=None) -> int:
		### TODO ###
		# get action from "behavior net", with epsilon-greedy selection  // 要以 pytorch 的形式去做
		if random.random() <= epsilon:
			return random.randrange(self.env.action_space.n)
		else:
			action = self.behavior_net(observation).argmax().cpu().item()
			return action

	def choose_batch_actions(self, net, states) -> torch.Tensor:
		# In the double DQN, we will use "behavior" network Q(s, a) to choose a batch of action
		# these actions will be put into target network Q' e.t.c  Q_target = Q'(s, argmax<Q(s, a)>)
		if net == "behavior":
			_, actions = self.behavior_net(states).max(dim=1, keepdim=True)
			return actions
		elif net == "target":
			_, actions = self.target_net(states).max(dim=1, keepdim=True)
			return actions

	def Q_value(self, net, states, actions):
		# get Q values from actions and target network
		if net == "target":
			next_q_values = self.target_net.forward(states).gather(dim=1, index=actions)
			return next_q_values
		elif net == "behavior":
			q_values = self.behavior_net.forward(states).gather(dim=1, index=actions)
			return q_values

	# implement DDQN
	def update_behavior_network(self):
		# sample a minibatch of transitions
		state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.device)

		### TODO ###
		# calculate the loss and update the behavior network
		# 1. get Q(s,a) from behavior net
		# 2. get max_a Q(s',a) from target net
		# 3. calculate Q_target = r + gamma * max_a Q(s',a)
		# 4. calculate loss between Q(s,a) and Q_target
		# 5. update behavior net

		next_actions = self.choose_batch_actions("behavior", next_state)
		Q_target = reward + self.gamma * self.Q_value("target", next_state, next_actions)
		Q_output = self.Q_value("behavior", state, action)

		# Smooth L1 loss is something like Huber loss
		criterion = nn.SmoothL1Loss()
		loss = criterion(Q_output, Q_target)

		self.writer.add_scalar('DQN/Loss', loss.item(), self.total_time_step)

		self.optim.zero_grad()
		loss.backward()
		self.optim.step()

	