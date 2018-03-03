import numpy as np

class Env(object) :
	
	def __init__(self,bits) :
		self.bits = bits

	def reset(self) :
		self.state = np.random.randint(0,2,size = self.bits)
		self.goal_state = np.random.randint(0,2,size = self.bits)
		while (self.state == self.goal_state).all() :
			self.goal_state = np.random.randint(0,2,size = self.bits)
		return self.state, self.goal_state

	#the action is a number from 0 to self.bits	
	def step(self,action) :
		self.state = np.copy(self.state)
		self.state[action] = not  self.state[action]
		reward = -1.0
		done = False
		if (self.state == self.goal_state).all() :
			done = True
			reward = 0.0

		return np.copy(self.state),reward,done
