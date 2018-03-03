import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from env import Env
from buffers import Buffer
from matplotlib import pyplot as plt

flags = tf.app.flags
flags.DEFINE_boolean("DDQN", True, "if true uses double DQN else uses DQN algorithm")
flags.DEFINE_string("Her", "future",
				 "different strategies of choosing goal. Possible values are :- future, final, episode or None. If None HER is not used")
flags.DEFINE_integer("bit_size", 15, "the bit size")
flags.DEFINE_integer("num_epochs", 5, "number of epochs")

FLAGS = flags.FLAGS


class Model(object) :

	def __init__(self,bit_size,scope,reuse) :

		hidden_dim = 256
		with tf.variable_scope(scope,reuse = reuse) :
			self.inp = tf.placeholder(shape = [None,2*bit_size],dtype = tf.float32)
			net = self.inp
			net = slim.fully_connected(net,hidden_dim,activation_fn = tf.nn.relu)
			self.out = slim.fully_connected(net,bit_size,activation_fn = None)
			self.predict = tf.argmax(self.out,axis = 1)
			self.action_taken = tf.placeholder(shape = [None],dtype = tf.int32)
			action_one_hot = tf.one_hot(self.action_taken,bit_size)
			Q_val = tf.reduce_sum(self.out*action_one_hot,axis = 1)
			self.Q_target = tf.placeholder(shape = [None],dtype = tf.float32)
			self.loss = tf.reduce_mean(tf.square(Q_val - self.Q_target))
			self.train_step = tf.train.AdamOptimizer(learning_rate = 1e-3).minimize(self.loss)


def update_target_graph(from_scope,to_scope,tau) :
	from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
	to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)
	ops = []
	for (var1,var2) in zip(from_vars,to_vars) :
		ops.append(var2.assign(var2*tau + (1-tau)*var1))

	return ops	

def updateTarget(ops,sess) :
	for op in ops :
		sess.run(op)

bit_size = FLAGS.bit_size
tau = 0.95
buffer_size = 1e6
batch_size = 128

optimisation_steps = 40
num_epochs = FLAGS.num_epochs
num_cycles = 50
num_episodes = 16
K = 4
gamma = 0.98

bit_env = Env(bit_size)
replay_buffer = Buffer(buffer_size,batch_size)

model = Model(bit_size,scope = 'model',reuse = False)
target_model = Model(bit_size,scope = 'target_model',reuse = False)

update_ops_initial = update_target_graph('model','target_model',tau = 0.0)
update_ops = update_target_graph('model','target_model',tau = tau)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

#making the target network same as main network
updateTarget(update_ops_initial,sess)

total_loss = []
success_rate = []

for i in range(num_epochs) :
	for j in range(num_cycles) :
		total_reward = 0.0
		successes = []
		for k in range(num_episodes) :
			state,goal_state = bit_env.reset()
			episode_experience = []
			succeeded = False
			for t in range(bit_size) :
				inp_state = np.concatenate([state,goal_state],axis = -1)
				action = sess.run(model.predict,feed_dict = {model.inp : [inp_state]})[0]
				next_state,reward,done = bit_env.step(action)
				episode_experience.append((state,action,reward,next_state,goal_state))
				total_reward+=reward
				
				state = next_state
				if done :
					if succeeded :
						continue
					else :
						succeeded = True
						
			successes.append(succeeded)
						
			for t in range(bit_size) :
				s,a,r,s_,g = episode_experience[t]
				inputs = np.concatenate([s,g],axis = -1)
				inputs_ = np.concatenate([s_,g],axis = -1)
				replay_buffer.add(inputs,a,r,inputs_)
				if not FLAGS.Her == 'None':

					if not FLAGS.Her == 'final' :
						for h in range(K) :
							if FLAGS.Her == 'future' :
								future = np.random.randint(t,bit_size)
							else :
								future = np.random.randint(0,bit_size)	
							g_ = episode_experience[future][3]
							new_inputs = np.concatenate([s,g_],axis = -1)
							new_inputs_ = np.concatenate([s_,g_],axis = -1)
							if (np.array(s_) == np.array(g_)).all():
								r_ = 0.0
							else :
								r_ = -1.0

							replay_buffer.add(new_inputs,a,r_,new_inputs_)
					else :
						g_ = episode_experience[bit_size - 1][3]
						new_inputs = np.concatenate([s,g_],axis = -1)
						new_inputs_ = np.concatenate([s_,g_],axis = -1)
						if (np.array(s_) == np.array(g_)).all():
							r_ = 0.0
						else :
							r_ = -1.0
						replay_buffer.add(new_inputs,a,r_,new_inputs_)

					
		losses = []			
		for k in range(optimisation_steps) :
			state,action,reward,next_state = replay_buffer.sample()				
			target_net_Q = sess.run(target_model.out,feed_dict = {target_model.inp : next_state})

			if FLAGS.DDQN :
				main_net_predict = sess.run(model.predict,feed_dict = {model.inp : next_state})	
				doubleQ = np.reshape(target_net_Q[range(main_net_predict.shape[0]),main_net_predict],[-1])
				target_reward = np.clip(np.reshape(reward,[-1]) + gamma * doubleQ,-1. / (1 - gamma), 0)
			else :
				target_reward = np.clip(np.reshape(reward,[-1]) + gamma * np.reshape(np.max(target_net_Q,axis = -1),[-1]),-1. / (1 - gamma), 0)

			_,loss = sess.run([model.train_step,model.loss],feed_dict = {model.inp : state, model.action_taken : np.reshape(action,[-1]), model.Q_target : target_reward})
			losses.append(loss)

		losses = np.array(losses)	
		
		updateTarget(update_ops,sess)
		success_rate.append(np.mean(successes))


		print('after %d cycles, reward is %g, successes ratio is %g and mean loss is %g'%(i*num_cycles+j,total_reward,np.mean(successes),np.mean(losses)))
		

plt.plot(success_rate)
plt.xlabel('number of cycles')
plt.ylabel('Success ratio')
plt.title('Progress of success ratio with episode cycles')
plt.show()
