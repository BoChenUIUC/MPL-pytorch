import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from test import Simulator
from compression.transformer import Transformer
from compression.ddpgtrain import Trainer
from compression.ddpgbuffer import MemoryBuffer
from sortedcontainers import SortedDict
from tqdm import tqdm

# setup
classes_num = 24
batch_size = 1
print_step = 1
eval_step = 1
PATH = 'backup/rsnet.pth'

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)

class RSNet(nn.Module):
	def __init__(self):
		super(RSNet, self).__init__()
		EPS = 0.003
		self.fc1 = nn.Linear(6,256)
		self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

		self.fc2 = nn.Linear(256,128)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(128,64)
		self.fc3.weight.data = fanin_init(self.fc3.weight.data.size())

		self.fc4 = nn.Linear(64,1)
		self.fc4.weight.data.uniform_(-EPS,EPS)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = F.tanh(self.fc4(x))

		x = x * 0.5 + 0.5

		return x

class ParetoFront:
	def __init__(self):
		self.stopping_criterion = 20
		self._reset()

	def _reset(self):
		print('Reset environment.')
		# points on pareto front
		# (acc,cr,c_param)
		self.data = SortedDict()
		# average compression param of cfgs
		# on and not on pareto front
		self.dominated_c_param = np.zeros(6,dtype=np.float64)
		self.dominated_cnt = 1e-6
		self.dominating_c_param = np.zeros(6,dtype=np.float64)
		self.dominating_cnt = 1e-6

	def add(self, c_param, dp):
		reward = 0
		# check the distance of (accuracy,bandwidth) to the previous Pareto Front
		to_remove = set()
		add_new = True
		for point in self.data:
			# if there is a same point, we dont add this
			if point[:2] == dp: 
				add_new = False
				break
			# if a point is dominated
			if point[0] <= dp[0] and point[1] <= dp[1]:
				to_remove.add(point)
			# if the new point is dominated
			# maybe 0 reward is error is small?
			elif point[0] >= dp[0] and point[1] >= dp[1]:
				if max(-dp[0]+point[0],-dp[1]+point[1])<=0.1:
					reward = 0
				else:
					reward = -1
				add_new = False
				break

		# remove dominated points
		for point in to_remove:
			self.dominated_c_param += self.data[point]
			self.dominated_cnt += 1
			self.dominating_c_param -= self.data[point]
			self.dominating_cnt -= 1
			del self.data[point]

		# update the current Pareto Front
		if add_new:
			self.dominating_c_param += c_param
			self.dominating_cnt += 1
			self.data[dp] = c_param
			reward = self._area()
		else:
			self.dominated_c_param += c_param
			self.dominated_cnt += 1

		# what if there is a noisy point (.99,.99)
		return reward

	def _area(self):
		# approximate area
		area = 0
		left = 0
		for datapoint in self.data:
			area += (datapoint[0]-left)*datapoint[1]
			left = datapoint[0]
		return area

	def get_observation(self):
		new_state = np.concatenate((self.dominating_c_param/self.dominating_cnt,self.dominated_c_param/self.dominated_cnt))
		if int(self.dominated_cnt + self.dominating_cnt)>=self.stopping_criterion:
			print(self.data.keys())
			self._reset()
		return new_state

class C_Generator:
	def __init__(self):
		MAX_BUFFER = 1000000
		S_DIM = 12
		A_DIM = 6
		A_MAX = 0.5 #[-.5,.5]

		self.ram = MemoryBuffer(MAX_BUFFER)
		self.trainer = Trainer(S_DIM, A_DIM, A_MAX, self.ram)
		self.paretoFront = ParetoFront()

	def get(self):
		# get an action from the actor
		state = np.float32(self.paretoFront.get_observation())
		self.action = self.trainer.get_exploration_action(state)
		# self.C_param = self.uniform_init_gen()
		# self.action = np.array([.1,.1,.1,0,0,0],dtype=np.float64)
		return self.action

	def uniform_init_gen(self):
		# 0,1,2:feature weights; 3,4:lower and upper; 5:order
		output = np.zeros(6,dtype=np.float64)
		output[:4] = np.random.randint(1,10,4)
		output[4] = np.random.randint(output[3],11)
		output[:5] /= 10
		output[5] = np.random.randint(0,5) #[1/3,1/2,1,2,3]
		return output   

	def optimize(self, datapoint, done):
		# if one episode ends, do nothing
		if done: 
			self.trainer.save_models(0)
			return
		# use (accuracy,bandwidth) to update observation
		state = self.paretoFront.get_observation()
		reward = self.paretoFront.add(self.action, datapoint)
		new_state = self.paretoFront.get_observation()
		# add experience to ram
		self.ram.add(state, self.action, reward, new_state)
		# optimize the network 
		self.trainer.optimize()


def RL_train(net):
	np.random.seed(123)
	criterion = nn.MSELoss(reduction='sum')
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	log_file = open('training.log', "w", 1)

	# setup target network
	# so that we only do this once
	sim = Simulator()
	cgen = C_Generator()
	num_cfg = 100 # number of cfgs to be explored
	selected_ranges = [10,50,100,500,1000,2000]
	print('Num batches:',num_cfg,sim.num_batches)

	TF = Transformer('compression')
	# the pareto front can be restarted, need to try

	for bi in range(num_cfg):
		# DDPG-based generator
		C_param = cgen.get()
		print_str = str(bi)+str(C_param)
		print(print_str)
		# apply the compression param chosen by the generator
		fetch_start = time.perf_counter()
		dps = []
		print_str = str(bi)
		for r in selected_ranges:
			# the function to get results from cloud model
			dp = sim.get_one_point(datarange=(0,r), TF=TF, C_param=np.copy(C_param))
			dps.append(dp)
			print_str += '\t'+str(dp[0])+'\t'+str(dp[1])
		print(print_str)
		log_file.write(print_str + '\n')
		# optimize generator
		cgen.optimize(dps[-1],False)

	torch.save(net.state_dict(), PATH)

def dual_train(net):
	np.random.seed(123)
	criterion = nn.MSELoss(reduction='sum')
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	log_file = open('training.log', "w", 1)
	log_file.write('Training...\n')

	# setup target network
	# so that we only do this once
	sim = Simulator(10)
	cgen = C_Generator()
	num_cfg = 1#sim.point_per_sim//batch_size
	print('Num batches:',num_cfg,sim.point_per_sim)

	for epoch in range(10):
		running_loss = 0.0
		TF = Transformer('compression')
		# the pareto front can be restarted, need to try

		for bi in range(num_cfg):
			inputs,labels = [],[]
			# DDPG-based generator
			C_param = cgen.get()
			# batch result of mAP and compression ratio
			batch_acc, batch_cr = [],[]
			for k in range(batch_size):
				di = bi*batch_size + k # data index
				# start counting the compressed size
				TF.reset()
				# apply the compression param chosen by the generator
				fetch_start = time.perf_counter()
				# the function to get results from cloud model
				sim_result = sim.get_one_point(index=di, TF=TF, C_param=np.copy(C_param))
				fetch_end = time.perf_counter()
				# get the compression ratio
				cr = TF.get_compression_ratio()
				batch_acc += [sim_result]
				batch_cr += [cr]
				print_str = str(di)+str(C_param)+'\t'+str(sim_result)+'\t'+str(cr)+'\t'+str(fetch_end-fetch_start)
				print(print_str)
				log_file.write(print_str+'\n')
				inputs.append(C_param)
				labels.append(sim_result) # accuracy of IoU=0.5
			# optimize generator
			cgen.optimize((np.mean(batch_acc),np.mean(batch_cr)),False)
			log_file.write(print_str+'\n')
			# transform to tensor
			inputs = torch.FloatTensor(inputs)#.cuda()
			labels = torch.FloatTensor(labels)#.cuda()

			# zero gradient
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = net(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# print statistics
			running_loss += loss.item()
			val_loss = abs(torch.mean(labels.cpu()-outputs.cpu()))
			print_str = '{:d}, {:d}, loss {:.6f}, val loss {:.6f}'.format(epoch + 1, bi + 1, loss.item(), val_loss)
			print(print_str)
			log_file.write(print_str + '\n')
		print_str = str(cgen.paretoFront.data.keys())
		print(print_str)
		cgen.optimize(None,True)
		torch.save(net.state_dict(), PATH)

if __name__ == "__main__":
	# prepare network
	net = RSNet()
	# net.load_state_dict(torch.load('backup/rsnet.pth'))
	# net = net.cuda()
	RL_train(net)

