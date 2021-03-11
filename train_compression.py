import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from mpl import Simulator
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
	def __init__(self,name='RE'):
		self.stopping_criterion = 20
		self.reset()
		self.pf_file = open(name+'_pf.log', "w", 1)
		self.area_file = open(name+'_area.log', "w", 1)
		self.reward_file = open(name+'_reward.log', "w", 1)

	def reset(self):
		print('Reset environment.')
		# points on pareto front
		# (acc,cr,c_param)
		self.data = SortedDict()
		# init with points at two ends
		self.data[(0,1)] = (0,None)
		self.data[(1,0)] = (np.pi/2,None)
		# average compression param of cfgs
		# on and not on pareto front
		self.dominated_c_param = np.zeros(6,dtype=np.float64)
		self.dominated_cnt = 1e-6
		self.dominating_c_param = np.zeros(6,dtype=np.float64)
		self.dominating_cnt = 1e-6
		self.reward = 0

	def add(self, c_param, dp):
		reward = 0
		# check the distance of (accuracy,bandwidth) to the previous Pareto Front
		to_remove = set()
		add_new = True
		for point in self.data:
			if point in [(0,1),(1,0)]:continue
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
			self.dominated_c_param += self.data[point][1]
			self.dominated_cnt += 1
			self.dominating_c_param -= self.data[point][1]
			self.dominating_cnt -= 1
			del self.data[point]

		# update the current Pareto Front
		if add_new:
			self.dominating_c_param += c_param
			self.dominating_cnt += 1
			angle = np.arctan(dp[1]/dp[0])
			pre_score = self._distribution_score()
			self.data[dp] = (angle,c_param)
			cur_score = self._distribution_score()
			reward = cur_score/pre_score
			# 1. area as reward
			# reward = self._area()
			# 2. accuracy as reward for encouragement
			# reward = dp[0]*dp[1]
			# 3. product as reward
			# 4. only give reward to acc if delta acc>0.1
			# too small accuracy should be penalized
			# min angle as reward
		else:
			self.dominated_c_param += c_param
			self.dominated_cnt += 1

		# what if there is a noisy point (.99,.99)
		self.reward += reward
		return reward

	def _distribution_score(self):
		angle_arr = [self.data[dp][0] for dp in self.data]
		if len(angle_arr)==2:return 1
		angle_diff = np.diff(angle_arr)
		return 1/(np.std(angle_diff)/np.mean(angle_diff))

	def _area(self):
		# approximate area
		area = 0
		left = 0
		for datapoint in self.data:
			area += (datapoint[0]-left)*datapoint[1]
			left = datapoint[0]
		return area

	def save(self):
		self.pf_file.write('ParetoFront\n')
		for k in self.data:
			if k in [(0,1),(1,0)]:continue
			self.pf_file.write(str(float(k[0]))+' '+str(k[1])+' '+' '.join([str(n) for n in self.data[k][1]])+'\n')
		self.area_file.write(str(self._area())+'\n')
		self.reward_file.write(str(self.reward)+'\n')

	def end_of_episode(self):
		return int(self.dominated_cnt + self.dominating_cnt)>=self.stopping_criterion

	def get_observation(self):
		new_state = np.concatenate((self.dominating_c_param/self.dominating_cnt,self.dominated_c_param/self.dominated_cnt))
		return new_state

class C_Generator:
	def __init__(self,name='DDPG',explore=True):
		MAX_BUFFER = 1000000
		S_DIM = 12
		A_DIM = 6
		A_MAX = 0.5 #[-.5,.5]

		self.name = name
		self.ram = MemoryBuffer(MAX_BUFFER)
		self.trainer = Trainer(S_DIM, A_DIM, A_MAX, self.ram)
		self.paretoFront = ParetoFront(name)
		self.explore = explore

	def get(self):
		if self.name == 'DDPG':
			self.action = self._DDPG_action()
		else:
			self.action = self._RE_action()
		return self.action

	def _DDPG_action(self):
		# get an action from the actor
		state = np.float32(self.paretoFront.get_observation())
		if self.explore:
			action = self.trainer.get_exploration_action(state)
		else:
			action = self.trainer.get_exploitation_action(state)
		action = (action+0.5)%1-0.5
		return action

	def _RE_action(self):
		return np.random.random(6)-0.5

	def _GPR_action(self):
		pass

	def _RF_action(self):
		pass

	def _MPL_action(self):
		pass

	def save(self):
		self.paretoFront.save()

	def optimize(self, datapoint, done):
		if self.name == 'DDPG':
			self._DDPG_optimize(datapoint, done)
		elif self.name == 'RE':
			self.paretoFront.add(self.action, datapoint)

	def _DDPG_optimize(self, datapoint, done):
		# if one episode ends, do nothing
		# use (accuracy,bandwidth) to update observation
		state = self.paretoFront.get_observation()
		reward = self.paretoFront.add(self.action, datapoint)
		new_state = self.paretoFront.get_observation()
		# add experience to ram
		self.ram.add(state, self.action, reward, new_state)
		# optimize the network 
		self.trainer.optimize()
		# reset PF if needed
		if self.explore and self.paretoFront.end_of_episode():
			self.paretoFront.save()
			self.paretoFront.reset()

def pareto_front_approx(net):
	EXP_NAME = 'DDPG'
	np.random.seed(123)
	torch.manual_seed(2)
	criterion = nn.MSELoss(reduction='sum')
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	cfg_file = open(EXP_NAME+'_cfg.log', "w", 1)
	acc_file = open(EXP_NAME+'_acc.log', "w", 1)
	cr_file = open(EXP_NAME+'_cr.log', "w", 1)

	# setup target network
	# so that we only do this once
	sim = Simulator(train=True)
	cgen = C_Generator(name=EXP_NAME,explore=True)
	num_cfg = 100 # number of cfgs to be explored
	datarange = [0,100]
	print('Num configs:',num_cfg, 'total batches:', sim.num_batches)

	TF = Transformer('compression')
	# the pareto front can be restarted, need to try

	for bi in range(num_cfg):
		# DDPG-based generator
		C_param = cgen.get()
		# apply the compression param chosen by the generator
		map50,cr = sim.get_one_point(datarange=datarange, TF=TF, C_param=np.copy(C_param))
		# optimize generator
		cgen.optimize((map50,cr),False)
		# write logs
		cfg_file.write(' '.join([str(n) for n in C_param])+'\n')
		acc_file.write(str(float(map50))+'\n')
		cr_file.write(str(cr)+'\n')
	cgen.save()
	torch.save(net.state_dict(), PATH)

def test_run(net):
	np.random.seed(123)
	torch.manual_seed(2)
	criterion = nn.MSELoss(reduction='sum')
	optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
	cfg_file = open('cfg.log', "w", 1)
	acc_file = open('acc.log', "w", 1)
	cr_file = open('cr.log', "w", 1)

	# setup target network
	# so that we only do this once
	sim = Simulator()
	cgen = C_Generator(explore=True)
	num_cfg = 100 # number of cfgs to be explored
	selected_ranges = [10*i for i in range(1,10)]+[100*i for i in range(1,8)]+[782]
	print('Num batches:',num_cfg,sim.num_batches)

	TF = Transformer('compression')
	# the pareto front can be restarted, need to try

	for bi in range(num_cfg):
		# DDPG-based generator
		C_param = cgen.get()
		# apply the compression param chosen by the generator
		map50s,crs = sim.get_multi_point(selected_ranges, TF=TF, C_param=np.copy(C_param))
		# optimize generator
		cgen.optimize((map50s[-1],crs[-1]),False)
		# write logs
		cfg_file.write(' '.join([str(n) for n in C_param])+'\n')
		acc_file.write(' '.join([str(n) for n in map50s])+'\n')
		cr_file.write(' '.join([str(n) for n in crs])+'\n')
		# if the total reward reaches some point, start profiling and end

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
	# test_run(net)
	pareto_front_approx(net)

