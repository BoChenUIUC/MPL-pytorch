import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from compression.transformer import Transformer
from compression.ddpgtrain import Trainer
from compression.ddpgbuffer import MemoryBuffer
from sortedcontainers import SortedDict
from tqdm import tqdm
from mpl import Simulator
import mobopt as mo
# MOO
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.factory import get_problem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
from pymoo.model.problem import Problem

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

def config2points(EXP_NAME):
	points = []
	acc_file = open(EXP_NAME+'_acc.log')
	cr_file = open(EXP_NAME+'_cr.log')
	cfg_file = open(EXP_NAME+'_cfg.log')

	for l1,l2,l3 in zip(acc_file.readlines(),cr_file.readlines(),cfg_file.readlines()):
		acc = float(l1.strip())
		cr = float(l2.strip())
		C_param = [float(n) for n in l3.strip().split() ]
		points.append((C_param,(acc,cr)))
	return points

def configs2paretofront(EXP_NAME):
	pf = ParetoFront(EXP_NAME,10000)
	points = config2points(EXP_NAME)
	for C_param,dp in points:
		pf.add(C_param,(acc,cr))
	pf.save()

def comparePF(name1,name2):
	# output coverage
	pf1 = ParetoFront(name1,10000)
	pf2 = ParetoFront(name2,10000)
	points1 = config2points(name1)
	points2 = config2points(name2)
	cov_file = open(name1 + '_' + name2 + '.log', "w", 1)
	for pt1,pt2 in zip(points1,points2):
		pf1.add(*pt1)
		pf2.add(*pt2)
		cov1 = pf1.cov(pf2)
		cov2 = pf2.cov(pf1)
		cov_file.write(str(cov1)+' '+str(cov2)+' '+
					str(pf1.area())+' '+str(pf2.area())+' '+
					str(pf1.uniformity())+' '+ str(pf2.uniformity())+'\n')


class ParetoFront:
	def __init__(self,name='RE',stopping_criterion=100):
		self.stopping_criterion = stopping_criterion
		self.reset()
		self.pf_file = open(name+'_pf.log', "w", 1)

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
		non_trivial = False
		for point in self.data:
			if point in [(0,1),(1,0)]:continue
			# if there is a same point, we dont add this
			if point[:2] == dp: 
				add_new = False
				break
			# if a point is dominated
			if point[0] <= dp[0] and point[1] <= dp[1]:
				to_remove.add(point)
				# more requirement on cr
				if point[0] <= dp[0] or point[1]+0.05 < dp[1]:
					non_trivial = True
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
			# pre_score = self.uniformity()
			self.data[dp] = (angle,c_param)
			# cur_score = self.uniformity()
			# reward = cur_score/pre_score if non_trivial else 0
			reward = dp[0]
		else:
			self.dominated_c_param += c_param
			self.dominated_cnt += 1

		# what if there is a noisy point (.99,.99)
		self.reward += reward
		return reward

	def cov(self,other):
		covered = 0.0
		for dp1 in other.data:
			if dp1 in [(0,1),(1,0)]:continue
			dominated = False
			for dp2 in self.data:
				if dp2 in [(0,1),(1,0)]:continue
				if dp2[0]>dp1[0] and dp2[1]>dp1[1]:
					dominated = True
					break
			if dominated:covered += 1
		return covered/(len(other.data)-2)

	def uniformity(self):
		angle_arr = [self.data[dp][0] for dp in self.data]
		if len(angle_arr)==2:return 1
		angle_diff = np.diff(angle_arr)
		return 1/(np.std(angle_diff)/np.mean(angle_diff)/len(angle_diff))

	def area(self):
		# approximate area
		area = 0
		right = 1
		for datapoint in self.data:
			if datapoint in [(0,1),(1,0)]:continue
			assert(datapoint[1]<=right and 1>=datapoint[0])
			area += (1 - datapoint[0])*(right-datapoint[1])
			right = datapoint[1]
		return area

	def save(self):
		for k in self.data:
			if k in [(0,1),(1,0)]:continue
			self.pf_file.write(str(float(k[0]))+' '+str(k[1])+' '+' '.join([str(n) for n in self.data[k][1]])+'\n')

	def end_of_episode(self):
		return int(self.dominated_cnt + self.dominating_cnt)>=self.stopping_criterion

	def get_observation(self):
		new_state = np.concatenate((self.dominating_c_param/self.dominating_cnt,self.dominated_c_param/self.dominated_cnt))
		return new_state

class C_Generator:
	def __init__(self,name='CCVE',explore=True):
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
		if self.name == 'CCVE':
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

	def optimize(self, datapoint, done):
		if self.name == 'CCVE':
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
			self.paretoFront.reset()

# NAGA2
def pareto_front_approx_nsga2():
	class MyProblem(Problem):
		def __init__(self):
			super().__init__(n_var=6, n_obj=2, n_constr=0, xl=np.array([-.5]*6), xu=np.array([.5]*6))
			self.sim = Simulator(train=True)
			self.TF = Transformer('compression')
			self.datarange = [0,100]
			self.cfg_file = open('NSGA2_cfg.log', "w", 1)
			self.acc_file = open('NSGA2_acc.log', "w", 1)
			self.cr_file = open('NSGA2_cr.log', "w", 1)
			self.iter = 0

		def _evaluate(self, x, out, *args, **kwargs):
			print(x.shape)
			points = np.array([])
			for row in range(x.shape[0]):
				acc,cr = self.sim.get_one_point(datarange=self.datarange, TF=self.TF, C_param=x)
				points.append(np.array([float(acc),cr]))
				self.cfg_file.write(' '.join([str(n) for n in x])+'\n')
				self.acc_file.write(str(float(acc))+'\n')
				self.cr_file.write(str(cr)+'\n')
				print('Iter:',self.iter)
				self.iter += 1
			out["F"] = points

	problem = MyProblem()

	algorithm = NSGA2(pop_size=100)

	res = minimize(problem,
					algorithm,
					('n_gen', 5),
					seed=1,
					verbose=False)

	# plot = Scatter()
	# plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
	# plot.add(res.F, color="red")
	# plot.show()

# PFA using MOBO
def pareto_front_approx_mobo():
	d = {}
	d['cfg_file'] = open('MOBO_cfg.log', "w", 1)
	d['acc_file'] = open('MOBO_acc.log', "w", 1)
	d['cr_file'] = open('MOBO_cr.log', "w", 1)
	d['iter'] = 0
	def objective(x):
		sim = Simulator(train=True)
		TF = Transformer('compression')
		datarange = [0,100]
		acc,cr = sim.get_one_point(datarange=datarange, TF=TF, C_param=x)
		d['cfg_file'].write(' '.join([str(n) for n in x])+'\n')
		d['acc_file'].write(str(float(acc))+'\n')
		d['cr_file'].write(str(cr)+'\n')
		print('Iter:',d['iter'])
		d['iter'] += 1
		return np.array([float(acc),cr])
	Optimizer = mo.MOBayesianOpt(target=objective,
		NObj=2,
		pbounds=np.array([[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5],[-0.5,0.5]]))
	Optimizer.initialize(init_points=50)
	front, pop = Optimizer.maximize(n_iter=1000)

# PFA
def pareto_front_approx():
	EXP_NAME = 'CCVE'
	cfg_file = open(EXP_NAME+'_cfg.log', "w", 1)
	acc_file = open(EXP_NAME+'_acc.log', "w", 1)
	cr_file = open(EXP_NAME+'_cr.log', "w", 1)

	# setup target network
	# so that we only do this once
	sim = Simulator(train=True)
	cgen = C_Generator(name=EXP_NAME,explore=True)
	num_cfg = 1000 # number of cfgs to be explored
	datarange = [0,100]
	print(EXP_NAME,'num configs:',num_cfg, 'total batches:', sim.num_batches)

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

# input: pf file/JPEG/JPEG2000
# output: pf file on test
def evaluation():
	EXP_NAME = 'CCVE'
	np.random.seed(123)
	torch.manual_seed(2)

	sim = Simulator()
	pf = ParetoFront(EXP_NAME)
	TF = Transformer(name=EXP_NAME)
	datarange = [0,sim.num_batches]

	if EXP_NAME == 'CCVE':
		# sim.get_one_point(datarange, TF=TF, C_param=C_param)
		# pf.add(C_param,(map50,cr))
		with open('DDPG_pf.log','r') as f:
			for line in f.readlines:
				tmp = line.strip().split(' ')
	else:
		map50,cr = sim.get_one_point(datarange, TF=TF, C_param=None)
		print(map50,cr)

# determine sample size
def test_run():
	np.random.seed(123)
	torch.manual_seed(2)
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
	np.random.seed(123)
	torch.manual_seed(2)
	# prepare network
	# net = RSNet()
	# net.load_state_dict(torch.load('backup/rsnet.pth'))
	# net = net.cuda()

	# determine lenght of episode
	# test_run()

	# use ddpg or re for approx
	# pareto_front_approx()

	# convert from .log file to pf for eval
	# configs2paretofront('CCVE')

	# compute coverage, maybe also hypervolume?
	# comparePF('CCVE','RE')

	# pareto_front_approx_mobo()

	pareto_front_approx_nsga2()

