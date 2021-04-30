import cv2
import numpy as np
import time
import torch
import glob
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from huffman import HuffmanCoding

no_of_hidden_units = 196

def orthorgonal_regularizer(w,scale,cuda=False):
	N, C, H, W = w.size()
	w = w.view(N*C, H, W)
	weight_squared = torch.bmm(w, w.permute(0, 2, 1))
	ones = torch.ones(N * C, H, H, dtype=torch.float32)
	diag = torch.eye(H, dtype=torch.float32)
	tmp = ones - diag
	if cuda:tmp = tmp.cuda()
	loss_orth = ((weight_squared * tmp) ** 2).sum()
	return loss_orth*scale

class Attention(nn.Module):

	def __init__(self, channels, hidden_channels):
		super(Attention, self).__init__()
		f_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.f_conv = spectral_norm(f_conv)
		g_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.g_conv = spectral_norm(g_conv)
		h_conv = nn.Conv2d(channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.h_conv = spectral_norm(h_conv)
		v_conv = nn.Conv2d(hidden_channels, channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.v_conv = spectral_norm(v_conv)
		self.gamma = torch.nn.Parameter(torch.FloatTensor([0.0]))
		self.hidden_channels = hidden_channels
		self.channels = channels

	def forward(self,x):
		nb, nc, imgh, imgw = x.size() 

		f = (self.f_conv(x)).view(nb,self.hidden_channels,-1)
		g = (self.g_conv(x)).view(nb,self.hidden_channels,-1)
		h = (self.h_conv(x)).view(nb,self.hidden_channels,-1)

		s = torch.matmul(f.transpose(1,2),g)
		beta = F.softmax(s, dim=-1)
		o = torch.matmul(beta,h.transpose(1,2))
		o = self.v_conv(o.transpose(1,2).view(nb,self.hidden_channels,imgh,imgw))
		x = self.gamma * o + x

		return x

class Resblock_up(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(Resblock_up, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
		# self.relu1 = nn.LeakyReLU()
		deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
		self.deconv1 = spectral_norm(deconv1)

		self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
		# self.relu2 = nn.LeakyReLU()
		deconv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
		self.deconv2 = spectral_norm(deconv2)

		self.bn3 = nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
		# self.relu3 = nn.LeakyReLU()
		deconv_skip = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
		self.deconv_skip = spectral_norm(deconv_skip)

	def forward(self, x_init):
		x = self.deconv1(F.relu(self.bn1(x_init)))
		x = self.deconv2(F.relu(self.bn2(x)))
		x_init = self.deconv_skip(F.relu(self.bn3(x_init)))
		return x + x_init

class ContextExtractor(nn.Module):

	def __init__(self):
		super(ContextExtractor, self).__init__()
		self.conv1 = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
		self.bn1 = nn.BatchNorm2d(3, momentum=0.01, eps=1e-3)
		self.conv2 = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
		self.bn2 = nn.BatchNorm2d(3, momentum=0.01, eps=1e-3)
		self.conv3 = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
		self.bn3 = nn.BatchNorm2d(3, momentum=0.01, eps=1e-3)
		self.conv4 = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1)
		self.bn4 = nn.BatchNorm2d(3, momentum=0.01, eps=1e-3)

	def forward(self, x):
		x = self.conv1(F.relu(self.bn1(x)))
		x = self.conv2(F.relu(self.bn2(x)))
		x = self.conv3(F.relu(self.bn3(x)))
		x1 = self.conv4(F.relu(self.bn4(x)))
		x = (torch.tanh(x)+1)/2
		x1 = (torch.tanh(x1)+1)/2
		return x,x1


class LightweightEncoder(nn.Module):

	def __init__(self, channels, kernel_size=4, num_centers=8, use_subsampling=True):
		super(LightweightEncoder, self).__init__()
		self.sample = nn.Conv2d(3, channels, kernel_size=kernel_size, stride=kernel_size, padding=0, bias=True)
		self.sample = spectral_norm(self.sample)
		self.centers = torch.nn.Parameter(torch.rand(num_centers))
		# self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
		# self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)
		self.pool1 = nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)
		self.pool2 = nn.Conv2d(3, 3, kernel_size=2, stride=2, padding=0)
		self.unpool = nn.Upsample(scale_factor=2, mode='nearest')

		if use_subsampling:
			self.ctx = ContextExtractor()
		self.use_subsampling = use_subsampling

	def forward(self, x):
		# sample from input
		if self.use_subsampling:
			x_init,thresh = x
			x = self.sample(x_init)
		else:
			x = self.sample(x)

		# subsampling
		# data to be sent: mask + actual data
		B,C,H,W = x.size()
		assert(H%4==0 and W%4==0)
		if self.use_subsampling:
			# feature L1, L2(top/most lossy) 
			feat_1,feat_2 = self.ctx(x_init)
			feat_1_,feat_2_ = self.unpool(feat_1),self.unpool(self.unpool(feat_2))
			# thresh 1,2,3
			th_1, th_2 = thresh
			# sub-sample
			ss_1 = self.unpool(self.pool1(x))
			ss_2 = self.unpool(self.unpool(self.pool2(self.pool1(x))))
			# conditions
			cond_2 = feat_2_<th_2
			cond_1 = torch.logical_and(torch.logical_not(cond_2),feat_1_<th_1)
			# subsampled data in different areas
			data_1 = self.pool1(x)[torch.logical_and(self.unpool(feat_2)>=th_2,feat_1<th_1)]
			data_2 = self.pool2(self.pool1(x))[feat_2<th_2]
			cond_0 = torch.logical_not(torch.logical_or(cond_1,cond_2))
			data_0 = x[cond_0]
			comp_data = torch.cat((data_0,data_1,data_2),0)
			# affected data in the original shape
			x = torch.where(cond_1, ss_1, x)
			x = torch.where(cond_2, ss_2, x)

		# quantization
		xsize = list(x.size())
		x = x.view(*(xsize + [1]))
		quant_dist = torch.pow(x-self.centers, 2)
		softout = torch.sum(self.centers * nn.functional.softmax(-quant_dist, dim=-1), dim=-1)
		minval,index = torch.min(quant_dist, dim=-1, keepdim=True)
		hardout = torch.sum(self.centers * (minval == quant_dist), dim=-1)
		x = softout
		if self.use_subsampling:
			comp_data = comp_data.view(*(list(comp_data.size()) + [1]))
			quant_dist = torch.pow(comp_data-self.centers, 2)
			index = torch.min(quant_dist, dim=-1, keepdim=True)[1]
			# running length coding
			huffman = HuffmanCoding()
			real_size = len(huffman.compress(index.view(-1).cpu().numpy())) * 4 # bit
			real_size += H*W*C*B/4# + H*W*C*B/16
			esti_size = torch.count_nonzero(cond_0) + \
						torch.count_nonzero(cond_1)/4 + \
						torch.count_nonzero(cond_2)/16
			esti_cr = 1/16.*esti_size/(H*W*C*B)
			real_cr = 1/16.*real_size/(H*W*C*B*8)
			index = index.view(-1).unsqueeze(-1)
			index_nums = torch.arange(0, 8)#.cuda()
			counts = torch.sum(index==index_nums,dim=0)
			counts = counts/torch.sum(counts)
			std = torch.std(counts)
			return x,(esti_cr,real_cr,std)
		else:
			huffman = HuffmanCoding()
			real_size = len(huffman.compress(index.view(-1).cpu().numpy())) * 4
			real_cr = 1/16.*real_size/(H*W*C*B*8)
			return x,real_cr

class Output_conv(nn.Module):

	def __init__(self, channels):
		super(Output_conv, self).__init__()
		self.bn = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)
		# self.relu = nn.LeakyReLU()#nn.ReLU(inplace=True)
		self.conv = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv = spectral_norm(self.conv)

	def forward(self, x):
		x = self.conv(F.relu(self.bn(x)))
		x = torch.tanh(x)
		x = (x+1)/2

		return x

def init_weights(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out')
		nn.init.constant_(m.bias, 0)


class DeepCOD(nn.Module):

	def __init__(self, kernel_size=4, num_centers=8, use_subsampling=True):
		super(DeepCOD, self).__init__()
		out_size = 3
		self.encoder = LightweightEncoder(out_size, kernel_size=4, num_centers=num_centers, use_subsampling=use_subsampling)
		self.attention_1 = Attention(out_size,no_of_hidden_units)
		self.resblock_up1 = Resblock_up(out_size,no_of_hidden_units)
		self.attention_2 = Attention(no_of_hidden_units,no_of_hidden_units)
		self.resblock_up2 = Resblock_up(no_of_hidden_units,no_of_hidden_units)
		# self.conv1 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
		# self.conv1 = spectral_norm(self.conv1)
		# self.bn1 = nn.BatchNorm2d(no_of_hidden_units)
		# self.conv2 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
		# self.conv2 = spectral_norm(self.conv2)
		# self.bn2 = nn.BatchNorm2d(no_of_hidden_units)
		self.output_conv = Output_conv(no_of_hidden_units)
		

	def forward(self, x):
		x,r = self.encoder(x)

		# reconstruct
		x = self.attention_1(x)
		x = self.resblock_up1(x)
		x = self.attention_2(x)
		x = self.resblock_up2(x)
		# x = self.conv1(F.relu(self.bn1(x)))
		# x = self.conv2(F.relu(self.bn2(x)))
		x = self.output_conv(x)
		
		return x,r

if __name__ == '__main__':
	image = torch.randn(1,3,32,32)
	model = DeepCOD()
	output,r = model((image,(.5,.5)))
	# print(model)
	print(r)
	# print(output.shape)
	# weight = torch.diag(torch.ones(4)).repeat(3,3,1,1)
	# print(weight.size())
	# print(model.encoder.sample.weight.size())
	# r = orthorgonal_regularizer(model.sample.weight,1,False)
	# print(r)
	# for name, param in model.named_parameters():
	# 	print('name is {}'.format(name))
	# 	print('shape is {}'.format(param.shape))