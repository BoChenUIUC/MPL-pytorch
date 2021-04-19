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

def orthorgonal_regularizer(weight,scale,cuda=False):
	cin,cout,h,w = weight.size()
	weight = weight.view(cin*cout, h, w)
	w_transpose = torch.transpose(weight, 1, 2)
	w_mul = torch.matmul(weight, w_transpose)
	identity = torch.diag(torch.ones(h))
	identity = identity.repeat(cin*cout,1,1)
	if cuda:
		identity = identity.cuda()
	l2norm = torch.nn.MSELoss()
	ortho_loss = l2norm(w_mul, identity)
	return scale * ortho_loss

class Attention_full(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(Attention_full, self).__init__()
		f_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.f_conv = spectral_norm(f_conv)
		g_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.g_conv = spectral_norm(g_conv)
		h_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.h_conv = spectral_norm(h_conv)
		attn_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)
		self.attn_conv = spectral_norm(attn_conv)
		self.gamma = torch.nn.Parameter(torch.FloatTensor([1]))
		self.out_channels = out_channels

	def forward(self,x):
		nb, nc, imgh, imgw = x.size() 

		f = (self.f_conv(x)).view(nb,self.out_channels,-1)
		g = (self.g_conv(x)).view(nb,self.out_channels,-1)
		h = (self.h_conv(x)).view(nb,self.out_channels,-1)

		s = torch.matmul(f.transpose(1,2),g)
		beta = F.softmax(s, dim=-1)
		o = torch.matmul(beta,h.transpose(1,2))
		o = self.attn_conv(o.transpose(1,2).view(nb,self.out_channels,imgh,imgw))
		x = self.gamma * o

		return x

class Attention_2(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(Attention_2, self).__init__()
		f_conv = nn.Conv2d(in_channels, out_channels//8, kernel_size=1, stride=1, padding=0, bias=True)
		self.f_conv = spectral_norm(f_conv)
		g_conv = nn.Conv2d(in_channels, out_channels//8, kernel_size=1, stride=1, padding=0, bias=True)
		self.g_conv = spectral_norm(g_conv)
		h_conv = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1, padding=0, bias=True)
		self.h_conv = spectral_norm(h_conv)
		attn_conv = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=1, stride=1, padding=0, bias=True)
		self.attn_conv = spectral_norm(attn_conv)
		self.gamma = torch.nn.Parameter(torch.FloatTensor([1]))
		self.out_channels = out_channels

	def forward(self,x):
		nb, nc, imgh, imgw = x.size() 

		f = (self.f_conv(x)).view(nb,self.out_channels//8,-1)
		g = (self.g_conv(x)).view(nb,self.out_channels//8,-1)
		h = (self.h_conv(x)).view(nb,self.out_channels//4,-1)

		s = torch.matmul(f.transpose(1,2),g)
		beta = F.softmax(s, dim=-1)
		o = torch.matmul(beta,h.transpose(1,2))
		o = self.attn_conv(o.transpose(1,2).view(nb,self.out_channels//4,imgh,imgw))
		x = self.gamma * o

		return x

class Resblock_up(nn.Module):

	def __init__(self, channels):
		super(Resblock_up, self).__init__()
		self.bn1 = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)
		self.relu1 = nn.ReLU(inplace=True)
		deconv1 = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
		self.deconv1 = spectral_norm(deconv1)

		self.bn2 = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)
		self.relu2 = nn.ReLU(inplace=True)
		deconv2 = nn.ConvTranspose2d(channels, channels, 3, stride=1, padding=1)
		self.deconv2 = spectral_norm(deconv2)

		deconv_skip = nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1)
		self.deconv_skip = spectral_norm(deconv_skip)

	def forward(self, x_init):
		x = self.deconv1(self.relu1(self.bn1(x_init)))
		x = self.deconv2(self.relu2(self.bn2(x)))
		x_init = self.deconv_skip(x_init)
		return x + x_init

class Output_conv(nn.Module):

	def __init__(self, channels):
		super(Output_conv, self).__init__()
		self.bn = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)
		self.relu = nn.ReLU(inplace=True)
		self.conv = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)

	def forward(self, x):
		x = self.conv(self.relu(self.bn(x)))
		x = torch.tanh(x)
		x = (x+1)/2

		return x


class DeepCOD(nn.Module):

	def __init__(self, kernel_size=4, num_centers=8):
		super(DeepCOD, self).__init__()
		self.sample = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, bias=True)
		self.centers = torch.rand(num_centers)
		self.centers = torch.nn.Parameter(self.centers)
		self.attention_full = Attention_full(3,64)
		self.resblock_up1 = Resblock_up(64)
		# self.attention_2 =Attention_full(64,64)
		# self.resblock_up2 = Resblock_up(64)
		# self.attention_2 =Attention_2(64,64)
		# self.resblock_up2 = Resblock_up(64//4)
		self.output_conv = Output_conv(64)
		

	def forward(self, x): 
		# sample from input
		x = self.sample(x)

		# quantization
		xsize = list(x.size())
		x = x.view(*(xsize + [1]))
		quant_dist = torch.pow(x-self.centers, 2)
		softout = torch.sum(self.centers * nn.functional.softmax(-quant_dist, dim=-1), dim=-1)
		maxval = torch.min(quant_dist, dim=-1, keepdim=True)[0]
		hardout = torch.sum(self.centers * (maxval == quant_dist), dim=-1)
		# dont know how to use hardout, use this temporarily
		x = softout

		# reconstruct
		x = self.attention_full(x)
		x = self.resblock_up1(x)
		# x = self.attention_2(x)
		# x = self.resblock_up2(x)
		x = self.output_conv(x)
		
		return x

if __name__ == '__main__':
	image = torch.randn(1,3,32,32)
	model = DeepCOD()
	output = model(image)
	print(model)
	print(output.shape)
	# for p in model.parameters():
	# 	print(p.shape)