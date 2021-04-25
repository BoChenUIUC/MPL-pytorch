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

no_of_hidden_units = 196
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 = nn.Conv2d(3, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		self.ln1 = nn.LayerNorm([no_of_hidden_units,32,32])
		self.lrelu1 = nn.LeakyReLU()

		self.conv2 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=2, padding=1)
		self.ln2 = nn.LayerNorm([no_of_hidden_units,16,16])
		self.lrelu2 = nn.LeakyReLU()

		self.conv3 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		self.ln3 = nn.LayerNorm([no_of_hidden_units,16,16])
		self.lrelu3 = nn.LeakyReLU()

		self.conv4 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=2, padding=1)
		self.ln4 = nn.LayerNorm([no_of_hidden_units,8,8])
		self.lrelu4 = nn.LeakyReLU()

		self.conv5 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		self.ln5 = nn.LayerNorm([no_of_hidden_units,8,8])
		self.lrelu5 = nn.LeakyReLU()

		self.conv6 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		self.ln6 = nn.LayerNorm([no_of_hidden_units,8,8])
		self.lrelu6 = nn.LeakyReLU()

		self.conv7 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		self.ln7 = nn.LayerNorm([no_of_hidden_units,8,8])
		self.lrelu7 = nn.LeakyReLU()

		self.conv8 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=2, padding=1)
		self.ln8 = nn.LayerNorm([no_of_hidden_units,4,4])
		self.lrelu8 = nn.LeakyReLU()

		self.pool = nn.MaxPool2d(4, 4)
		self.fc1 = nn.Linear(no_of_hidden_units, 1)

	def forward(self, x, extract_features=0):
		x = self.ln1(self.lrelu1(self.conv1(x)))
		x = self.ln2(self.lrelu2(self.conv2(x)))
		x = self.ln3(self.lrelu3(self.conv3(x)))
		x = self.ln4(self.lrelu4(self.conv4(x)))
		x = self.ln5(self.lrelu5(self.conv5(x)))
		x = self.ln6(self.lrelu6(self.conv6(x)))
		x = self.ln7(self.lrelu7(self.conv7(x)))
		x = self.ln8(self.lrelu8(self.conv8(x)))
		x = self.pool(x)
		x = x.view(-1, no_of_hidden_units)
		y1 = self.fc1(x)
		return y1

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.conv2 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
        self.conv2 = spectral_norm(self.conv2)
        self.bn2 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv3 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
        self.conv3 = spectral_norm(self.conv3)
        self.bn3 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv4 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
        self.conv4 = spectral_norm(self.conv4)
        self.bn4 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv5 = nn.ConvTranspose2d(no_of_hidden_units, no_of_hidden_units, 4, stride=2, padding=1)
        self.conv5 = spectral_norm(self.conv5)
        self.bn5 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv6 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
        self.conv6 = spectral_norm(self.conv6)
        self.bn6 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv7 = nn.ConvTranspose2d(no_of_hidden_units, no_of_hidden_units, 4, stride=2, padding=1)
        self.conv7 = spectral_norm(self.conv7)
        self.bn7 = nn.BatchNorm2d(no_of_hidden_units)

        self.conv8 = nn.Conv2d(no_of_hidden_units, 3, 3, stride=1, padding=1)
        self.conv8 = spectral_norm(self.conv8)
    def forward(self,x):
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(F.relu(self.conv5(x)))
        x = self.bn6(F.relu(self.conv6(x)))
        x = self.bn7(F.relu(self.conv7(x)))
        x = self.conv8(x)
        return torch.tanh(x)

def compute_gradient_penalty(D, real_samples, fake_samples, cuda):
	"""Calculates the gradient penalty loss for WGAN GP"""
	# Random weight term for interpolation between real and fake samples
	alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
	if cuda:alpha = alpha.cuda()
	# Get random interpolation between real and fake samples
	interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
	d_interpolates = D(interpolates)
	fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
	if cuda: fake = fake.cuda()
	# Get gradient w.r.t. interpolates
	gradients = torch.autograd.grad(
		outputs=d_interpolates,
		inputs=interpolates,
		grad_outputs=fake,
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty

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
	# w_transpose = torch.transpose(w, 1, 2)
	# w_mul = torch.matmul(w, w_transpose)
	# identity = torch.diag(torch.ones(h))
	# identity = identity.repeat(cin*cout,1,1)
	# if cuda:
	# 	identity = identity.cuda()
	# l2norm = torch.nn.MSELoss()
	# ortho_loss = l2norm(w_mul, identity)
	# return scale * ortho_loss

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

class LightweightEncoder(nn.Module):

	def __init__(self, channels, kernel_size=4, num_centers=8):
		super(LightweightEncoder, self).__init__()
		self.sample = nn.Conv2d(3, channels, kernel_size=kernel_size, stride=kernel_size, padding=0, bias=True)
		self.sample = spectral_norm(self.sample)
		self.centers = torch.nn.Parameter(torch.rand(num_centers))

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

		return x

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

	def __init__(self, kernel_size=4, num_centers=8):
		super(DeepCOD, self).__init__()
		out_size = 3
		self.encoder = LightweightEncoder(out_size, kernel_size=4, num_centers=8)
		self.attention_1 = Attention(out_size,no_of_hidden_units)
		self.resblock_up1 = Resblock_up(out_size,no_of_hidden_units)
		self.conv1 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
		self.conv1 = spectral_norm(self.conv1)
		self.bn1 = nn.BatchNorm2d(no_of_hidden_units)
		self.conv2 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
		self.conv2 = spectral_norm(self.conv2)
		self.bn2 = nn.BatchNorm2d(no_of_hidden_units)
		self.attention_2 = Attention(no_of_hidden_units,no_of_hidden_units)
		self.resblock_up2 = Resblock_up(no_of_hidden_units,no_of_hidden_units)
		self.conv3 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
		self.conv3 = spectral_norm(self.conv3)
		self.bn3 = nn.BatchNorm2d(no_of_hidden_units)
		self.conv4 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, 3, stride=1, padding=1)
		self.conv4 = spectral_norm(self.conv4)
		self.bn4 = nn.BatchNorm2d(no_of_hidden_units)
		self.output_conv = Output_conv(no_of_hidden_units)
		

	def forward(self, x): 
		x = self.encoder(x)

		# reconstruct
		x = self.attention_1(x)
		x = self.resblock_up1(x)
		x = self.conv1(F.relu(self.bn1(x)))
		x = self.conv2(F.relu(self.bn2(x)))
		x = self.attention_2(x)
		x = self.resblock_up2(x)
		x = self.conv3(F.relu(self.bn3(x)))
		x = self.conv4(F.relu(self.bn4(x)))
		x = self.output_conv(x)
		
		return x

if __name__ == '__main__':
	image = torch.randn(1,3,32,32)
	model = DeepCOD()
	output = model(image)
	print(model)
	# print(output.shape)
	# weight = torch.diag(torch.ones(4)).repeat(3,3,1,1)
	# print(weight.size())
	print(model.encoder.sample.weight.size())
	# r = orthorgonal_regularizer(model.sample.weight,1,False)
	# print(r)
	# for name, param in model.named_parameters():
	# 	print('name is {}'.format(name))
	# 	print('shape is {}'.format(param.shape))