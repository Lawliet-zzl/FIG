import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import data_loader
from models import *

class NHLoss(nn.Module):
	"""docstring for NHLoss"""
	def __init__(self):
		super(NHLoss, self).__init__()
	def forward(self, outputs):
		return (F.softmax(outputs, dim=1) * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()

class SamplerSGLD(nn.Module):
	"""docstring for SamplerSGLD"""
	def __init__(self):
		super(SamplerSGLD, self).__init__()

	def init_samples(self, bs):
		samples = torch.FloatTensor(bs, 3, 32, 32).uniform_(-1, 1)
		return samples

	def adjust_sgld_lr(self, lr, idx):
		if idx % 100 == 99:
			lr *= 0.9
		return lr

	def get_energy_func(self, net, samples):
		outputs = net(samples) / 5
		return torch.sum((1 - torch.exp(outputs))*outputs)

	def get_conf_func(self, net, samples):
		outputs = net(samples)
		return torch.max(F.softmax(outputs.data, dim=1), dim=1)[0].mean().item()

	def update_samples(self, idx, net, samples, lr):
		E = self.get_energy_func(net, samples)

		# samples.requires_grad = True
		# E.backward()
		# data_grad = samples.grad.data
		data_grad = torch.autograd.grad(E, [samples], retain_graph=True)[0]
		data_grad = data_grad.data

		# Collect the element-wise sign of the data gradient
		data_grad = data_grad.sign()

		samples.data += -lr*data_grad / 2 + lr*torch.randn_like(samples)

		# Adding clipping to maintain [-1,1] range
		samples.data = torch.clamp(samples.data, -1, 1)
			
		lr = self.adjust_sgld_lr(lr, idx)

		return samples, lr

	def forward(self, net, T=20, num_samples=1):
		net.eval()

		samples = self.init_samples(num_samples)

		samples = samples.cuda()
		samples.requires_grad = True
		
		lr = 0.1

		for idx in range(T):
			samples, lr = self.update_samples(idx, net, samples, lr)
			conf = self.get_conf_func(net, samples)
			if conf > 0.9:
				break
		energy = self.get_energy_func(net, samples).item() / num_samples

		net.train()
		return samples

def build_model(model, num_classes):
	if model == 'ResNet18':
		net = ResNet18(num_classes=num_classes)
	elif model == 'DenseNet121':
		net = DenseNet121(num_classes=num_classes)
	elif model == 'VGG19':
		net = VGG('VGG19',num_classes=num_classes)
	elif model == 'ShuffleNetV2':
		net = ShuffleNetV2(num_classes=num_classes)

	net.cuda()
	net = torch.nn.DataParallel(net)
	cudnn.benchmark = True
	return net

def load_data(dataset, batch_size):
	mean, std = data_loader.get_known_mean_std(dataset)
	if dataset == 'CIFAR10':
		return data_loader.getCIFAR10(mean, std, batch_size)
	else:
		return data_loader.getSVHN(mean, std, batch_size)

def test(testloader, net, criterion):
	net.eval()
	test_loss = 0
	correct = 0
	total = 0
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(testloader):
			inputs, targets = inputs.cuda(), targets.cuda()

			outputs = net(inputs)

			loss = criterion(outputs, targets)

			test_loss += loss.item()
			_, predicted = torch.max(outputs.data, 1)
			total += targets.size(0)

			correct += predicted.eq(targets).sum().item()
	
	test_loss = test_loss/idx
	test_acc = 100.*correct/total

	return test_loss, test_acc

def adjust_learning_rate(optimizer, epoch, decay_epochs):
	if epoch in decay_epochs:
		for param_group in optimizer.param_groups:
			new_lr = param_group['lr'] * 0.1
			param_group['lr'] = new_lr

def save_result(logname, epoch, train_loss, train_acc, test_loss, test_acc, optimizer):
	with open(logname, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow([epoch, optimizer.state_dict()['param_groups'][0]['lr'],
			train_loss, train_acc, test_loss, test_acc])

def save_model(pthname, net):
	torch.save(net.state_dict(), pthname + '_FINAL.pth')

def load_pretrained(net, filename):
	pthname = ('pths/' + filename)
	net.load_state_dict(torch.load(pthname))

def mixup_data(inputs, labels, alpha):

	lam = np.random.beta(alpha, alpha) if alpha > 0 else 1

	index = torch.randperm(inputs.size(0))

	mixed_inputs = lam*inputs + (1 - lam)*inputs[index, :]

	y_a, y_b = labels, labels[index]
	return mixed_inputs, y_a, y_b, lam

def fgsm_attack(net, criterion, inputs, labels, epsilon):
	net.eval()

	inputs.requires_grad = True
	outputs = net(inputs)
	loss = criterion(outputs, labels)
	loss.backward()

	data_grad = inputs.grad.data
	sign_data_grad = data_grad.sign()
	perturbed_image = inputs + epsilon*sign_data_grad
	net.train()
	return perturbed_image, sign_data_grad

def generate_noise(num = 1, ntype='gaussian'):
	if ntype == 'gaussian':
		return torch.randn(num, 3, 224, 224)
	elif ntype == 'uniform':
		return torch.FloatTensor(num, 3, 224, 224).uniform_(-1, 1)



