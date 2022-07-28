"""
Created on SAT May 16 2020
@author: Zhilin Zhao
"""
from __future__ import print_function

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

#import torchvision.models as models
from model_func import *
import OODMeasures

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument("--decay_epochs", nargs="+", type=int, default=[100, 150], 
	help="decay learning rate by decay_rate at these epochs")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="ResNet18", type=str, help='model type (default: ResNet18)')
parser.add_argument('--dataset', default="CIFAR10", type=str, help='dataset')
parser.add_argument('--name', default='0', type=str, help='name of run')
parser.add_argument('--seed', default=20200608, type=int, help='random seed')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--no-augment', dest='augment', action='store_false', help='use standard augmentation (default: True)')
parser.add_argument('--precision', default=100000, type=float)
parser.add_argument('--decay', default=5e-4, type=float, help='weight decay')

parser.add_argument('--num-classes', default=10, type=int, help='the number of classes (default: 100)')
parser.add_argument('--alg', default='Baseline', type=str, help='algorithm')

parser.add_argument('--T', default=1000, type=int, help='Sampling times (1~10 for training, 10001 for generating)')
parser.add_argument('--K', default=0.01, type=float, help='number of selected OODs: 0 ~ 0.2')

args = parser.parse_args()

def init_setting():

	# init seed
	if args.seed != 0:
		torch.manual_seed(args.seed)

	# Address
	if not os.path.isdir('results'):
		os.mkdir('results')
	if not os.path.isdir('pths'):
		os.mkdir('pths')

	filename = (args.dataset + '_' + args.model + '_' +  args.alg)
	pthname = ('pths/' + filename)
	logname = ('results/' + filename + '_LOG.csv')

	#if not os.path.exists(logname):
	with open(logname, 'w') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss', 'test acc'])

	print('Training: ' + filename)
	return logname, pthname

def train(trainloader, net, criterion, optimizer, epoch):
	net.train()
	train_loss = 0
	correct = 0
	total = 0

	for idx, (inputs, targets) in enumerate(trainloader):
		inputs, targets = inputs.cuda(), targets.cuda()

		outputs = net(inputs)

		loss = criterion(outputs, targets)

		train_loss += loss.item()
		_, predicted = torch.max(outputs.data, 1)
		total += targets.size(0)
		correct += (predicted == targets).sum().item()

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	train_loss = train_loss/idx
	train_acc = 100.*correct/total
	return train_loss, train_acc

def train_finetune(trainloader, net, criterion_ID, criterion_OOD, optimizer, sampler):
	net.train()

	ITER = 0
	iteration = 1000

	while ITER < iteration:

		for idx, (inputs_ID, targets) in enumerate(trainloader):

			targets = targets.cuda()
			num_ID = inputs_ID.size(0)
			num_OOD = max(int(targets.size(0)*args.K), 1)

			inputs_OOD = sampler(net, T=args.T, num_samples=num_OOD)
			
			inputs_OOD = inputs_OOD.cpu()
			inputs = torch.cat((inputs_ID, inputs_OOD), dim=0)
			inputs = inputs.cuda()
			outputs = net(inputs)
			outputs_ID = outputs[0:num_ID]
			outputs_OOD = outputs[num_ID: num_ID + num_OOD]

			loss_ID = criterion_ID(outputs_ID, targets)
			loss_OOD = criterion_OOD(outputs_OOD)
			loss = loss_ID + loss_OOD
			
			_, predicted = torch.max(outputs_ID.data, 1)
			total = num_ID
			correct = (predicted == targets).sum().item()

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			if ITER >= iteration:
				break
			ITER += 1

def Baseline(logname, net, criterion, trainloader, testloader):
	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)
	for epoch in tqdm(range(0, args.epoch)):
		train_loss, train_acc = train(trainloader, net, criterion, optimizer, epoch)
		test_loss, test_acc = test(testloader, net, criterion)
		save_result(logname, epoch, train_loss, train_acc, test_loss, test_acc, optimizer)
		adjust_learning_rate(optimizer, epoch, args.decay_epochs)
	return net

def FIG(logname, net, criterion, trainloader, testloader):
	filename = (args.dataset + '_' + args.model + '_Baseline_FINAL.pth')
	load_pretrained(net, filename)
	criterion_OOD = NHLoss()
	sampler = SamplerSGLD()
	optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=args.decay)
	train_finetune(trainloader, net, criterion, criterion_OOD, optimizer, sampler)
	test_loss, test_acc = test(testloader, net, criterion)
	save_result(logname, 0, 0, 0, test_loss, test_acc, optimizer)
	return net

def main():

	logname, pthname = init_setting()

	trainloader, testloader = load_data(dataset = args.dataset, batch_size = args.batch_size)

	net = build_model(model = args.model, num_classes = args.num_classes)
	criterion = nn.CrossEntropyLoss()

	if args.alg == 'Baseline':
		net = Baseline(logname, net, criterion, trainloader, testloader)
	else:
		net = FIG(logname, net, criterion, trainloader, testloader)

	OODMeasures.testDetection(logname, net, args.dataset, testloader, criterion, precision=args.precision)
	save_model(pthname, net)

if __name__ == '__main__':
	main()

