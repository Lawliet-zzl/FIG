import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import os
import random

def getCIFAR10(mean, std, batch_size):
	transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])
	trainset = datasets.CIFAR10(root='../data_FIG/CIFAR10', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=9)
	testset = datasets.CIFAR10(root='../data_FIG/CIFAR10', train=False, download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=9)
	return trainloader, testloader

def getSVHN(mean, std, batch_size):
	transform = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
		])
	trainset = datasets.SVHN(root='../data_FIG/SVHN', split='test', download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=9)
	testset = datasets.SVHN(root='../data_FIG/SVHN', split='train', download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=9)
	return trainloader, testloader

def get_known_mean_std(dataset):
	if dataset == 'CIFAR10':
		mean = (0.4914, 0.4822, 0.4465)
		std = (0.2470, 0.2435, 0.2616)
	elif dataset == 'SVHN':
		mean = (0.4377, 0.4438, 0.4728)
		std = (0.1980, 0.2010, 0.1970)
	return mean, std
	
def getTinyImagenet(mean, std, version):
	transform = transforms.Compose([
	transforms.Resize((32,32)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])
	if version == 'crop':
		print("Loading TinyImageNet (c) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data/TinyImageNet_crop',transform=transform)
	elif version == 'resize':
		print("Loading TinyImageNet (r) dataset (out-of-distribution)")
		dataset = datasets.ImageFolder('../data/TinyImageNet_resize',transform=transform)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=9)
	return dataloader







