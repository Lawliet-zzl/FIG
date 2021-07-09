from __future__ import print_function
import numpy as np
import torch
import csv
import torch.nn.functional as F
import data_loader

def tpr95(soft_IN, soft_OOD, precision):
	#calculate the falsepositive error when tpr is 95%

	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision # precision:200000

	total = 0.0
	fpr = 0.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		if tpr <= 0.9505 and tpr >= 0.9495:
			fpr += error2
			total += 1
	if total == 0:
		print('corner case')
		fprBase = 1
	else:
		fprBase = fpr/total
	return fprBase

def auroc(soft_IN, soft_OOD, precision):
	#calculate the AUROC
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision
	aurocBase = 0.0
	fprTemp = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		aurocBase += (-fpr+fprTemp)*tpr
		fprTemp = fpr
	aurocBase += fpr * tpr
	#improve
	return aurocBase

def auroc_XY(soft_IN, soft_OOD, precision):
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision
	aurocBase = 0.0
	fprTemp = 1.0
	tprs = []
	fprs = []
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fpr = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		tprs.append(tpr)
		fprs.append(fpr)
	return tprs, fprs

def auprIn(soft_IN, soft_OOD, precision):
	#calculate the AUPR

	precisionVec = []
	recallVec = []
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(start, end, gap):
		tp = np.sum(np.sum(X1 >= delta)) / np.float(len(X1))
		fp = np.sum(np.sum(Y1 >= delta)) / np.float(len(Y1))
		if tp + fp == 0: continue
		precision = tp / (tp + fp)
		recall = tp
		precisionVec.append(precision)
		recallVec.append(recall)
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def auprOut(soft_IN, soft_OOD, precision):
	#calculate the AUPR
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	auprBase = 0.0
	recallTemp = 1.0
	for delta in np.arange(end, start, -gap):
		fp = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		tp = np.sum(np.sum(Y1 < delta)) / np.float(len(Y1))
		if tp + fp == 0: break
		precision = tp / (tp + fp)
		recall = tp
		auprBase += (recallTemp-recall)*precision
		recallTemp = recall
	auprBase += recall * precision

	return auprBase

def detection(soft_IN, soft_OOD, precision):
	#calculate the minimum detection error
	Y1 = soft_OOD
	X1 = soft_IN
	end = np.max([np.max(X1), np.max(Y1)])
	start = np.min([np.min(X1),np.min(Y1)])
	gap = (end- start)/precision

	errorBase = 1.0
	for delta in np.arange(start, end, gap):
		tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
		error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
		errorBase = np.minimum(errorBase, (tpr+error2)/2.0)

	return errorBase

def get_softmax(net, dataloader):
	net.eval()
	res = np.array([])
	with torch.no_grad():
		for idx, (inputs, targets) in enumerate(dataloader):
			inputs= inputs.cuda()
			outputs = net(inputs)
			softmax_vals, predicted = torch.max(F.softmax(outputs.data, dim=1), dim=1)
			res = np.append(res, softmax_vals.cpu().numpy())
	return res

def test_OOD(soft_IN, soft_OOD, precision=200000):
	OOD_detection = np.array([0.0,0.0,0.0,0.0,0.0])
	OOD_detection[0] = auroc(soft_IN, soft_OOD, precision)*100
	OOD_detection[1] = auprIn(soft_IN, soft_OOD, precision)*100
	OOD_detection[2] = auprOut(soft_IN, soft_OOD, precision)*100
	OOD_detection[3] = tpr95(soft_IN, soft_OOD, precision)*100
	OOD_detection[4] = detection(soft_IN, soft_OOD, precision)*100
	return OOD_detection

def testDetection(logname, net, dataset, testloader, criterion, precision):

	mean, std = data_loader.get_known_mean_std(dataset)
	versions = ['crop', 'resize']
	soft_ID = get_softmax(net, testloader)

	with open(logname, 'a') as logfile:
		logwriter = csv.writer(logfile, delimiter=',')
		logwriter.writerow(['type', 'OOD', 'AUROC', 'AUPR(IN)', 'AUPR(OUT)', 'FPR(95)', 'Detection'])

	for version in versions:
		soft_OOD = get_softmax(net, data_loader.getTinyImagenet(mean, std, version))
		OOD_detection = test_OOD(soft_ID, soft_OOD, precision)
		with open(logname, 'a') as logfile:
			logwriter = csv.writer(logfile, delimiter=',')
			logwriter.writerow([ 'Final', 'TinyImagenet ' + version, OOD_detection[0], OOD_detection[1], OOD_detection[2], OOD_detection[3], OOD_detection[4]])