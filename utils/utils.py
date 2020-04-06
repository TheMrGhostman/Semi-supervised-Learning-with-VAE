import numpy as np 
import time
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import matplotlib.pyplot as plt


def paremeters_summary(model):
	s = []
	for p in model.parameters():
		dims = p.size()
		n = np.prod(p.size())
		s.append((dims, n))
	return s, np.sum([j for i,j in s])


def plot_loss(obj, figsize=(25,18), downsample=None):
	"""
	: param obj: 	Object type SVI or Trainer
	"""
	loss_train = obj.loss_history["train"]
	axe_t = np.arange(len(loss_train))/10
	loss_val = obj.loss_history["validation"]
	axe_v = np.arange(len(loss_val))
	if downsample!=None:
		axe_t = axe_t[::downsample]
		loss_train = loss_train[::downsample]
	plt.figure(figsize=figsize)
	plt.plot(axe_t, loss_train, lw=0.5)
	plt.plot(axe_v, loss_val, lw=0.5)
	plt.ylabel("loss")
	plt.xlabel("Epochs")


	if "val_accuracy" in obj.loss_history.keys():
		print("plotting accuracy")
		plt.figure("Accuracy", figsize=figsize)
		plt.plot(obj.loss_history["val_accuracy"])
		plt.ylabel("Accuracy")
		plt.ylim(0, 1)
		plt.grid(True)
	plt.show()
