import numpy as np 
import time
import math

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


def gaussian_nll(y_true, mu, sigma):
	"""
	Negative log likelihood (loss function) of gaussian random variable

	: param y_true: 	target value
	: param mu: 		mean of distribution ... size = (batch_size, output_dim)
	: param sigma: 		standard deviation of distribution  ... size (batch_size, 1) <– same variance
	
	returns mean loss per sample (not per point)
	"""

	dim = mu.shape[1]/2
	var = sigma.pow(2)
	#print(mu.shape, sigma.shape)
	return torch.mean(torch.sum((y_true-mu).pow(2), axis=1)/(2*var) + dim*torch.log(var)) + dim*math.log(2*math.pi)


def sample_mse(y_true, y_pred):
	return torch.mean(torch.sum((y_true-y_pred).pow(2), axis=1))


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
	plt.show()


class SVI(nn.Module):
	"""
	Stochastic Variational Inference - for training of my VAEs
	"""
	def __init__(self, vae_model, optimizer, scheduler=None, loss_function="BCE", **kwargs):
		"""
		: param vae_model: 			Variational AutoEncoder model 
										which has self.encoder, self.decoder and forward function
		: param optimizer: 			Optimizer (e.g. torch.optim.Adam)
		: param scheduler:			Schduler for optimizer's learing rate. 
										(e.g. torch.optim.lr_scheduler.MultiStepLR)
		: param loss_function: 		Type of loss function ("BCE", "MSE" or "GaussianNLL")
		: param tensorboard:		Boolean parameter. Save graph and losses in epochs 
										to tensorboard (True/False).
		: param model_name:			If tonsorboard==True => model_name is name of tensorboar log.
		: param set_device:			If you want to use specific device to train NN.(e.g. torch.device("cpu"))
										(if set_device=None => 
											torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
		: param verbose:			Boolean parameter

		"""
		super(SVI, self).__init__()
		self.model = vae_model
		self.optimizer = optimizer
		self.scheduler = scheduler
		self.tensorboard = False
		self.model_name = None

		if loss_function not in ["BCE", "GaussianNLL", "MSE"]:
			raise ValueError("Unknown loss function")
		else:
			self.loss_fn = loss_function
		self.loss_history = {"train":[], "validation":[]}

		#optional params
		if kwargs.get("tensorboard") == True:
			self.tensorboard = True
			if kwargs.get("model_name")!= None:
				self.tb = SummaryWriter(comment=kwargs.get("model_name"))
			else:
				self.tb = SummaryWriter()
		
		if kwargs.get("set_device")!=None:
			self.device = kwargs.get("set_device")
		else:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.model.to(self.device)

		if kwargs.get("verbose")==True:
			self.verbose = True
			print(self.device) 
		else:
			self.verbose=False

	def loss_function(self, y_pred, y_true, mu, sigma):
		if self.loss_fn == "BCE":
			reconstruction_loss = nn.BCELoss(reduction='sum')(y_pred, y_true)
		elif self.loss_fn == "GaussianNLL":
			assert np.shape(y_pred)[0]==2
			mu_out, sigma_out = y_pred
			reconstruction_loss = gaussian_nll(y_true=y_true, mu=mu_out, sigma=sigma_out)
			#self.loss_history["MSE"].append(sample_mse(y_true=y_true, y_pred=mu_out).item())
		elif self.loss_fn == "MSE":
			reconstruction_loss = sample_mse(y_true=y_true, y_pred=y_pred)
			#nn.MSELoss(reduction='sum')(y_pred, y_true)

		KLD = - 0.5 * torch.mean(torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2), axis=1)) 
		# střední hodnota přes batch, ale suma přes latentní prostor
		#print(reconstruction_loss.item(), KLD.item())
		return reconstruction_loss + KLD

	def evaluate(self, y_pred, y_true, mu, sigma):
		"""
			need to be fixed
		"""
		if self.loss_fn == "BCE":
			reconstruction_loss = nn.BCELoss(reduction='sum')(y_pred, y_true)
		if self.loss_fn == "GaussianNLL":
			assert np.shape(y_pred)[0]==2
			mu_out, sigma_out = y_pred
			reconstruction_loss = gaussian_nll(y_true=y_true, mu=mu_out, sigma=sigma_out)
		if self.loss_fn == "MSE":
			reconstruction_loss = nn.MSELoss(reduction='sum')(y_pred, y_true)


		KLD = - 0.5 * torch.mean(torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2), axis=1)) 
		pass

	def forward(self, epochs, train_loader, validation_loader, flatten=False):
		#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		#self.model.to(device)
		n_batches = len(train_loader)
		print_every = n_batches//10

		if not isinstance(epochs, range):
			epochs = range(epochs)

		n_epochs = max(epochs)+1

		for epoch in epochs:
			self.model.train()
			train_loss = 0
			for i, (train_sample, _) in enumerate(train_loader, 0):
				train_sample = train_sample.to(self.device)
				#=================forward================
				output, mu, sigma = self.model.forward(train_sample)
				"""
				Chci maximalizovat variational lower bound (ELBO) =>
				minimalizuju -ELBO tzn. min[-E(log(p(x|z))) + KL(q(z|x)||p(z))]  
				= min(BCE + KL) resp. min(-log[N(x|mu_out, sigma_out)] + KLD)
				"""
				if flatten:
					train_sample = train_sample.view(-1, self.model.original_dim)
				loss = self.loss_function(y_pred=output, y_true=train_sample, mu=mu, sigma=sigma)
				# KLD = KLD / (batch_size * 784) 
							
				train_loss += loss.item()
				#=================backward===============
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				#=================log====================
				if ((i + 1) % print_every == 0): # and isinstance(history_train_loss, list)
					self.loss_history["train"].append(loss.item())
					if self.tensorboard:
						self.tb.add_scalar("Loss/Train_per_batch", loss.item(), epoch+(i+1)/n_batches)
						if epoch==0:
							self.tb.add_graph(self.model, train_sample)

			self.model.eval()
			validation_loss=0
			with torch.no_grad():
				for j, (validation_sample, _) in enumerate(validation_loader, 0):
					validation_sample = validation_sample.to(self.device)
					output, mu, sigma = self.model.forward(validation_sample)

					if flatten:
						validation_sample = validation_sample.view(-1, self.model.original_dim)

					validation_loss += self.loss_function(y_pred=output, y_true=validation_sample, mu=mu, sigma=sigma).item()

			validation_loss /= len(validation_loader)
			self.loss_history["validation"].append(validation_loss)

			if self.verbose:
				print("Epoch [{}/{}], average_loss:{:.4f}, validation_loss:{:.4f}"\
					  .format(epoch+1, n_epochs, train_loss/n_batches, validation_loss))
			if self.tensorboard:
				self.tb.add_scalar("Loss/train", train_loss/n_batches, epoch+1)
				self.tb.add_scalar("Loss/validation", validation_loss, epoch+1)
			if self.scheduler !=None:
				self.scheduler.step()

		return self.loss_history


class Trainer(nn.Module):
	"""
	Class Trainer was made for easier training of Neural Networks for classification or regression. 
	Insted of defining whole trining precedure every time, it's now possible to do it in 2-3 lines of code.
	"""
	def __init__(self, model, optimizer, loss_function, scheduler=None, **kwargs):
		"""
		: param vae_model: 			Variational AutoEncoder model 
										which has self.encoder, self.decoder and forward function
		: param optimizer: 			Optimizer (e.g. torch.optim.Adam)
		: param loss_function: 		Type of loss function (nn.CrossEntropyLoss()) 
		: param scheduler:			Schduler for optimizer's learing rate. 
										(e.g. torch.optim.lr_scheduler.MultiStepLR)
		: param tensorboard:		Boolean parameter. Save graph and losses in epochs 
										to tensorboard (True/False).
		: param model_name:			If tonsorboard==True => model_name is name of tensorboar log.
		: param set_device:			If you want to use specific device to train NN.(e.g. torch.device("cpu"))
										(if set_device=None => 
											torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
		: param verbose:			Boolean parameter

		"""
		super(Trainer, self).__init__()
		self.model = model
		self.optimizer = optimizer
		self.scheduler = scheduler

		self.loss_fn = loss_function
		self.loss_history = {"train":[], "validation":[], "val_accuracy":[]}

		#optional params
		if kwargs.get("tensorboard") == True:
			self.tensorboard = True
			if kwargs.get("model_name")!= None:
				self.tb = SummaryWriter(comment=kwargs.get("model_name"))
			else:
				self.tb = SummaryWriter()
		else: 
			self.tensorboard = False
		
		if kwargs.get("set_device")!=None:
			self.device = kwargs.get("set_device")
		else:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.model.to(self.device)

		if kwargs.get("verbose")==True:
			self.verbose = True
			print(self.device) 
		else:
			self.verbose=False

	def tensorboard_save(self, epoch, tr_loss, val_loss, acc=None):
		"""
		function for saving informations to tensorboard
		"""
		self.tb.add_scalar("Loss/train", tr_loss, epoch)
		self.tb.add_scalar("Loss/validation", val_loss, epoch)
		"""
		for i in range(len(self.model.model)-1): # iteration through layers
			self.tb.add_histogram(f"layer_{i}/wight", self.model.model[i].layer.weight, epoch)
			self.tb.add_histogram(f"layer_{i}/bias", self.model.model[i].layer.bias, epoch)
			self.tb.add_histogram(f"layer_{i}/wight_grad", self.model.model[i].layer.weight.grad, epoch)
			self.tb.add_histogram(f"layer_{i}/bias_grad", self.model.model[i].layer.bias.grad, epoch)
		"""
		if acc!=None:
			self.tb.add_scalar("Accurary/validation", acc, epoch)

	def forward(self, epochs, train_loader, validation_loader):
		n_batches = len(train_loader)
		print_every = n_batches//10

		if not isinstance(epochs, range):
			epochs = range(epochs)
		n_epochs = max(epochs)+1

		for epoch in epochs:
			self.model.train()
			train_loss = 0
			for i, (train_sample, y_true) in enumerate(train_loader, 0):
				train_sample = train_sample.to(self.device)
				y_true = y_true.to(self.device)
				#=================forward================
				y_pred = self.model.forward(train_sample)

				loss = self.loss_fn(y_pred, y_true)
							
				train_loss += loss.item()
				#=================backward===============
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				#=================log====================
				if ((i + 1) % print_every == 0): # and isinstance(history_train_loss, list)
					self.loss_history["train"].append(loss.item())
					if self.tensorboard and epoch==0:
						if isinstance(self.model, nn.Sequential):
							self.tb.add_graph(self.model, train_sample)
						else if isinstance(self.model.model, nn.Sequential):
							self.tb.add_graph(self.model.model, train_sample)
						else:
							self.tb.add_graph(self.model, train_sample)
						#self.tb.add_scalar("Loss/Train_per_batch", loss.item(), epoch+0.1*(i+1)//print_every)

			validation_loss=0
			self.model.eval( )
			with torch.no_grad():
				for j, (validation_sample, y_valid_true) in enumerate(validation_loader, 0):
					validation_sample = validation_sample.to(self.device)
					y_valid_true = y_valid_true.to(self.device)
					y_valid_pred = self.model.forward(validation_sample)

					validation_loss += self.loss_fn(y_valid_pred, y_valid_true).item()

			validation_loss /= len(validation_loader)
			self.loss_history["validation"].append(validation_loss)
			acc = accuracy_score(y_valid_true.cpu().detach(), torch.argmax(y_valid_pred.cpu().detach(), axis=1))
			self.loss_history["val_accuracy"].append(acc)

			if self.verbose:
				print("Epoch [{}/{}], average_loss:{:.4f}, validation_loss:{:.4f}, val_accuracy:{:,.4f}"\
						.format(epoch+1, n_epochs, train_loss/n_batches, validation_loss, acc))
			if self.tensorboard:
				self.tensorboard_save(epoch=epoch, tr_loss=train_loss/n_batches, val_loss=validation_loss, acc=acc)
			if self.scheduler!=None:
				self.scheduler.step()

		return self.loss_history
