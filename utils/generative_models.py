import math
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from .losses import Gaussian_NLL


class One_Hot(nn.Module):
	def __init__(self, n_classes):
		super(One_Hot, self).__init__()
		self.n_classes = n_classes
		self.register_buffer("class_matrix", torch.diag(torch.ones(n_classes)))

	def forward(self, p):
		return self.class_matrix[p]


class SemiSupervisedGenerativeModel(nn.Module):
	def __init__(self, encoder, decoder, classifier, y_dim, likelihood=Gaussian_NLL, include_y=False):
		super(SemiSupervisedGenerativeModel, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.classifier = classifier
		self.likelihood = likelihood(reduction="none")
		self.y_dim = y_dim
		self.one_hot = One_Hot(self.y_dim)
		self.include_y = include_y
		
	def elbo(self, X, y=None):
		supervised = False if y is None else True      
		if supervised:
			p_y_pred = self.classifier(X)
			# y to one hot encoding
			y_oh = self.one_hot(y)
			# encoding to latent space
			if self.include_y:
				z, z_mu, z_sigma = self.encoder(X, y_oh) #torch.cat([X,y_oh], axis=1)
			else:
				z, z_mu, z_sigma = self.encoder(X) 
			# decoding or reconstructing input
			X_hat = self.decoder(torch.cat([z, y_oh], axis=1))
			# compute ELBO
			# classification loss
			loss_clf = nn.CrossEntropyLoss()(p_y_pred, y)
			# log p_y of prior distribution
			log_py = torch.log(torch.tensor(1./self.y_dim))
			# K-L divergence
			kld = - 0.5 * torch.sum(1 + torch.log(z_sigma.pow(2)) - z_mu.pow(2) - z_sigma.pow(2), axis=1)
			# log p_x ... reconstruction error
			log_px = self.likelihood(X_hat, X)
			#print(X_hat.shape)
			#print(X.shape)
			# standard ELBO_xy
			ELBO = log_px + log_py - kld 
			return torch.mean(ELBO), loss_clf # returns scalar losses        
		else:
			# E[q(y|x)] = sum q(y|x) <- monte carlo improvement <- inaccurate decisions on start of training
			X_expanded = torch.cat(self.y_dim*[X]).float()        
			y_oh = []
			for i in range(self.y_dim):
				y_oh.append(i*torch.ones(X.shape[0]))
			y_oh_expanded = self.one_hot(torch.cat(y_oh,axis=0).long())
			# encoding to latent space
			if self.include_y:
				z, z_mu, z_sigma = self.encoder(X_expanded,y_oh_expanded) #torch.cat([X_expanded,y_oh_expanded], axis=1)
			else:
				z, z_mu, z_sigma = self.encoder(X_expanded) 
			# reconstructing input
			X_hat = self.decoder(torch.cat([z, y_oh_expanded.float()], axis=1))
			# predicting class of input (during trianing we can predict class after decoding thanks to MC treatment)
			y_pred = self.classifier(X)
			p_y_pred = F.softmax(y_pred, dim=1)
			# compute ELBO
			# log p_y of prior distribution
			log_py = torch.log(torch.tensor(1./self.y_dim))
			# K-L divergence
			kld = - 0.5 * torch.sum(1 + torch.log(z_sigma.pow(2)) - z_mu.pow(2) - z_sigma.pow(2), axis=1)
			# log p_x ... reconstruction error
			log_px = self.likelihood(X_hat, X_expanded)
			# standard ELBO_xy
			ELBO = log_px + log_py - kld 
			# E_q(y|x)[-ELBO_xy - log(q(y|x))] = E_q(y|x)[-ELBO_xy] + H(q(y|x)) ... matrix version
			ELBO = torch.mul(p_y_pred, ELBO.view(self.y_dim, X.shape[0]).T - torch.log(p_y_pred+1e-8))
			# final ELBO_x
			ELBO = torch.sum(ELBO, axis=1)
			return torch.mean(ELBO)
   
	def encode(self, X, y):
		if self.include_y:
			y_oh = self.one_hot(y) if y.shape[1]!=self.y_dim else y
			z, z_mu, z_sigma = self.encoder(torch.cat([X,y_oh], axis=1)) 
		else:
			z, z_mu, z_sigma = self.encoder(X)
		return z, z_mu, z_sigma
	
	def decode(self, Z, y):
		y_oh = self.one_hot(y) if y.shape[1]!=self.y_dim else y
		return self.decoder(torch.cat([Z, y_oh], axis=1))
	
	def classify(self, X):
		return self.classifier(X)

	def forward(self, x, y=None):
		supervised = False if y is None else True
		if not supervised:
			y = self.classifier(x)
		z, z_mu, z_sigma = self.encode(x,y)
		return self.decode(z, y)
				

# class made for training SemiSupervidesGenerativeModel
class Generative_Model_Trainer(nn.Module):
	def __init__(self, model, optimizer, scheduler=None, lr=1e-3, **kwargs):
		super(Generative_Model_Trainer, self).__init__()
		self.model = model
		self.optimizer = optimizer(self.model.parameters(), lr=1e-3)
		
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") if kwargs.get("set_device")==None else kwargs.get("set_device")
		print("Using device {}".format(self.device))
		
		self.scheduler = scheduler

		#optional params
		if kwargs.get("tensorboard") == True:
			self.tensorboard = True
			if kwargs.get("model_name")!= None:
				self.tb = SummaryWriter(comment=kwargs.get("model_name"))
			else:
				self.tb = SummaryWriter()
		else: 
			self.tensorboard = False
		
		self.model = self.model.to(self.device)
		
		self.verbose = kwargs.get("verbose") if kwargs.get("verbose") != None else False

		

	def reset_losses(self):
		self.loss_history = {
							"train_total_loss":0., 
							"train_classifier_loss":0., 
							"train_supervised_loss":0., 
							"train_unsupervised_loss":0.,
							"validation_total_loss":0.,
							"validation_classifier_loss":0.,
							"validation_supervised_loss":0.,
							"validation_unsupervised_loss":0.,
							"validation_accuracy":0.
							 }
	
	def tensorboard_push_losses(self, epoch, n_train_batches, n_valid_batches):
		"""
		function for saving losses to tensorboard
		"""
		self.tb.add_scalar("Loss/train_total_loss", self.loss_history["train_total_loss"]/n_train_batches, epoch)
		self.tb.add_scalar("Loss/train_classifier_loss", self.loss_history["train_classifier_loss"]/n_train_batches, epoch)
		self.tb.add_scalar("Loss/train_supervised_loss", self.loss_history["train_supervised_loss"]/n_train_batches, epoch)
		self.tb.add_scalar("Loss/train_unsupervised_loss", self.loss_history["train_unsupervised_loss"]/n_train_batches, epoch)
		
		self.tb.add_scalar("Loss/validation_total_loss", self.loss_history["validation_total_loss"]/n_valid_batches, epoch)
		self.tb.add_scalar("Loss/validation_classifier_loss", self.loss_history["validation_classifier_loss"]/n_valid_batches, epoch)
		self.tb.add_scalar("Loss/validation_supervised_loss", self.loss_history["validation_supervised_loss"]/n_valid_batches, epoch)
		self.tb.add_scalar("Loss/validation_unsupervised_loss", self.loss_history["validation_unsupervised_loss"]/n_valid_batches, epoch)
		self.tb.add_scalar("Accuracy/validation", self.loss_history["validation_accuracy"]/n_valid_batches, epoch)


	def forward(self, epochs, supervised_dataset, unsupervised_dataset, validation_dataset, batch_size):
		if not isinstance(epochs, range):
			epochs = range(epochs)
		n_epochs = max(epochs)+1

		unsupervised = torch.utils.data.DataLoader(
							dataset=torch.tensor(unsupervised_dataset).float(), 
							batch_size=batch_size//2, 
							shuffle=False, 
							sampler=torch.utils.data.RandomSampler(
								unsupervised_dataset, 
								replacement=False
								)
							)
		supervised = torch.utils.data.DataLoader(
							dataset=supervised_dataset,
							batch_size=batch_size//2, 
							shuffle=False, 
							sampler=torch.utils.data.RandomSampler(
								supervised_dataset, 
								replacement=True,
								num_samples=unsupervised_dataset.shape[0]
								)
							)

		validation = torch.utils.data.DataLoader(dataset=validation_dataset, batch_size=batch_size)

		for epoch in epochs:
			self.model.train()
			self.reset_losses()
			# ============== training ==============
			for i in range(len(unsupervised)):
				X_unsup = next(iter(unsupervised))
				X_sup, y_sup = next(iter(supervised))

				X_unsup = X_unsup.to(self.device)
				X_sup = X_sup.to(self.device)
				y_sup = y_sup.to(self.device)

				# ============== forward ===========
				L, CLF = self.model.elbo(X_sup, y_sup)
				L = -L
				U = -self.model.elbo(X_unsup)

				alpha = 0.1 * (batch_size//2)*2  # correction -> even numbers
				J = L + U + alpha*CLF

				# logging losses
				self.loss_history["train_total_loss"] += J.detach().item()
				self.loss_history["train_supervised_loss"] += L.detach().item()
				self.loss_history["train_classifier_loss"] += CLF.detach().item()
				self.loss_history["train_unsupervised_loss"] += U.detach().item()
				
				#print(f"J: {np.exp(J.detach().item())}, L: {L.detach().item()}, U: {U.detach().item()}, CLF: {CLF.detach().item()}")

				# ============ backward ============
				self.optimizer.zero_grad()
				J.backward()
				self.optimizer.step()

			# ============= validation =============
			self.model.eval()
			with torch.no_grad():
				acc = 0
				for x, y in validation:
					x_sup = x[:batch_size//2]
					y_sup = y[:batch_size//2]
					x_unsup = x[batch_size//2:]
					
					x_sup = x_sup.to(self.device)
					y_sup = y_sup.to(self.device)
					x_unsup = x_unsup.to(self.device)

					# ============== forward ===========
					L, CLF = self.model.elbo(x_sup, y_sup)
					L = -L
					U = -self.model.elbo(x_unsup)

					alpha = 0.1 * (batch_size//2)*2  # correction -> even numbers
					J = L + U + alpha*CLF
					# logging losses
					self.loss_history["validation_total_loss"] += J.detach().item()
					self.loss_history["validation_supervised_loss"] += L.detach().item()
					self.loss_history["validation_classifier_loss"] += CLF.detach().item()
					self.loss_history["validation_unsupervised_loss"] += U.detach().item()
					
					# classification
					x = x.to(self.device)
					y = y.to(self.device)
					y_valid_pred = self.model.classify(x)
					self.loss_history["validation_accuracy"] += accuracy_score(y.cpu().detach(), torch.argmax(y_valid_pred.cpu().detach(), axis=1))
			if self.verbose:
				print("Epoch [{}/{}], average_loss:{:.4f}, validation_loss:{:.4f}, val_accuracy:{:,.4f}"\
						.format(epoch+1, n_epochs,self.loss_history["train_total_loss"]/len(unsupervised), self.loss_history["validation_total_loss"]/len(validation), self.loss_history["validation_accuracy"]/len(validation)))
					
			if self.tensorboard:
				self.tensorboard_push_losses(epoch=epoch, n_train_batches=len(unsupervised), n_valid_batches=len(validation))
				
			if self.scheduler!=None:
				self.scheduler.step()
