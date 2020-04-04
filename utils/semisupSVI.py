import math
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F

class one_hot(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.class_matrix = torch.diag(torch.ones(n_classes))

    def __call__(self, p):
        return self.class_matrix[p]


def gaussian_nll(y_true, mu, sigma, reduction="mean"):
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
	if reduction=="mean":
		return - (torch.mean(torch.sum((y_true-mu).pow(2), axis=1)/(2*var) + dim*torch.log(var)) + dim*math.log(2*math.pi))
	if reduction=="none":
		return - (torch.sum((y_true-mu).pow(2), axis=1)/(2*var) + dim*torch.log(var) + dim*math.log(2*math.pi))


class SS_SVI(nn.Module):
	def __init__(self, model, likelihood="GaussianNLL", **kwargs):
		super(SS_SVI, self).__init__()
		self.model = model

		if likelihood not in ["BCE", "GaussianNLL", "MSE"]:
			raise ValueError("Unknown likelihood")
		else:
			self.likelihood = likelihood

	def reconstruction_loss(self, y_pred, y_true):
		if self.likelihood == "BCE":
			reconstruction_loss = nn.BCELoss(reduction='sum')(y_pred, y_true)
		elif self.likelihood == "GaussianNLL":
			assert np.shape(y_pred)[0]==2
			mu_out, sigma_out = y_pred
			reconstruction_loss = gaussian_nll(y_true=y_true, mu=mu_out, sigma=sigma_out)
			#self.loss_history["MSE"].append(sample_mse(y_true=y_true, y_pred=mu_out).item())
		elif self.loss_fn == "MSE":
			reconstruction_loss = sample_mse(y_true=y_true, y_pred=y_pred)
			#nn.MSELoss(reduction='sum')(y_pred, y_true)

	def forward(self, X, y=None):
		supervised = False if y in None else True

		#	y-> z 
		#	|   |
		#   ->x<-

		if supervised:
			p_y_pred = self.model.classify(X)

			y_oh = one_hot(self.model.y_dim)(y)
			z, z_mu, z_sigma = self.model.encoder(torch.cat([X,y_oh], axis=1)) 

			X_hat = self.model.decoder(torch.cat([z, y_oh], axis=1))

			# losses
			# classification loss
			loss_clf = nn.CrossEntropyLoss()(p_y_pred, y)

			# log p_y
			y_prior = 1/self.model.y_dim *torch.ones_like(y_oh)
			log_py = torch.mean(torch.log(y_prior),axis=1)

			# K-L divergence
			kld = - 0.5 * torch.mean(torch.sum(1 + torch.log(z_sigma.pow(2)) - z_mu.pow(2) - z_sigma.pow(2), axis=1))

			# log p_x ... reconstruction error
			log_px = self.reconstruction_loss(X_hat, X)

			# final loss of current flow
			likelihood = log_px + log_py - kld 

			return likelihood, loss_clf # returns scalar loss

		else:
			X_epanded = torch.cat(self.model.y_dim*[X])

			# E[q(y|x)] = sum q(y|x) <- monte carlo improvement <- inaccurate decisions on start of training
			y_oh = []
			for i in range(self.model.y_dim):
			   	y_oh.append(i*torch.ones(X.shape[0]))
			y_oh_expanded = one_hot(self.model.y_dim)(torch.cat(y_oh,axis=0).long())

			z, z_mu, z_sigma = self.model.encoder(torch.cat([X_expanded,y_oh_expanded], axis=1)) 
			
			X_hat = self.model.decoder(torch.cat([z, y_oh_expanded], axis=1))

			y_pred = self.model.classify(X)
			p_y_pred = F.softmax(y_pred)

			# losses
			kld = - 0.5 * torch.mean(torch.sum(1 + torch.log(z_sigma.pow(2)) - z_mu.pow(2) - z_sigma.pow(2), axis=1))

			log_px = self.reconstruction_loss(X_hat, X_expanded, reduction="none")

			y_prior = 1/self.model.y_dim *torch.ones_like(p_y_pred)
			log_py = - nn.CrossEntropyLoss(y_prior, p_y_pred, reduction="none")

			likelihood = log_px + log_py - kld 

			likelihood = torch.multiply(p_y_pred, likelihood.view(self.model.y_dim, X.shape[0]).T - torch.log(p_y_pred))

			likelihood = torch.sum(likelihood, axis=1)

			return torch.mean(likelihood)


class Generative_Model_Trainer(nn.Module)
	def __init__(self, model, optimizer, scheduler=None, lr=1e-3, **kwargs):
		self.model = model
		self.optimizer = optimizer(self.model.parameters(), lr=1e-3)
		self.elbo = SS_SVI(self.model, likelihood="GaussianNLL", set_device=set_device)

		if kwargs.get("set_device")!=None:
			self.device = kwargs.get("set_device")
		else:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

		self.model.to(self.device)

	def reset_losses(self):
		self.loss_history = {
							"train_total_loss":0, 
							"train_classifier_loss":0, 
							"train_supervised_loss":0, 
							"train_unsupervised_loss":0,
							"validation_total_loss":0,
							"validation_classifier_loss":0,
							"validation_supervised_loss":0,
							"validation_unsupervised_loss":0
							 }

	def forward(self, epochs, supervised_dataset, unsupervised_dataset, validation_dataset, batch_size):
		if not isinstance(epochs, range):
			epochs = range(epochs)
		n_epochs = max(epochs)+1

		unsupervised = torch.utils.data.DataLoader(
							dataset=unsupervised_dataset, 
							batch_size=batch_size, 
							shuffle=False, 
							sampler=torch.utils.data.RandomSampler(
								unsupervised_dataset, 
								replacement=False
								)
							)
		supervised = torch.utils.data.DataLoader(
							dataset=supervised_dataset,
							batch_size=batch_size, 
							shuffle=False, 
							sampler=torch.utils.data.RandomSampler(
								supervised_dataset, 
								replacement=True,
								num_samples=unsupervised_dataset.shape[0]
								)
							)

		validation = torch.utils.data.DataLoader(dataset=validation, batch_size=batch_size)

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
				L, CLF = self.elbo(X_sup, y_sup)
				L = -L
				U = -self.elbo(X_unsup)

				alpha = 0.1 * batch_size
				J = L + U + alpha*CLF

				# logging losses
				self.loss_history["train_total_loss"] += J.detach().item()
				self.loss_history["train_supervised_loss"] += L.detach().item()
				self.loss_history["train_classifier_loss"] += CLF.detach().item()
				self.loss_history["train_unsupervised_loss"] += U.detach().item()

				# ============ backward ============
				self.optimizer.zero_grad()
				J.backward()
				self.optimizer.step()

			# ============= validation =============
			self.model.eval()
			with torch.no_grad():
				for x, y in validation:
					x = x.to(self.device)
					y = y.to(self.device)

					self.model





		"""
			for i in range(self.num_labels):
	            y_us = i*tf.ones([tf.shape(self.x_u)[0]], tf.int32)
	            y_us = tcl.one_hot_encoding(y_us, num_classes = self.num_labels)
	            
	            z_u, mu_u, logvar_u = self.encode(self.x_u, y_us, reuse = True)
	            mu_recon_u, logvar_recon_u = self.decode(z_u, y_us, reuse = True)
	            
	            _likelihood_u = self.likelihood(self.x_u, mu_recon_u, logvar_recon_u,\
	                                y_us, z_u, mu_u, logvar_u)
	            _likelihood_u = tf.expand_dims(_likelihood_u, 1)
	            
	            if i == 0:
	                likelihood_u = tf.identity( _likelihood_u )
	            else:
	                likelihood_u = tf.concat([likelihood_u, _likelihood_u], 1)

	        scores_u = self.classify(self.x_u, reuse = True)
			y_u_prob = tf.nn.softmax(scores_u, dim=-1)

			# add the H(q(y|x))
			likelihood_u = tf.multiply(y_u_prob, likelihood_u + -tf.log(y_u_prob)) 
			likelihood_u = tf.reduce_sum(likelihood_u, 1)

			alpha = 0.1 * self.batch_size
			self.loss_clf = tf.reduce_sum(self.loss_clf, 0)
			self.loss_l = - tf.reduce_sum(likelihood_l, 0)
			self.loss_u = - tf.reduce_sum(likelihood_u, 0)
			self.loss = (self.loss_l + alpha* self.loss_clf + self.loss_u)/self.batch_size
		"""
		"""		
		def likelihood(self, x, mu_x, logvar_x, y, z, mu_z, logvar_z):
	        # uniform prior
	        prior_y = (1. / self.num_labels) * tf.ones([tf.shape(x)[0], 10], tf.float32)
	        logpy = - tf.nn.softmax_cross_entropy_with_logits(logits = prior_y, labels = y)
	        
	        kld = tf.reduce_sum(logpdf.KLD(mu_z, logvar_z), 1)
	        logpx = tf.reduce_sum(logpdf.gaussian(x, mu_x, logvar_x), 1)
	        likelihood = logpx + logpy - kld  
	        return likelihood


		y_oh = one_hot(y) if y not None else self.model.classify(X)
		z, z_mu, z_sigma = self.model.encoder(X)

		reconstruction = self.model.decoder(torch.cat([z,y_oh], axis=1))

		kld = - 0.5 * torch.mean(torch.sum(1 + torch.log(z_sigma.pow(2)) - z_mu.pow(2) - z_sigma.pow(2), axis=1))

		elbo_x_y = self.reconstruction_loss(y_pred=reconstruction, y_true=X) - kld +  
		"""






