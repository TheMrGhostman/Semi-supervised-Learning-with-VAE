import numpy as np 
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F 

from collections import OrderedDict
from IPython.core.debugger import set_trace
# Layers

class VariationalLayer(nn.Module):
	"""
	Variational Layer with reparametrization trick. 
	It's used as bottleneck of Variational AutoEncoder ie. output of encoder.
	
	"Linear layer for variational inference with reparametrization trick"
	"""
	def __init__(self, in_features, out_features, bias=True, return_KL=False):
		"""	
		: param in_features: 		Number of input features (number of neurons on input)
		: param out_features:		Number of output features (number of neurons on output)
		: param bias:				Include bias - True/False
		: param return_KL			Compute and return KL divergence - True/False (old models need it)
		"""
		super(VariationalLayer, self).__init__()
		self.mu = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
		self.rho = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
		self.softplus = nn.Softplus()
		self.return_KL = return_KL

	def forward(self, x_in):
		mu = self.mu(x_in)
		sigma = self.softplus(self.rho(x_in))
		eps = torch.randn_like(sigma)
		if self.return_KL:
			KL_div = - 0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2)) # kl_div 
			return mu + sigma*eps, KL_div, mu, sigma
		return mu + sigma*eps, mu, sigma


class VariationalDecoderOutput(nn.Module):
	"""
	Variational Layer where variances are same and not equal to 1
	
	3D example:
		mu = (mu_1, mu_2, mu_3).T

		C = (sigma, 0    , 0    )
			(0    , sigma, 0    )
			(0    , 0    , sigma)
	"""
	def __init__(self, in_features, out_features, bias=True):
		"""	
		: param in_features: 		Number of input features (number of neurons on input)
		: param out_features:		Number of output features (number of neurons on output)
		: param bias:				Include bias - True/False
		"""
		super(VariationalDecoderOutput, self).__init__()
		self.mu = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
		self.rho = nn.Linear(in_features=in_features, out_features=1, bias=bias)
		self.softplus = nn.Softplus()

	def forward(self, x_in):
		mu = self.mu(x_in)
		sigma = self.softplus(self.rho(x_in))
		return mu, sigma


class ConvTransposeDecoderOutput(nn.Module):
	def __init__(self, in_channels, in_features, out_features, kernel_size, stride=1, padding=0, bias=True):
		super(ConvTransposeDecoderOutput, self).__init__()
		self.mu = nn.ConvTranspose1d(
				in_channels=in_channels, 
				out_channels=1, 
				kernel_size=kernel_size, 
				stride=stride, 
				padding=padding
			)
		self.rho = nn.Linear(in_features=in_features, out_features=1, bias=bias)
		self.flatten = Flatten(out_features=in_features)
		self.flatten_mu = Flatten(out_features=out_features)
		self.softplus = nn.Softplus()

	def forward(self, x_in):
		"""
		x must be shape ... (-1, in_channels, size)
		"""
		mu = self.mu(x_in)
		x = self.flatten(x_in)
		sigma = self.softplus(self.rho(x))
		return self.flatten_mu(mu), sigma


class RecurrentDecoderOutput(nn.Module):
	def __init__(self, in_features, sequence_len, out_features, bias=True):
		"""	
		: param in_features: 		Number of input features (number of neurons on input)
		: param sequence_len		Length of sequence
		: param out_features:		Number of output features (number of neurons on output)
		: param bias:				Include bias - True/False
		"""
		super(RecurrentDecoderOutput, self).__init__()

		self.in_features = in_features
		self.sequence_len = sequence_len
		self.mu = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
		self.rho = nn.Linear(in_features=in_features*sequence_len, out_features=1, bias=bias)
		self.softplus = nn.Softplus()

	def forward(self, x_in):
		"""
		x must be shape ... (sequence_len, batch_size, in_features) 
		"""
		mu = self.mu(x_in) # now is shape (sequence_len, bactch_size, n_features == 1 )
		mu = mu[:,:,-1].T # (batch_size, sequence_len)

		x_in = x_in.permute(1,0,2).reshape(-1, self.sequence_len*self.in_features) #(batch_size, sequence_len*in_features) = (100, 128*160)
		sigma = self.softplus(self.rho(x_in))
		return mu, sigma


class LinearBlock(nn.Module):
	"""
	Simple linear block layer consisting of linear layer and activation function
	"""
	def __init__(self, input_dim, output_dim, activation, dropout, bias=True):
		"""
		: param input_dim		Number of input features (number of neurons from previos layer).
		: param output_dim		Number of output featuers (number of neurons of output)
		: param activation 		Activation function for this layer (nn.ReLU() <– object type)
		: param dropout         Probability of dropout p. (dropout==False -> no dropout)
		"""
		super(LinearBlock, self).__init__()
		self.layer = nn.Linear(in_features=input_dim, out_features=output_dim, bias=bias)
		self.activation = activation
		self.dropout = nn.Dropout(p=dropout) if dropout!=False else dropout

	def forward(self, X):
		"""
		: param X 			Input data (N, L) = (N samples in batch, Length of input featues per sample). (batch of data)
		"""
		if self.dropout==False:
			return self.activation(self.layer(X))
		else:
			return self.dropout(self.activation(self.layer(X)))


class CBD1dBlock(nn.Module):
	"""
	Block of layers consisting of Conv1d, BatchNorm1d, Activation and Dropout
	"""
	def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation=nn.ReLU(), dropout=False):
		"""
		: param in_channels		Number of input channels (input features).
		: param out_channels	Number of output channels (output features).
		: param activation 		Activation function for this layer (nn.ReLU() <– object type)
		: param dropout         Probability of dropout p. (dropout==False -> no dropout)
		"""
		super(CBD1dBlock, self).__init__()
		self.layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
		self.batch_norm = nn.BatchNorm1d(num_features=out_channels)
		self.activation = activation
		self.dropout = nn.Dropout(p=dropout) if dropout!=False else dropout

	def forward(self, X):
		"""
		: param X 			Input data in format (N, C, L) = (N samples in batch, Channels, Length of numbers in channel). (batch of data)
		"""
		CoBN = self.batch_norm(self.layer(X))
		if self.dropout==False:
			return self.activation(CoBN)
		else:
			return self.dropout(self.activation(CoBN))


class Flatten(nn.Module):
	def __init__(self, out_features):
		super(Flatten, self).__init__()
		self.output_dim = out_features

	def forward(self, x):
		return x.view(-1, self.output_dim)


class Reshape(nn.Module):
	def __init__(self, out_shape):
		super(Reshape, self).__init__()
		self.out_shape = out_shape

	def forward(self, x):
		return x.view(-1, *self.out_shape)

