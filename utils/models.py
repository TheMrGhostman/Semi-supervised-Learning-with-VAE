import numpy as np 
import time

import torch 
import torch.nn as nn
import torch.nn.functional as F 

from .layers import *

from IPython.core.debugger import set_trace


def paremeters_summary(model):
    s = []
    for p in model.parameters():
        dims = p.size()
        n = np.prod(p.size())
        s.append((dims, n))
    return s, np.sum([j for i,j in s])

# Models
# DeepDenseVAE mark I
class DeepDenseVAE_mark_I(nn.Module):
	def __init__(self, original_dim=28*28, latent_dim=20, encoder_dims=[400,200]):
		super(DeepDenseVAE_mark_I, self).__init__()
		self.original_dim = original_dim
		self.encoder = nn.Sequential(
							Flatten(out_features=original_dim),
							nn.Linear(in_features=original_dim, out_features=encoder_dims[0]),
							nn.ReLU(),
							nn.Linear(in_features=encoder_dims[0], out_features=encoder_dims[1]),
							nn.ReLU(),
							VariationalLayer(in_features=encoder_dims[1], out_features=latent_dim, return_KL=False)
							)
		self.decoder =  nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=encoder_dims[1]),
							nn.ReLU(),
							nn.Linear(in_features=encoder_dims[1], out_features=encoder_dims[0]),
							nn.ReLU(),
							nn.Linear(in_features=encoder_dims[0], out_features=original_dim)
							#VariationalDecoderOutput(in_features=encoder_dims[0], out_features=original_dim)
							)

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class DeepDenseVAE_mark_II(nn.Module):
	def __init__(self, original_dim=28*28, latent_dim=20, encoder_dims=[400,200]):
		super(DeepDenseVAE_mark_II, self).__init__()
		self.original_dim = original_dim
		self.encoder = nn.Sequential(
							Flatten(out_features=original_dim),
							nn.Linear(in_features=original_dim, out_features=encoder_dims[0]),
							nn.ReLU(),
							nn.Linear(in_features=encoder_dims[0], out_features=encoder_dims[1]),
							nn.ReLU(),
							VariationalLayer(in_features=encoder_dims[1], out_features=latent_dim, return_KL=False)
							)
		self.decoder =  nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=encoder_dims[1]),
							nn.ReLU(),
							nn.Linear(in_features=encoder_dims[1], out_features=encoder_dims[0]),
							nn.ReLU(),
							VariationalDecoderOutput(in_features=encoder_dims[0], out_features=original_dim)
							)

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class DeepDenseVAE_mark_III(nn.Module):
	# best so far
	def __init__(self, original_dim=28*28, latent_dim=20, encoder_dims=[400,200]):
		super(DeepDenseVAE_mark_III, self).__init__()
		self.original_dim = original_dim
		self.encoder = nn.Sequential(
							Flatten(out_features=original_dim),
							nn.Linear(in_features=original_dim, out_features=encoder_dims[0]),
							nn.LeakyReLU(),
							nn.Linear(in_features=encoder_dims[0], out_features=encoder_dims[1]),
							nn.LeakyReLU(),
							VariationalLayer(in_features=encoder_dims[1], out_features=latent_dim, return_KL=False)
							)
		self.decoder =  nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=encoder_dims[1]),
							nn.LeakyReLU(),
							nn.Linear(in_features=encoder_dims[1], out_features=encoder_dims[0]),
							nn.LeakyReLU(),
							VariationalDecoderOutput(in_features=encoder_dims[0], out_features=original_dim)
							)


class DeepDenseVAE_mark_IV(nn.Module):
	def __init__(self, original_dim=28*28, latent_dim=20, encoder_dims=[400,200,100]):
		super(DeepDenseVAE_mark_IV, self).__init__()
		self.original_dim = original_dim
		self.encoder = nn.Sequential(
							Flatten(out_features=original_dim),
							nn.Linear(in_features=original_dim, out_features=encoder_dims[0]),
							nn.LeakyReLU(),
							nn.Linear(in_features=encoder_dims[0], out_features=encoder_dims[1]),
							nn.LeakyReLU(),
							nn.Linear(in_features=encoder_dims[1], out_features=encoder_dims[2]),
							nn.LeakyReLU(),
							VariationalLayer(in_features=encoder_dims[2], out_features=latent_dim, return_KL=False)
							)
		self.decoder =  nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=encoder_dims[2]),
							nn.LeakyReLU(),
							nn.Linear(in_features=encoder_dims[2], out_features=encoder_dims[1]),
							nn.LeakyReLU(),
							nn.Linear(in_features=encoder_dims[1], out_features=encoder_dims[0]),
							nn.LeakyReLU(),
							VariationalDecoderOutput(in_features=encoder_dims[0], out_features=original_dim)
							)

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class DeepDenseVAE_mark_V(nn.Module):
	def __init__(self, original_dim=28*28, latent_dim=20, encoder_dims=[400,200]):
		super(DeepDenseVAE_mark_V, self).__init__()
		self.original_dim = original_dim
		self.encoder = nn.Sequential(
							nn.Linear(in_features=original_dim, out_features=encoder_dims[0]),
							nn.ELU(),
							nn.Linear(in_features=encoder_dims[0], out_features=encoder_dims[1]),
							nn.ELU(),
							VariationalLayer(in_features=encoder_dims[1], out_features=latent_dim, return_KL=False)
							)
		self.decoder =  nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=encoder_dims[1]),
							nn.ELU(),
							nn.Linear(in_features=encoder_dims[1], out_features=encoder_dims[0]),
							nn.ELU(),
							VariationalDecoderOutput(in_features=encoder_dims[0], out_features=original_dim)
							)

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in) 
		return self.decoder(z), mu, sigma


class DeepConvVAE(nn.Module):
	def __init__(self, original_dim, latent_dim):
		super(DeepConvVAE, self).__init__()
		self.original_dim = original_dim
		self.encoder = nn.Sequential(
							Reshape(out_shape=(1,160)),
							nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2, padding=0),
							nn.ReLU(),
							nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0),
							nn.ReLU(),
							nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
							nn.ReLU(),
							Flatten(out_features=64*18),
							VariationalLayer(in_features=64*18, out_features=latent_dim, return_KL=False)
							)
		self.decoder = nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=64*18),
							nn.ReLU(),
							Reshape(out_shape=(64,18)),
							nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0),
							nn.ReLU(),
							nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0),
							nn.ReLU(),
							ConvTransposeDecoderOutput(
								in_channels=16, 
								in_features=16*78, 
								out_features=self.original_dim, 
								kernel_size=6, 
								stride=2
								)
							#Flatten(out_features=self.original_dim),
							#VariationalOutputLayer(in_features=, out_features=latent_dim, return_KL=False)
							)

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class DeepConvVAE_ReLU(nn.Module):
	def __init__(self, original_dim, latent_dim):
		super(DeepConvVAE_ReLU, self).__init__()
		self.original_dim = original_dim
		self.encoder = nn.Sequential(
							Reshape(out_shape=(1,160)),
							nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2, padding=0),
							nn.ReLU(),
							nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0),
							nn.ReLU(),
							nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
							nn.ReLU(),
							Flatten(out_features=64*18),
							VariationalLayer(in_features=64*18, out_features=latent_dim, return_KL=False)
							)
		self.decoder = nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=64*18),
							nn.ReLU(),
							Reshape(out_shape=(64,18)),
							nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0),
							nn.ReLU(),
							nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0),
							nn.ReLU(),
							ConvTransposeDecoderOutput(
								in_channels=16, 
								in_features=16*78, 
								out_features=self.original_dim, 
								kernel_size=6, 
								stride=2
								)
							#Flatten(out_features=self.original_dim),
							#VariationalOutputLayer(in_features=, out_features=latent_dim, return_KL=False)
							)

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class DeepConvVAE_ELU(nn.Module):
	def __init__(self, original_dim, latent_dim):
		super(DeepConvVAE_ELU, self).__init__()
		self.original_dim = original_dim
		self.encoder = nn.Sequential(
							Reshape(out_shape=(1,160)),
							nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2, padding=0),
							nn.ELU(),
							nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0),
							nn.ELU(),
							nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=0),
							nn.ELU(),
							Flatten(out_features=64*18),
							VariationalLayer(in_features=64*18, out_features=latent_dim, return_KL=False)
							)
		self.decoder = nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=64*18),
							nn.ELU(),
							Reshape(out_shape=(64,18)),
							nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0),
							nn.ELU(),
							nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0),
							nn.ELU(),
							ConvTransposeDecoderOutput(
								in_channels=16, 
								in_features=16*78, 
								out_features=self.original_dim, 
								kernel_size=6, 
								stride=2
								)
							#Flatten(out_features=self.original_dim),
							#VariationalOutputLayer(in_features=, out_features=latent_dim, return_KL=False)
							)

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class CBN_VAE(nn.Module):
	def __init__(self, original_dim, latent_dim, activation=nn.ReLU(inplace=True)):
		super(CBN_VAE, self).__init__()
		self.encoder = nn.Sequential(
							Reshape(out_shape=(1, original_dim)),
							nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2, padding=0, bias=False), # n*1*160 -> n*16*78
							activation,
							nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=False),#n*16*78->n*32*38
							nn.BatchNorm1d(num_features=32),
							activation,
							nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False),#n*32*38->n*64*18
							nn.BatchNorm1d(num_features=64),
							activation,
							Flatten(out_features=64*18),
							nn.Linear(in_features=64*18, out_features=256),
							activation,
							VariationalLayer(in_features=256, out_features=latent_dim, return_KL=False)
							)
		self.decoder = nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=256),
							activation,
							nn.Linear(in_features=256, out_features=64*18),
							Reshape(out_shape=(64,18)),
							nn.BatchNorm1d(num_features=64),
							activation,
							nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0),
							nn.BatchNorm1d(num_features=32),
							activation,
							nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0),
							#nn.BatchNorm1d(num_features=16),
							activation,
							ConvTransposeDecoderOutput(
								in_channels=16, 
								in_features=16*78, 
								out_features=original_dim, 
								kernel_size=6, 
								stride=2
								)
							)

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class CBND_VAE(nn.Module):
	def __init__(self, original_dim, latent_dim, dropout,activation=nn.ReLU(inplace=True)):
		super(CBN_VAE, self).__init__()
		self.encoder = nn.Sequential(
							Reshape(out_shape=(1, original_dim)),
							nn.Conv1d(in_channels=1, out_channels=16, kernel_size=6, stride=2, padding=0, bias=False), # n*1*160 -> n*16*78
							activation,
							nn.Conv1d(in_channels=16, out_channels=32, kernel_size=4, stride=2, bias=False),#n*16*78->n*32*38
							nn.BatchNorm1d(num_features=32),
							activation,
							nn.Dropout(p=dropout),
							nn.Conv1d(in_channels=32, out_channels=64, kernel_size=4, stride=2, bias=False),#n*32*38->n*64*18
							nn.BatchNorm1d(num_features=64),
							activation,
							nn.Dropout(p=dropout),
							Flatten(out_features=64*18),
							nn.Linear(in_features=64*18, out_features=256),
							activation,
							nn.Dropout(p=dropout),
							VariationalLayer(in_features=256, out_features=latent_dim, return_KL=False)
							)
		self.decoder = nn.Sequential(
							nn.Linear(in_features=latent_dim, out_features=256),
							activation,
							nn.Dropout(p=dropout),
							nn.Linear(in_features=256, out_features=64*18),
							Reshape(out_shape=(64,18)),
							nn.BatchNorm1d(num_features=64),
							activation,
							nn.Dropout(p=dropout),
							nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0),
							nn.BatchNorm1d(num_features=32),
							activation,
							nn.Dropout(p=dropout),
							nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=0),
							#nn.BatchNorm1d(num_features=16),
							activation,
							ConvTransposeDecoderOutput(
								in_channels=16, 
								in_features=16*78, 
								out_features=original_dim, 
								kernel_size=6, 
								stride=2
								)
							)

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class DeepLSTM_VAE(nn.Module):
	def __init__(self, sequence_len, n_features, latent_dim, hidden_size=128, num_layers=2, batch_size=100, use_cuda=True):
		# ověřit predikci pro jiný batch size !!!!!!!!!!!!!!!!
		super(DeepLSTM_VAE, self).__init__()

		self.sequence_len = sequence_len
		self.n_features = n_features
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_cuda = use_cuda

		if self.use_cuda and torch.cuda.is_available():
			self.dtype = torch.cuda.FloatTensor
		else:
			self.dtype = torch.float32

		self.encoder_reshape = Reshape(out_shape=(self.sequence_len, self.n_features))
		self.encoder_lstm = nn.LSTM(
									input_size=n_features,
									hidden_size=hidden_size,
									num_layers=num_layers,
									batch_first=False,
									bidirectional=False
									) 
		self.encoder_output = VariationalLayer(
									in_features=hidden_size, 
									out_features=latent_dim, 
									return_KL=False
									)

		self.decoder_hidden = nn.Linear(
									in_features=latent_dim,
									out_features=hidden_size,
									bias=True	
									)
		self.decoder_lstm = nn.LSTM(
									input_size=1,
									hidden_size=hidden_size,
									num_layers=num_layers,
									batch_first=False,
									bidirectional=False
									) 
		self.decoder_output = RecurrentDecoderOutput(
									in_features=hidden_size,
									sequence_len=sequence_len,
									out_features=n_features,
									bias=True
									)
		self.decoder_input = torch.zeros(
									self.sequence_len, 
									self.batch_size, 
									self.n_features, 
									requires_grad=True
									).type(self.dtype)
		self.decoder_c_0 = torch.zeros(
									self.num_layers,
									self.batch_size,
									self.hidden_size,
									requires_grad=True 
									).type(self.dtype)


	def encoder(self, x_in):
		x = self.encoder_reshape(x_in)
		#set_trace()
		x = x.permute(1, 0, 2)
		_,(h_end, c_end) = self.encoder_lstm(x)
		h_end = h_end[-1, :, :] # shape(batch_size, num_features)
		return self.encoder_output(h_end)

	def decoder(self, z_in):
		h_state = self.decoder_hidden(z_in)
		#set_trace()
		h_0 = torch.stack([h_state for _ in range(self.num_layers)])
		lstm_output, _ = self.decoder_lstm(self.decoder_input, (h_0, self.decoder_c_0))
		mu, sigma = self.decoder_output(lstm_output)
		return mu, sigma

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class DeepLSTM_VAE_MSE(nn.Module):
	def __init__(self, sequence_len, n_features, latent_dim, hidden_size=128, num_layers=2, batch_size=100, use_cuda=True):
		# ověřit predikci pro jiný batch size !!!!!!!!!!!!!!!!
		super(DeepLSTM_VAE_MSE, self).__init__()

		self.sequence_len = sequence_len
		self.n_features = n_features
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_cuda = use_cuda

		if self.use_cuda and torch.cuda.is_available():
			self.dtype = torch.cuda.FloatTensor
		else:
			self.dtype = torch.float32

		self.encoder_reshape = Reshape(out_shape=(self.sequence_len, self.n_features))
		self.encoder_lstm = nn.LSTM(
									input_size=n_features,
									hidden_size=hidden_size,
									num_layers=num_layers,
									batch_first=False,
									bidirectional=False
									) 
		self.encoder_output = VariationalLayer(
									in_features=hidden_size, 
									out_features=latent_dim, 
									return_KL=False
									)

		self.decoder_hidden = nn.Linear(
									in_features=latent_dim,
									out_features=hidden_size,
									bias=True	
									)
		self.decoder_lstm = nn.LSTM(
									input_size=1,
									hidden_size=hidden_size,
									num_layers=num_layers,
									batch_first=False,
									bidirectional=False
									) 
		self.decoder_output = nn.Linear(
									in_features=hidden_size,
									out_features=n_features,
									bias=True
									)
		self.decoder_input = torch.zeros(
									self.sequence_len, 
									self.batch_size, 
									self.n_features, 
									requires_grad=True
									).type(self.dtype)
		self.decoder_c_0 = torch.zeros(
									self.num_layers,
									self.batch_size,
									self.hidden_size,
									requires_grad=True 
									).type(self.dtype)


	def encoder(self, x_in):
		x = self.encoder_reshape(x_in)
		#set_trace()
		x = x.permute(1, 0, 2)
		_,(h_end, c_end) = self.encoder_lstm(x)
		h_end = h_end[-1, :, :] # shape(batch_size, num_features)
		return self.encoder_output(h_end)

	def decoder(self, z_in):
		h_state = self.decoder_hidden(z_in)
		#set_trace()
		h_0 = torch.stack([h_state for _ in range(self.num_layers)])
		lstm_output, _ = self.decoder_lstm(self.decoder_input, (h_0, self.decoder_c_0))

		mu = self.decoder_output(lstm_output)
		return mu[:,:,-1].T

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


class DeepGRU_VAE(nn.Module):
	def __init__(self, sequence_len, n_features, latent_dim, hidden_size=128, num_layers=2, batch_size=100, use_cuda=True):
		# ověřit predikci pro jiný batch size !!!!!!!!!!!!!!!!
		super(DeepLSTM_VAE, self).__init__()

		self.sequence_len = sequence_len
		self.n_features = n_features
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.use_cuda = use_cuda

		if self.use_cuda and torch.cuda.is_available():
			self.dtype = torch.cuda.FloatTensor
		else:
			self.dtype = torch.float32

		self.encoder_reshape = Reshape(out_shape=(self.sequence_len, self.n_features))
		self.encoder_lstm = nn.GRU(
									input_size=n_features,
									hidden_size=hidden_size,
									num_layers=num_layers,
									batch_first=False,
									bidirectional=False
									) 
		self.encoder_output = VariationalLayer(
									in_features=hidden_size, 
									out_features=latent_dim, 
									return_KL=False
									)

		self.decoder_hidden = nn.Linear(
									in_features=latent_dim,
									out_features=hidden_size,
									bias=True	
									)
		self.decoder_lstm = nn.GRU(
									input_size=1,
									hidden_size=hidden_size,
									num_layers=num_layers,
									batch_first=False,
									bidirectional=False
									) 
		self.decoder_output = RecurrentDecoderOutput(
									in_features=hidden_size,
									sequence_len=sequence_len,
									out_features=n_features,
									bias=True
									)
		self.decoder_input = torch.zeros(
									self.sequence_len, 
									self.batch_size, 
									self.n_features, 
									requires_grad=True
									).type(self.dtype)
		self.decoder_c_0 = torch.zeros(
									self.num_layers,
									self.batch_size,
									self.hidden_size,
									requires_grad=True 
									).type(self.dtype)


	def encoder(self, x_in):
		x = self.encoder_reshape(x_in)
		#set_trace()
		x = x.permute(1, 0, 2)
		_,(h_end, c_end) = self.encoder_lstm(x)
		h_end = h_end[-1, :, :] # shape(batch_size, num_features)
		return self.encoder_output(h_end)

	def decoder(self, z_in):
		h_state = self.decoder_hidden(z_in)
		#set_trace()
		h_0 = torch.stack([h_state for _ in range(self.num_layers)])
		lstm_output, _ = self.decoder_lstm(self.decoder_input, (h_0, self.decoder_c_0))
		mu, sigma = self.decoder_output(lstm_output)
		return mu, sigma

	def forward(self, x_in):
		z, mu, sigma = self.encoder(x_in)
		return self.decoder(z), mu, sigma


# Classification models
class DNN(nn.Module):
	def __init__(self, n_layers=2, neurons=[160, 400, 200], output_dim=10 ,activation=nn.Sigmoid(), dropout=False):
		super(DNN, self).__init__()
		self.n_layers = n_layers
		self.neurons = neurons
		self.check()
		order_dict = []
		for layer in range(n_layers):
			order_dict.append(
				(f"layer {layer+1}", 
				 LinearBlock(input_dim=self.neurons[layer], 
							 output_dim=self.neurons[layer+1], 
							 activation=activation, 
							 dropout=dropout)
				))
		order_dict.append(("layer final", nn.Linear(in_features=neurons[-1], out_features=output_dim)))
		self.model = nn.Sequential(OrderedDict(order_dict))

	def check(self):
		if len(self.neurons)!=self.n_layers+1:
			raise ValueError("Number of layser is NOT equal to length of list of neurons")
	
	def forward(self, x_in):
		return self.model(x_in)





