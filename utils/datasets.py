import numpy as np 
import torch


class SupervisedDataset(torch.utils.data.Dataset):
	def __init__(self, dataset, labels, transform=None):
		self.X = torch.tensor(dataset.astype('float32')) if isinstance(dataset, np.ndarray) else dataset.float()
		self.y = torch.tensor(labels).long() if isinstance(labels, np.ndarray) else labels.float()
		self.transform = transform

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample_X = self.X[idx] 
		sample_y = self.y[idx]

		if self.transform:
			sample_X = self.transform(sample_X)

		return sample_X, sample_y


class H_alphaSequences(torch.utils.data.Dataset):
	def __init__(self, dataset, labels, transform=None):
		self.X = torch.tensor(dataset.astype('float32'))
		self.y = torch.tensor(labels).long()
		self.transform = transform

	def __len__(self):
		return self.X.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		sample_X = self.X[idx] 
		sample_y = self.y[idx]

		if self.transform:
			sample_X = self.transform(sample_X)

		return sample_X, sample_y


def H_alphaSequencesSplit():
	pass


class BinaryMNIST(torch.utils.data.Dataset):
	"""Binary MNIST dataset for autoencoders"""

	def __init__(self, original_mnist, labels, threshold=0.5, transform=None):
		"""
		Args:
			original_mnist (np.array) : original MNIST dataset in array form (from tensorflow.datsets)
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		self.threshold = threshold
		self.dataset = self.preprocess(original_mnist.astype('float32'))
		self.transform = transform
		self.labels = torch.from_numpy(labels.astype('float32'))

	def preprocess(self, dataset):
		dataset = dataset / 255
		dataset[dataset >= self.threshold] = 1
		dataset[dataset < self.threshold] = 0
		return dataset

	def __len__(self):
		return self.dataset.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		sample = self.dataset[idx]

		if self.transform:
			sample = self.transform(sample)

		return sample, self.labels[idx]


class SemiSupervisedBinaryMNIST(torch.utils.data.Dataset):
	"""Binary MNIST dataset for Semi-supervised Learning with Deep Generative Models"""

	def __init__(self, original_mnist, labels, number_of_labels, threshold=0.5, transform=None):
		"""
		Args:
			original_mnist (np.array) : original MNIST dataset in array form (from tensorflow.datsets)
			transform (callable, optional): Optional transform to be applied
				on a sample.
		"""
		if number_of_labels%10!=0:
			raise ValueError("Number of labels per class must be same. Enter number which is devisible by 10 (number of classes.")
		self.number_of_labels = number_of_labels
		self.threshold = threshold
		self.dataset = self.preprocess(original_mnist.astype('float32'))
		if self.number_of_labels == "all":
			self.labels = labels
		else:
			self.labels = self.drop_labels(labels)
		self.transform = transform

	def preprocess(self, dataset):
		dataset = dataset / 255
		dataset[dataset >= self.threshold] = 1
		dataset[dataset < self.threshold] = 0
		return dataset

	def drop_labels(self, labels): 
		# add: uniform choice from all classes
		each_class = self.number_of_labels/10
		classes = {}
		for cl in range(10):
			for i,j in enumerate(self.dataset):
				if labels[i]==cl:
					classes[cl].append(j) 
		# ještě dodělat 

		rand_index = np.random.choice(np.arange(len(self.dataset)), size=self.number_of_labels) # indexes in which labels are known
		new_labels = np.full_like(labels, np.nan, dtype=np.double)
		new_labels[rand_index] = labels[rand_index]
		return torch.from_numpy(new_labels)

	def __len__(self):
		return self.dataset.shape[0]

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		sample_x = self.dataset[idx]
		sample_y = self.labels[idx]

		if self.transform:
			sample_x = self.transform(sample_x)

		return sample_x, sample_y 


def BinaryMNIST_split(dtype="supervised", number_of_labels=None, threshold=0.5, transform=None):
	#import tensorflow 
	#(train_features, train_labels), (test_features, test_labels) = tensorflow.keras.datasets.mnist.load_data()	
	import sklearn.datasets as datasets
	X, y = datasets.fetch_openml('mnist_784', version=1, return_X_y=True)
	train_features, train_labels = X[:55000].reshape(-1,28,28), y[:55000]
	test_features, test_labels = X[55000:].reshape(-1,28,28), y[55000:]

	if dtype=="supervised":
		return BinaryMNIST(train_features, labels=train_labels, threshold=threshold, transform=transform), \
			BinaryMNIST(test_features, labels=test_labels, threshold=threshold, transform=transform) 

	elif dtype=="semisupervised":
		return (SemiSupervisedBinaryMNIST(train_features, train_labels, number_of_labels=number_of_labels, threshold=threshold, transform=transform),\
			SemiSupervisedBinaryMNIST(test_features, test_labels, number_of_labels=number_of_labels, threshold=threshold, transform=transform))

	else:
		raise ValueError("dtype is not standard!")










