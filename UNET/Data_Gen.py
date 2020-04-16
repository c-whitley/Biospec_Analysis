import numpy as np

from keras import utils
from keras.preprocessing.image import ImageDataGenerator


class CustomDataGen(utils.Sequence):
	"""
	A custom image generator to be used in loading hyperspectral image files of different formats.
	Most functions will be inherited from the standard keras image generator class
	"""

	def __init__(self, list_IDs, labels, batch_size = 32, dim = (32,32,32), n_channels = 1,
		n_classes = 1, shuffle = True, seed = np.random.random()):

		self.dim = dim
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.n_channels = n_channels
		self.n_classes = n_classes
		self.shuffle = shuffle
		self.seed = seed

		# Allow use of a custom function to be supplied to load and preprocess the data
		self.function = self.__on_epoch_end

		if self.shuffle:
			self.on_epoch_end()


	def __on_epoch_end(self):
		"""
		Updates indexes after each epoch
		"""

		self.indexes = np.arange(len(self.list_IDs))

		if self.shuffle == True:
			np.random.shuffle(self.indexes)


	def __data_generation(self, list_IDs_temp):
	  """
	  Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)
	  """

	  # Initialization
	  X = np.empty((self.batch_size, *self.dim, self.n_channels))
	  y = np.empty((self.batch_size), dtype=int)

	  # Generate data
	  for i, ID in enumerate(list_IDs_temp):
		  # Store sample
		  X[i,] = np.load('data/' + ID + '.npy')

		  # Store class
		  y[i] = self.labels[ID]

	  return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


	def __len__(self):
		"""
		Get the number of batches per epoch
		"""

		return int(np.floor(len(self.list_IDs))/self.batch_size)


	def __getitem__(self, index):
	  """
	  # Generate indexes of the batch
	  """

	  indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

	  # Find list of IDs
	  list_IDs_temp = [self.list_IDs[k] for k in indexes]

	  # Generate data
	  X, y = self.__data_generation(list_IDs_temp)

	  return X, y


class DataGenCustom(ImageDataGenerator):
	"""
	Custom version of the Image Generator from Keras
	"""

	def test(self):

		print(help(self))

	def envi_flow(self):

		pass

