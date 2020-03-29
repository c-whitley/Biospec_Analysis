import tensorflow as tf

import tensorflow.keras.layers
from tensorflow.keras.models import Model


class UNet:

	def __init__(self, n_classes, input_size):

		self.n_classes = kwargs.get(n_classes)
		self.input_size = input_size

	def make_unet(self):

		input_ = layers.Input(self.input_size)

		c1 = layers.Conv2D(32, (3,3), kernel_initializer = "he_normal", padding = "same", batchnorm = True)(input_)
		p1 = layers.MaxPooling2D((2, 2))(c1)
		p1 = layers.Dropout(0.1)(p1)

		outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c1)


		self.model = layers.Model(inputs = [input_], outputs = [outputs])
		self.model.compile(optimizer = "Adam", loss = "categorical_crossentropy", metrics = ["accuracy"])


print("Hello")
