import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_reduced(im, n_components = 10):
	
	shape = im.shape
	tab = im.reshape(-1, shape[-1])
	
	pipe = Pipeline([
		("Scaler", StandardScaler()),
		("PCA", PCA(n_components = n_components))
	])
	
	tab = pipe.fit_transform(tab)
	
	return tab.reshape(shape[0], shape[1], -1)


def image_import(file, transformer, start = 1000, end = 1800, paraffin = (1340, 1490)):

	image = file.load()
	wn = np.array(image.metadata["wn"], dtype = float)
	
	starti, endi, psi, pei = [np.argmin(np.abs(i - wn)) for i in [start, end, paraffin[0], paraffin[1]]]
	
	
	tab = pd.DataFrame(image.reshape(-1, image.shape[-1]), columns = wn)
	
	# Only use wavenumbers within specified range
	tab = tab.iloc[:,starti:endi]

	if not paraffin == False:
		# Drop paraffin
		tab = tab.drop(wn[psi:pei], axis = 1)



	tab = transformer.fit_transform(tab.T).T


	return tab.reshape(image.shape[:-1]+(-1,))