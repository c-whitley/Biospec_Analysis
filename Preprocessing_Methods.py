import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA

from scipy.spatial import ConvexHull

import multiprocessing as mp


class PCA_denoiser( BaseEstimator, TransformerMixin ):

    #Class Constructor 
    def __init__( self, n_components):

        self.n_components = n_components 

    
    #Return self nothing else to do here    
    def transform( self, X):

        pca_values = self.denoiser.fit(X)
        
        filtered = np.dot(self.denoiser.transform(X)[:,:self.n_components], self.denoiser.components_[:self.n_components,:])

        values = np.add(filtered, np.mean(X, axis = 0).values.reshape(1,-1))

        return values

    
    #Method that describes what we need this transformer to do
    def fit( self, X, y = None ):

        self.denoiser = PCA(self.n_components)

        return self

def rubberband_baseline(spectrum, wn):

    points = np.column_stack([wn, spectrum])

    verts = ConvexHull(points).vertices

    # Rotate convex hull vertices until they start from the lowest one
    verts = np.roll(verts, -verts.argmin())
    # Leave only the ascending part
    verts = verts[:verts.argmax()]

    baseline = np.interp(wn, wn[verts], spectrum[verts])

    return baseline


class Rubber_Band( BaseEstimator, TransformerMixin ):
    """
    Applies a rubber band correction to the input matrix of spectra.
    Must be supplied as a shape (n_samples, n_wavenumbers)
    """

    def __init__(self, wn, n_jobs = 4):

        self.wn = wn
        self.n_jobs = n_jobs
   

    def transform(self, x):

        return self.y - self.baseline


    def fit(self, y):

        if isinstance(y, pd.DataFrame):

            self.y = y.values

        self.y = y

        pool = mp.Pool(processes=self.n_jobs)

        self.baseline = np.array([pool.apply(rubberband_baseline, args=(spectrum, self.wn)) 
        for spectrum in np.apply_along_axis(lambda row: row, axis = 0, arr=self.y)])

        return self