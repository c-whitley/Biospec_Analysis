from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_reduced(im, n_components = 10):
    
    shape = im.shape
    tab = im.reshape(-1, shape[-1])
    
    pipe = Pipeline([
        #("Scaler", StandardScaler()),
        ("PCA", PCA(n_components = n_components))
    ])
    
    tab = pipe.fit_transform(tab)
    
    return tab.reshape(shape[0], shape[1], -1)