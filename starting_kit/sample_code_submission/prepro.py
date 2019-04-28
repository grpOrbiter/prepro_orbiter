from sklearn.decomposition import PCA

### START NEW CONTRIBUTION OF GROUP ORBITER ###
def transforme(X, i):
    pca = PCA(n_components=i)
    X = pca.fit_transform(X)
    return X
### END NEW CONTRIBUTION OF GROUP ORBITER ###
