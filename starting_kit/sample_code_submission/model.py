import numpy as np
import pickle
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from os.path import isfile

### NEW CONTRIBUTION OF GROUP ORBITER ###
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

import matplotlib.pyplot as plt # à supprimer !!!!!
### NEW CONTRIBUTION OF GROUP ORBITER ###

def requires_grad(p):
    return p.requires_grad

### NEW CONTRIBUTION OF GROUP ORBITER ###
def transforme(X, i):
    pca = PCA(n_components=i)
    X = pca.fit_transform(X)
    return X
### NEW CONTRIBUTION OF GROUP ORBITER ###

class baselineModel(BaseEstimator):
    def __init__(self, max_depth=5):
        """
        Using DecisionTreeClassifier from sklearn as Baseline Model
        Has one parameter which is the max depth of the tree (base value of 5)
        """
        super(baselineModel, self).__init__()
        self.classifier = DecisionTreeClassifier(max_depth=max_depth)
        self.num_train_samples=0
        self.num_feat=1
        self.num_labels=1
        self.is_trained=False
        ### NEW CONTRIBUTION OF GROUP ORBITER ###
        #ajout d'attributs
        self.best_prepro=False
        self.value_best_prepro=0
        ### NEW CONTRIBUTION OF GROUP ORBITER ###

    def fit(self, X, y):
        ### NEW CONTRIBUTION OF GROUP ORBITER ###
        if self.best_prepro:
            X = transforme(X, self.value_best_prepro)
        ### NEW CONTRIBUTION OF GROUP ORBITER ###
        
        self.num_train_samples = X.shape[0]
        if X.ndim>1: self.num_feat = X.shape[1]
        print("FIT: dim(X)= [{:d}, {:d}]".format(self.num_train_samples, self.num_feat))
        num_train_samples = y.shape[0]
        if y.ndim>1: self.num_labels = y.shape[1]
        print("FIT: dim(y)= [{:d}, {:d}]".format(num_train_samples, self.num_labels))
        if (self.num_train_samples != num_train_samples):
            print("ARRGH: number of samples in X and y do not match!")
        self.is_trained=True
        self.classifier.fit(X, y)

    def predict(self, X):
        ### NEW CONTRIBUTION OF GROUP ORBITER ###
        if self.best_prepro:
            X = transforme(X, self.value_best_prepro)
        ### NEW CONTRIBUTION OF GROUP ORBITER ###
        
        num_test_samples = X.shape[0]
        if X.ndim>1:
            num_feat = X.shape[1]
        print("PREDICT: dim(X)= [{:d}, {:d}]".format(num_test_samples, num_feat))
        if (self.num_feat != num_feat):
            print("ARRGH: number of features in X does not match training data!")
        print("PREDICT: dim(y)= [{:d}, {:d}]".format(num_test_samples, self.num_labels))
        y = np.zeros([num_test_samples, self.num_labels])
        return self.classifier.predict(X)

    def save(self, path="./"):
        #pickle.dump(self, open(path + '_model.pickle', "wb"))
        pass

    def load(self, path="./"):
        #modelfile = path + '_model.pickle'
        #if isfile(modelfile):
        #    with open(modelfile, 'rb') as f:
        #        self = pickle.load(f)
        #    print("Model reloaded from: " + modelfile)
        return self


### NEW CONTRIBUTION OF GROUP ORBITER ###  
def tests_auto(X_train, Y_train):

    Y_train = Y_train.ravel()
    #definition des paramètres à tester pour le classifier SVC:
    tuned_parameters = {'kernel':('linear', 'rbf'),'C':[1, 10, 100], 'gamma': [1e-3, 1e-4]} 

    print("1° cas : recherche des meilleurs paramètres avec les images de base :")
    grid = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy')

    debut = time.time()
    grid.fit(X_train, Y_train)
    fin = time.time() - debut
    
    max_cv_non_prepro = grid.best_score_
    
    #Affichage du meilleur score avec les meilleurs paramètres (temps)
    print("Les meilleurs paramètres sont : {}, qui donnent un score de : {} (en {} secondes)".format(grid.best_params_, round(grid.best_score_, 3), round(fin)))
    
    #Graphe scores de cv avec le prepro de base et détermination meilleurs paramètres pour l'apprentissage
    grid_mean_scores = [result for result in grid.cv_results_['mean_test_score']]
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(121)
    plt.plot(range(0, len(grid_mean_scores)), grid_mean_scores)
    plt.xlabel('N° de test des paramètres')
    plt.ylabel('Cross-Validation Accuracy')

    plt.title('Cross Validation sans notre preprocessing')
    plt.show

    
    print("\n2° cas : recherche des améliorations possibles de notre preprocessing (PCA):")
    result = []

    for i in range(128, 1024, 128):
        X = transforme(X_train, i)
        g = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='accuracy')
        debut = time.time()
        g.fit(X, Y_train)
        fin = time.time() - debut
        res = [i, fin, g.best_score_, g.best_params_, g.best_estimator_]
        print("Pour n_components = {}, on obtient un score de {}.".format(i, round(res[2], 3)))
        result.append(res)
        
    #Affichage des différentes valeurs pour connaitre la meilleur pour le prepro
    valeurs_prepro_GSCV = []
    for i in range(0, len(result)):
        valeurs_prepro_GSCV.append(result[i][2])
    plt.subplot(122)
    plt.plot(range(128, 1024, 128), valeurs_prepro_GSCV)
    plt.xlabel('Valeur de n_components')
    #plt.ylabel('Cross-Validation Accuracy')
    plt.title('Cross Validation avec notre preprocessing')
    plt.show
    
    #déterminer la cv max dans result
    max_cv_prepro = result[0]
    for i in range(0, len(result)):
        if result[i][2] > max_cv_prepro[2]:
            max_cv_prepro = result[i]
    
    #Petite phrase qui affiche les meilleurs paramètres
    print("Le meilleur paramètre pour n_components : {}, qui donnent un score de : {}".format(max_cv_prepro[0], round(max_cv_prepro[2],3)))
    
    #Modifications a apportées en conséquences
    retour = baselineModel()
    if max_cv_prepro[2] > max_cv_non_prepro:
        #On chooisit de faire notre prepro puis l'apprentissage avec les paramètres qui ont le mieux réussit
        print("\nLe model choisit est celui avec notre preprocessing !")
        retour.classifier = max_cv_prepro[4]
        retour.best_prepro = True
        retour.value_best_prepro = max_cv_prepro[0]
    else:
        print("\nLe model choisit est celui sans notre preprocessing !")
        retour.classifier = grid.best_estimator_
        
    return retour
    
    
