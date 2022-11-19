# -*- coding: utf-8 -*-

# Import de packages externes

import sys
sys.path.append('../../../Biblioteque/BibliotequePython/Supervised')

import ClassifierSupervised as  cls

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 

k_voisin = 5
nb_iter = range(3)
nom_fichier = '../../../Data/mnist_split/'

# Création jeu de données
list_data = []
for iter in tqdm(nb_iter) :

    train_x = pd.read_csv(nom_fichier + "train_x" + str(iter) + ".csv").to_numpy().astype(float)[:100,:]
    test_x = pd.read_csv(nom_fichier + "test_x" + str(iter) + ".csv").to_numpy().astype(float)[:100,:]
    train_y = pd.read_csv( nom_fichier + "train_y" + str(iter) + ".csv").to_numpy()[:,0].astype(int)[:100]
    test_y = pd.read_csv(nom_fichier + "test_y" + str(iter) + ".csv").to_numpy()[:,0].astype(int)[:100]
    list_data.append((train_x, test_x, train_y, test_y))

temps_time = []
temps_acc = []
for train_x, test_x, train_y, test_y in tqdm(list_data) :
    
    debut = time.time()
    knn =  cls.ClassifierKnnVectoriser(k=k_voisin)
    knn.fit(train_x,train_y)
    acc = knn.accuracy(test_x, test_y)
    fin = time.time()
    temps_time.append(fin - debut)
    temps_acc.append(acc)
  


chemin = '../../../Image/ImagePython/KPlusProcheVoisin/'
nom = 'knn_vectoriser_python'
plt.figure()
plt.scatter(range(len(temps_time)),temps_time)
# plt.show()
plt.xlabel('Nombre itération validation croisée sklearn')
plt.ylabel("Temps d'exécution en seconde")
plt.title("Temps exécution knn k=5 python ")
plt.savefig(chemin+'time'+nom+'.png')

plt.figure()
plt.scatter(range(len(temps_acc)),temps_acc)
plt.savefig(chemin+'acc'+nom+'.png')

f = open(chemin + nom +'.txt','w+')
f.write('temps_acc mean -> ' + str(np.array(temps_acc).mean())+'\n')
f.write('temps_acc std -> ' + str(np.array(temps_acc).std())+'\n')
f.write('temps_time mean -> ' + str(np.array(temps_time).mean())+'\n')
f.write('temps_time std -> ' + str(np.array(temps_time).std())+'\n')
f.close()