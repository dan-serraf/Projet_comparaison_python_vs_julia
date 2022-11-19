# -*- coding: utf-8 -*-
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from tqdm import tqdm
import time

import sys
sys.path.append('../../../Biblioteque/BibliotequePython/UnSupervised')

import ClassifierUnSupervised as  clu


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

    
# Creation modele

temps_time = []
temps_acc = []

for train_x, test_x, train_y, test_y in tqdm(list_data) :
    debut = time.time()
    kmeans =  clu.ClassifierKmens(k=10,learning_rate=0.005,iter_max=100)
    kmeans = kmeans.fit(train_x)
    a = np.array(sorted(Counter(train_y).values()))
    lst = []
    for i in kmeans.cluster :
        lst.append(len(kmeans.cluster[i]))
    while len(a) !=  len(lst) :
        lst.append(0)
    b = np.array(sorted(lst))
   
    acc = (1 - np.abs(a-b).sum()/sum(a))
    fin = time.time()
    temps_time.append(fin - debut)
    temps_acc.append(acc)
  



chemin = '../../../Image/ImagePython/Kmeans/'
nom = 'kmeans_python'
plt.figure()
plt.scatter(range(len(temps_time)),temps_time)
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