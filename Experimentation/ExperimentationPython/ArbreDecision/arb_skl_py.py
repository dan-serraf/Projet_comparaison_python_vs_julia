# -*- coding: utf-8 -*-

# Import de packages externes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm 
from sklearn.tree import DecisionTreeClassifier

profondeur = 50
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
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(train_x,train_y)
    y2 = clf.predict(test_x)
    acc = (y2 == test_y).mean()
    fin = time.time()
    temps_time.append(fin - debut)
    temps_acc.append(acc)
  

chemin = '../../../Image/ImagePython/ArbreDecision/'
nom = 'arb_skl_python'
plt.figure()
plt.scatter(range(len(temps_time)),temps_time)
plt.xlabel('Nombre itération validation croisée sklearn')
plt.ylabel("Temps d'exécution en seconde")
plt.title("Temps exécution arbre decision profondeur=50 python ")
plt.savefig(chemin+'time'+nom+'.png')

plt.figure()
plt.scatter(range(len(temps_acc)),temps_acc)
plt.savefig(chemin+'acc'+nom+'.png')

f = open(chemin + 'arb_python.txt','w+')
f.write('temps_acc mean -> ' + str(np.array(temps_acc).mean())+'\n')
f.write('temps_acc std -> ' + str(np.array(temps_acc).std())+'\n')
f.write('temps_time mean -> ' + str(np.array(temps_time).mean())+'\n')
f.write('temps_time std -> ' + str(np.array(temps_time).std())+'\n')
f.close()