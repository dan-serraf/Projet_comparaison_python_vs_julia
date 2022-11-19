# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from abc import ABC,abstractmethod
from tqdm import tqdm
import time

############## CLASSE ABSTRAITE ##############

############## NON SUPERVISER ##############

class ClassifierUnSupervised(ABC):
    """
    Classe abstraite qui represente notre classe de base pour les classes non supervisée qui sera 
    utilisée dans le cadre de classification binaire.
    """

    @abstractmethod
    def __init__(self):
        """
        Inialise un object de la classe ClassifierUnSupervised.
        @self : ClassifierUnSupervised
        return : ClassifierUnSupervised
        """

        raise NotImplementedError("Merci d'implémenter cette méthode.")
        
    @abstractmethod
    def fit(self, X):
        """
        Fonction qui va entrainé notre modèle.

        Hypothese :
        Dimension X : N lignes correspondants aux nombre d'exemples.
                      D colonnes correspondants aux différentes caractéristiques de nos données.              

        @self  : ClassifierUnSupervised
        @X     : numpy.ndarray
        return : ClassifierUnSupervised
        """
        raise NotImplementedError("Merci d'implémenter cette méthode.")

############## NON SUPERVISER ##############

class ClassifierBaseUnSupervised(ClassifierUnSupervised):
    """
    Classe qui represente notre classe de base pour les classes non supervisée qui sera 
    utilisée dans le cadre de classification binaire.
    """

    def __init__(self):
        """
        Inialise un object de la classe ClassifierBaseUnSupervised.
        @self  : ClassifierBaseUnSupervised
        return : ClassifierBaseUnSupervised
        """

        self.X = np.array([])
        return self

    def fit(self, X):
        """
        Fonction qui va entrainé notre modèle.

        Hypothese :
        Dimension X : N lignes correspondants aux nombre d'exemples.
                      D colonnes correspondants aux différentes caractéristiques de nos données.             

        @self  : ClassifierBaseUnSupervised
        @X     : numpy.ndarray
        return : ClassifierBaseUnSupervised
        """ 
        self.X = X
        return self



class ClassifierKmens(ClassifierBaseUnSupervised):
    """
    Classe qui represente le modèle de machine learning Kmeans qui sera 
    utilisée dans le cadre de classification.
    """

    def __init__(self, learning_rate=0.01,k=3,p=2,iter_max=100):
        """
        Inialise un object de la classe ClassifierKmens.

        Hypothese :
        learning_rate > 0 : représente le pas avec lesquel on considere que l'algorithme a convergé.
        k >= 1            : k représente le nombre de cluster que l'on obtiendra .
        p >= 1            : p représente la variable dans la distance de Minkowski, par défault k=2 (distance euclidienne).
        iter_max > 0      : iter_max représente le nombre d'itération maximale pour le calcule des centroides.

        @self          : ClassifierKmens
        @learning_rate : float
        @k             : int
        @p             : int
        @iter_max      : int
        @centroides    : numpy.ndarray
        @cluster       : dict[int->list[int]]
        @inerties      : numpy.ndarray
        return         : ClassifierKmens
        """

        super().__init__()
        self.k = k
        self.p = p
        self.learning_rate = learning_rate
        self.iter_max = iter_max
        self.centroides = np.array([])
        self.cluster =  dict()
        self.inerties =  []
        
     
    def normalisation(self):
        """
        Fonction qui va normaliser les données.

        Hypothese :
        return    : N lignes correspondants aux nombre d'exemples.
                    D colonnes correspondants aux différentes caractéristiques de notre donnée.
        
        @self     : ClassifierKmens
        return    : numpy.ndarray

        """
        mini = np.min(self.X, axis=0)
        maxi = np.max(self.X, axis=0)
        return (self.X - mini) / (maxi - mini)

    def dist_vect(self,Xi,Xj):
        """
        Fonction qui retourne la distance euclidienne entre 2 points .

        Hypothese :
        Dimension Xi : 1 lignes correspondants a un exemples.
                       D colonnes correspondants aux différentes caractéristiques de notre donnée.
        Dimension Xj : 1 lignes correspondants a un exemples.
                       D colonnes correspondants aux différentes caractéristiques de notre donnée.   
           
        @self  : ClassifierKmens
        @Xi    : numpy.ndarray
        @Xj    : numpy.ndarray
        return : float
        """
        return np.sqrt(np.sum((Xi-Xj)**2))

    def dist_vect_cosine(self,Xi,Xj):
        """
        Fonction qui retourne la distance cosine entre 2 points .

        Hypothese :
        Dimension Xi : 1 lignes correspondants a un exemples.
                       D colonnes correspondants aux différentes caractéristiques de notre donnée.
        Dimension Xj : 1 lignes correspondants a un exemples.
                       D colonnes correspondants aux différentes caractéristiques de notre donnée.   
           
        @self  : ClassifierKmens
        @Xi    : numpy.ndarray
        @Xj    : numpy.ndarray 
        return : float
        """
        return np.dot(Xi,Xj) / (np.linalg.norm(Xi)*np.linalg.norm(Xj))

    def centroide(self,Xi):
        """
        Fonction qui va calculer les centroides.

        Hypothese :
        Dimension Xi : 1 lignes correspondants a un exemple.
                       D colonnes correspondants aux différentes caractéristiques de notre donnée.
              
        @self   : ClassifierKmens
        @Xi     : numpy.ndarray
        return  : float

        """
        return np.mean(Xi, axis=0)

    def inertie_cluster(self,Xi):
        """
        Fonction qui retourne l'inertie d'un cluster.

        Hypothese :
        Dimension Xi  : k lignes correspondants aux différents donnés appartnant a un centroides.
                        D colonnes correspondants aux différentes caractéristiques de notre donnée.
           
        @self  : ClassifierKmens
        @Xi    : numpy.ndarray 
        return : int

        """
        return np.sum(np.sqrt(np.sum((Xi - self.centroide(Xi))**2, axis = 1))**2)

    def init_centroide(self):
        """
        Fonction qui va initialiser les centroides.

        Hypothese :     
        return    : k lignes correspondant aux différént centroide.
                    D colonnes correspondants aux différentes caractéristiques de notre donnée.
           
        @self     : ClassifierKmens   
        return    : numpy.ndarray

        """
        self.centroides = self.X[np.random.choice(len(self.X), self.k)]
        return self.centroides

    def plus_proche(self,Xi):
        """
        Fonction qui retourne le point le plus proche.

        Hypothese :
        Dimension Xi   : 1 lignes correspondants a un exemple.
                         D colonnes correspondants aux différentes caractéristiques de notre donnée.
        return         : indice du point le plus proche
           
        @self   : ClassifierKmens
        @Xi     : numpy.ndarray
        return  : int

        """
        return np.argmin(np.sum(pow(np.abs(self.centroides - Xi),self.p),axis=1))  

    def affecte_cluster(self):
        """
        Fonction qui va affecter les différents points de notre donnés X aux centroides les plus proche.      
           
        @self   : ClassifierKmens
         
        return  : dict[int ->  list[int]] : 
                [numero_cluster -> list des points qui appartienne a ce clusters]
                L'indice des points correspondents aux indices de ligne dans X.

        """
        self.cluster = dict()
        for i in range(len(self.X)):
            t = self.plus_proche(self.X[i, :])
         
            try :
                self.cluster[t].append(i)
            except:
                self.cluster[t] = [i]

        return self.cluster

    def nouveaux_centroides(self):
        """
        Fonction qui va mettre à jour les différents points de notre donnés X aux centroides les plus proche.

        Hypothese :
        return      : k lignes correspondant aux différént centroide.
                      D colonnes correspondants aux différentes caractéristiques de notre donnée.
           
        @self    : ClassifierKmens
        return   : numpy.ndarray
        
        """
    
        self.centroides = np.array( [self.centroide(self.X[ self.cluster[k]]) for k in self.cluster])
        return self.centroides

    def inertie_globale(self):
        """
        Fonction qui va calculer l'inertie globale du modèle.
        
        @self   : ClassifierKmens  
        return  : float

        """
        return np.array([ self.inertie_cluster(self.X[self.cluster[i]]) for i in self.cluster]).sum()


    def fit(self,X,verbose=False):
        """
        Fonction qui va calculer les k clustering.

        Hypothese :
        Dimension X : N lignes correspondant aux nombres d'exemples.
                      D colonnes correspondants aux différentes caractéristiques de notre donnée.
            verbose : True permet d'afficher le numero d'iteration, l'inertie et la différence d'
            inertie par rapport a l'iteration précédente.

        @self     : ClassifierKmens
        @X        : numpy.ndarray
        @verbose  : bool
        return    : ClassifierKmens
        """
        super().fit(X)
        self.init_centroide()
        self.affecte_cluster()
        self.nouveaux_centroides()
        
        self.inerties.append(self.inertie_globale())
        
        for iter in range(1,self.iter_max):
            self.affecte_cluster()
            self.nouveaux_centroides()
            self.inerties.append(self.inertie_globale())
            temps = abs(self.inerties[iter] - self.inerties[iter-1])
            # if  temps < self.learning_rate:
            #     break
            if verbose:
                print("iteration ", iter, " Inertie : ", self.inerties[iter], " Difference: ",  temps)
        
        self.inerties = np.array(self.inerties)
        return self

    def affiche_resultat(self):
        """
        Fonction qui va afficher les points avec leurs différents centroides.

        @self          : ClassifierKmens
        return         : None
        """
        plt.figure()
        for centre in (self.cluster):
            array = self.X[self.cluster[centre]]
            plt.scatter(array[:,0],array[:,1])
        plt.scatter(self.centroides[:,0],self.centroides[:,1],color='r',marker='x')
