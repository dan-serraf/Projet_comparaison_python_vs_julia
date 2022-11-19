# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from abc import ABC,abstractmethod

############## CLASSE ABSTRAITE ##############

############## SUPERVISER ##############

class ClassifierSupervised(ABC):
    """
    Classe abstraite qui represente notre classe de base pour les classes supervisée qui sera 
    utilisée dans le cadre de classification.
    """
    @abstractmethod
    def __init__(self):
        """
        Inialise un object de la classe ClassifierSupervised.
        @self : ClassifierSupervised
        return : ClassifierSupervised
        """
        raise NotImplementedError("Merci d'implémenter cette méthode.")
        
        
    @abstractmethod
    def fit(self, X, y):
        """
        Fonction qui va entrainé notre modèle.

        Hypothese :
        Dimension X : N lignes correspondants aux nombre d'exemples.
                      D colonnes correspondants aux différentes caractéristiques de nos données.
        Dimension y : N lignes correspondants aux labels pour chaque exemples.
                     

        @self  : ClassifierSupervised
        @X     : numpy.ndarray
        @y     : numpy.ndarray
        return : ClassifierSupervised
        """
        raise NotImplementedError("Merci d'implémenter cette méthode.")
    
    @abstractmethod
    def predict(self, X):
        """
        Fonction qui va prédire de quel classe appartient notre données X.

        Hypothese :
        Dimension X : 1 lignes correspondant a un exemples.
                      D colonnes correspondants aux différentes caractéristiques de notre donnée.

        @self  : ClassifierSupervised
        @X     : numpy.ndarray
        return : int
        """
        raise NotImplementedError("Merci d'implémenter cette méthode.")
    
    @abstractmethod
    def accuracy(self, X, y):
        """
        Fonction qui va retourner l'accuracy pour les données X avec leurs labels y.

        Hypothese :
        Dimension X : N lignes correspondants aux nombre d'exemples.
                      D colonnes correspondants aux différentes caractéristiques de nos données.
        Dimension y : N lignes correspondants aux labels pour chaque exemples.
        

        @self  : ClassifierSupervised
        @X     : numpy.ndarray
        @y     : numpy.ndarray
        return : float
        """
        raise NotImplementedError("Merci d'implémenter cette méthode.")


        
############## ALGORITHME BASE ##############

############## SUPERVISER ##############
class ClassifierBaseSupervised(ClassifierSupervised):
    """
    Classe qui represente notre classe de base pour les classes supervisée qui sera 
    utilisée dans le cadre de classification.
    """

    def __init__(self):
        """
        Inialise un object de la classe ClassifierBaseSupervised.
        @self  : ClassifierBaseSupervised
        return : ClassifierBaseSupervised
        """

        self.X = np.array([])
        self.y = np.array([])
        return self       
        
    def fit(self, X, y):
        """
        Fonction qui va entrainé notre modèle.

        Hypothese :
        Dimension X : N lignes correspondants aux nombre d'exemples.
                      D colonnes correspondants aux différentes caractéristiques de nos données.
        Dimension y : N lignes correspondants aux labels pour chaque exemples.
                     

        @self  : ClassifierBaseSupervised
        @X     : numpy.ndarray
        @y     : numpy.ndarray
        return : ClassifierBaseSupervised
        """
        self.X = X
        self.y = y
        return self        
    
    
    def accuracy(self, X, y):
        """
        Fonction qui va donnée l'accuracy de notre modèle sur les données X et y données en paramètre.

        Hypothese :
        Dimension X : N lignes correspondants aux nombre d'exemples.
                      D colonnes correspondants aux différentes caractéristiques de nos données.
        Dimension y : N lignes correspondants aux labels pour chaque exemples.
                     

        @self  : ClassifierBaseSupervised
        @X     : numpy.ndarray
        @y     : numpy.ndarray
        return : ClassifierBaseSupervised
        """
        return 0 if X.shape[0] == 0 else np.array([(self.predict(Xi) == y[i]) for i,Xi in enumerate(X)]).mean()


############## KNN ##############
    
class ClassifierKNN(ClassifierBaseSupervised):
    """
    Classe qui represente le modèle de machine learning K-Plus-Proche voisin qui sera 
    utilisée dans le cadre de classification.
    """

    def __init__(self, k=5 ): 
        """
        Inialise un object de la classe ClassifierKNN.

        Hypothese :
        k >= 1 : k représente le nombre de voisin que l'on va voir pour prédire, par défaut k=5 .
       
        @self  : ClassifierKNN
        @k     : int
        return : ClassifierKNN
        """

        super().__init__()
        self.k = k
              
        

    def getDistance(self,X):
        """
        Fonction qui retourne un array qui corresponds au distance entre X donnée en paramètre et 
        tous les autres données apprit précedemment lors de l'appel de fonction fit.

        Hypothese :
        Dimension X : 1 lignes correspondant a un exemples.
                      D colonnes correspondants aux différentes caractéristiques de notre donnée.
             return : N lignes correspondant entre X et les différents données apprit précédemment.

        @self  : ClassifierKNN
        @X     : numpy.ndarray
        return : numpy.ndarray 
        """

        return np.linalg.norm(self.X - X,axis=1)
    
    def getVoisin(self,X):
        """
        Fonction qui retourne un array qui corresponds au k plus proche voisin entre X donnée en paramètre et 
        tous les autres données apprit précedemment lors de l'appel de fonction fit.

        Hypothese :
        Dimension X : 1 lignes correspondant a un exemples.
                      D colonnes correspondants aux différentes caractéristiques de notre donnée.
             return : k lignes correspondant aux k plus proche voisins de X données en paramètre.

        @self  : ClassifierKNN
        @X     : numpy.ndarray
        return : numpy.ndarray 
        """
        return self.y[np.argsort(self.getDistance(X))[0:self.k]]

    def predict(self, X):
        """
        Fonction qui va prédire de quel classe appartient notre données X.

        Hypothese :
        Dimension X : 1 lignes correspondant a un exemples.
                      D colonnes correspondants aux différentes caractéristiques de notre donnée.

        @self  : ClassifierSupervised
        @X     : numpy.ndarray
        return : int
        """
        return np.bincount(self.getVoisin(X)).argmax()


class ClassifierKnnVectoriser(ClassifierBaseSupervised):
    """
    Classe qui represente le modèle de machine learning K-Plus-Proche voisin qui sera 
    utilisée dans le cadre de classification .
    """

    def __init__(self, k=5 ): 
        """
        Inialise un object de la classe ClassifierKnnVectoriser.

        Hypothese :
        k >= 1 : k représente le nombre de voisin que l'on va voir pour prédire, par défaut k=5 .
       
        @self  : ClassifierKnnVectoriser
        @k     : int
        return : ClassifierKnnVectoriser
        """

        super().__init__()
        self.k = k
       
    def getDistance(self, X):
        """
        Fonction qui retourne un array qui corresponds au distance entre X donnée en paramètre et 
        tous les autres données apprit précedemment lors de l'appel de fonction fit.

        Hypothese :
        Dimension X : N lignes correspondant a un exemples.
                      D colonnes correspondants aux différentes caractéristiques de notre donnée.
             return : N lignes correspondant entre X et les différents données apprit précédemment.

        @self  : ClassifierKnnVectoriser
        @X     : numpy.ndarray
        return : numpy.ndarray 
        """
        return -2 * self.X@X.T + np.sum(X**2,axis=1) + np.sum(self.X**2,axis=1)[:, np.newaxis]

    def getVoisin(self, X):
        """
        Fonction qui retourne un array qui corresponds au k plus proche voisin entre X donnée en paramètre et 
        tous les autres données apprit précedemment lors de l'appel de fonction fit.

        Hypothese :
        Dimension X : N lignes correspondant a un exemples.
                      D colonnes correspondants aux différentes caractéristiques de notre donnée.
             return : k lignes correspondant aux k plus proche voisins de X données en paramètre.

        @self  : ClassifierKnnVectoriser
        @X     : numpy.ndarray
        return : numpy.ndarray 
        """
        return np.argsort(self.getDistance( X), 0)[0:self.k, : ].T 
    
    def predict(self, X):
        """ Y : (array) : array de labels
            rend la classe majoritaire ()
        """
        y_pred = self.y[self.getVoisin(X)]
        return np.array([np.bincount(y).argmax() for y in y_pred])

    def accuracy(self, X, y):
        """
        Fonction qui va donnée l'accuracy de notre modèle sur les données X et y données en paramètre.

        Hypothese :
        Dimension X : N lignes correspondants aux nombre d'exemples.
                      D colonnes correspondants aux différentes caractéristiques de nos données.
        Dimension y : N lignes correspondants aux labels pour chaque exemples.
                     

        @self  : ClassifierKnnVectoriser
        @X     : numpy.ndarray
        @y     : numpy.ndarray
        return : ClassifierKnnVectoriser
        """
        return (y == self.predict(X)).mean()


class Node:
    """
    Classe abstraite qui represente notre classe de base pour un noeud de l'arbre.

    label       : Représente le nom de la valeur de la colonne sur lesquel on a divisé les données.
    threshold   : Correspond a None si la valeur est catégorielle et un nombre flotant si les 
                  valeurs sont numériques continues
    estFeuille  : Mis a True si le noeud est une feuille et False sinon
    enfants     : List des noeuds enfants du noeud courant.
    list_values : List des valeurs des noeuds enfants du noeud courant.

    @label       : String
    @threshold   : None / float
    @estFeuille  : bool
    @enfants     : list[Node]
    @list_values : list[String]
    """
    
    def __init__(self,estFeuille, label, threshold):
        """
        Inialise un object de la classe Node.

        label       : Représente le nom de la valeur de la colonne sur lesquel on a divisé les données.
        threshold   : Correspond a None si la valeur est catégorielle et un nombre flotant si les 
                    valeurs sont numériques continues
        estFeuille  : Mis a True si le noeud est une feuille et False sinon

        @self        : Node
        @label       : String
        @threshold   : None / float
        @estFeuille  : bool
        return       : Node
        """
        self.label = label
        self.threshold = threshold
        self.estFeuille = estFeuille
        self.enfants = []
        self.list_values = []
        
            
class ClassifierArbreDecision(ClassifierBaseSupervised):

    def __init__(self,profondeur):
        '''
        Inialise un object de la classe ClassifierArbreDecision.

        Hypothese :
        profondeur : Profondeur maximal de l'arbre supérieur à 0.

        @self              : ClassifierArbreDecision
        @profondeur        : int
        return             : ClassifierArbreDecision
        '''
        super().__init__()
        self.tree = None
        self.attribut_valeurs = np.array([])
        self.type_attribute = np.array([])
        self.attributes = []
        self.__index_attribut = { }
        self.profondeur = profondeur
        self.func_threshold = np.mean
       

    def afficheArbre(self):
        '''
        Affiche depuis la racine, l'arbre de décision construit suite à l'appel de fonction fit .

        @self   : ClassifierArbreDecision
        return  : None
        '''
        self.afficheNoeud(self.tree)
    
    
    def afficheNoeud(self, node, indent=""):
        '''
        Affiche depuis un noeud, l'arbre de décision construit suite à l'appel de fonction fit.

        @self   : ClassifierArbreDecision
        @node   : Node
        @indent : String
        return  : None
        '''
        if not node.estFeuille:
            if node.threshold is None:
                #discrete
                for index,child in enumerate(node.enfants):                
                    if child.estFeuille:
                        print(indent + node.label + " = " + str(node.list_values[index]) + " : " + child.label)
                    else:
                        print(indent + node.label + " = " + str(node.list_values[index]) + " : ")
                        self.afficheNoeud(child, indent + "	")
            else:
                #numerical
                leftChild = node.enfants[0]
                rightChild = node.enfants[1]
                
                if leftChild.estFeuille:
                    print(indent + node.label + " <= " + str(node.threshold) + " : " + leftChild.label)
                else:
                    print(indent + node.label + " <= " + str(node.threshold)+" : ")
                    self.afficheNoeud(leftChild, indent + "	")

                if rightChild.estFeuille:
                    print(indent + node.label + " > " + str(node.threshold) + " : " + rightChild.label)
                else:
                    print(indent + node.label + " > " + str(node.threshold) + " : ")
                    self.afficheNoeud(rightChild , indent + "	")

    def fit(self,X,y,attributes,type_attribute,attribut_valeurs,threshold='means'):
        '''
        Fonction qui va entrainé notre modèle.

        X                : N lignes correspondants aux nombre d'exemples.
                           D colonnes correspondants aux différentes caractéristiques de nos données.
        y                : N lignes correspondants aux labels pour chaque exemples.
        tree             : A l'initialisation None, puis a la fin de l'éxécution corresponds.
                           à l'arbre construit.
        attribut_valeurs : List des noms d'attributs.
        type_attribute   : List des types d'attributs.
        attributes       : List des attributs pas encore traité.
        __index_attribut : Dictionnaire clé : attribut de nom de colonne 
                                        valeur : index dans la list
        profondeur       : Profondeur maximal de l'arbre supérieur à 0.
        func_threshold   : Actuellement prends la valeur 'mean' ou 'median' sinon  

        @self              : ClassifierArbreDecision
        @X                 : numpy.ndarray
        @y                 : numpy.ndarray
        @tree              : Node / None
        @attribut_valeurs  : numpy.ndarray[string]
        @type_attribute    : numpy.ndarray[type]
        @attributes        : list[string]
        @__index_attribut  : dict[string -> int]
        @profondeur        : int
        @func_threshold    : string 
        return             : ClassifierArbreDecision
        '''

        super().fit(X,y)
        self.attribut_valeurs = attribut_valeurs
        self.type_attribute = type_attribute
        self.attributes = list(attributes)
        self.__index_attribut = { val : i for i,val in enumerate(self.attributes)}
        self.tree = self.construitArbreDecision(self.X,self.y, self.attributes)
        self.func_threshold = np.mean if threshold =='means'  else  np.median
        return self


    def construitArbreDecision(self, X,y, attributes):
        '''
        Fonction qui va construire notre arbre de décision.

        X          : N lignes correspondants aux nombre d'exemples.
                     D colonnes correspondants aux différentes caractéristiques de nos données.
        y          : N lignes correspondants aux labels pour chaque exemples.
        attributes : List des attributs pas encore traité.

        @self       : ClassifierArbreDecision
        @X          : numpy.ndarray
        @y          : numpy.ndarray
        @attributes : list[string]
        return      : Node
        '''

        bool_unique = self.classeUnique(y)

        if bool_unique is not False:#return a node with that class
            return Node(True, bool_unique, None)

        elif len(attributes) == 0 or len(self.type_attribute) - len(attributes) >= self.profondeur: 
            return Node(True, self.classeMajoritaire(y), None)

        else:
            (best,best_threshold,list_X,list_y,attribut_valeurs) = self.discretiseAttribute(X,y, attributes)
            
            attribut_modifier = attributes.copy()
            attribut_modifier.remove(best)
           
            node = Node(False, best, best_threshold)
            node.list_values = attribut_valeurs
            node.enfants = [self.construitArbreDecision(X,y, attribut_modifier) for X,y in zip(list_X,list_y) if len(y) > 0]
            return node




    def discretiseAttribute(self, X,y, attributes):
        '''
        Fonction qui va discrétiser nos données X et y.

        X          : N lignes correspondants aux nombre d'exemples.
                     D colonnes correspondants aux différentes caractéristiques de nos données.
        y          : N lignes correspondants aux labels pour chaque exemples.
        attributes : List des attributs pas encore traité.

        @self       : ClassifierArbreDecision
        @X          : numpy.ndarray
        @y          : numpy.ndarray
        @attributes : list[string]
        return      : Tuple(String * (None|float) * list[numpy.ndarray] * list[numpy.ndarray] * list[string])
        '''
        splitted_X = []
        splitted_y = []
        max_entropy = -np.inf
        best_attribute = -1
        best_threshold = None # None -> attributs discrets , threshold -> attributs continus 
        best_attribute_list = []

        for attribute in attributes:
            
            index = self.attributes.index(attribute)  # np.where( self.attributes == attribute )[0][0]
            attribut_valeurs = np.unique( X[:,index]) # #liste des valeurs  prises par l'attribut
            
            if self.isAttrDiscrete(attribute):
                
                list_X,list_y = self.discretiseAttributeDiscret( X,y, index,attribut_valeurs)
                e = self.gain(y, list_X,list_y)
                if e >= max_entropy:  
                    max_entropy,splitted_X,splitted_y,best_attribute,best_attribute_list = e,list_X,list_y,attribute,attribut_valeurs

            else:

                # arr = sorted(X[:,index])
                
                # for j in range(0, len(arr) - 1):
                #     if arr[j] != arr[j+1]:
                #         threshold = (arr[j] + arr[j+1]) / 2  round(X[:,index].mean(),2)

                threshold =  self.func_threshold(X[:,index])
                list_X,list_y = self.discretiseAttributeContinue( X,y, index,threshold)
                e = self.gain(y, list_X,list_y)
                if e >= max_entropy: 
                    max_entropy,splitted_X,splitted_y,best_attribute,best_threshold,best_attribute_list = e,list_X,list_y,attribute,threshold,attribut_valeurs
        return (best_attribute,best_threshold,splitted_X,splitted_y,best_attribute_list)
           
    
    def discretiseAttributeDiscret(self, X,y, index,attributs_valeurs): 
        '''
        Fonction qui va discrétiser nos données X et y discrete.

        X                  : N lignes correspondants aux nombre d'exemples.
                             D colonnes correspondants aux différentes caractéristiques de nos données.
        y                  : N lignes correspondants aux labels pour chaque exemples.
        index              : Numéro de l'index de l'attribut courant.
        attributs_valeurs  : List des noms d'attributs.

        @self              : ClassifierArbreDecision
        @X                 : numpy.ndarray
        @y                 : numpy.ndarray
        @index             : int
        @attribut_valeurs  : numpy.ndarray[string]
        return             : Tuple(list[numpy.ndarray] * list[numpy.ndarray])
        '''

        list_X = []
        list_y = []
        for v in attributs_valeurs :
            ind = np.where(X[:,index] == v)[0]
            list_X.append(X[ind,:])
            list_y.append(y[ind])
        return list_X,list_y
        
    def discretiseAttributeContinue(self, X,y, index,threshold):
        '''
        Fonction qui va discrétiser nos données X et y continue.

        X          : N lignes correspondants aux nombre d'exemples.
                     D colonnes correspondants aux différentes caractéristiques de nos données.
        y          : N lignes correspondants aux labels pour chaque exemples.
        index      : Numéro de l'index de l'attribut courant.
        threshold  : Valeur qui va séparer nos donnés.

        @self       : ClassifierArbreDecision
        @X          : numpy.ndarray
        @y          : numpy.ndarray
        @index      : int
        @threshold  : float
        return      : Tuple(list[numpy.ndarray] * list[numpy.ndarray])
        '''
        ind1,ind2 = np.where(X[:,index] <= threshold)[0],np.where(X[:,index] > threshold)[0]
        return [X[ind1],X[ind2]],[y[ind1],y[ind2]]
    
    def isAttrDiscrete(self, attribute):
        '''
        Fonction qui retourne True si l'attribut est discret et False sinon.

        attributes : List des attributs pas encore traité.

        @self       : ClassifierArbreDecision
        @attributes : string
        return      : bool
        '''
        return False if self.attribut_valeurs[attribute] == 1 else True

    def classeUnique(self, Y):
        '''
        Fonction qui retourne False s'il ne reste pas une unique classe si la classe 
        est unique on retourne le nom de la classe.

        y          : N lignes correspondants aux labels pour chaque exemples.
       
        @self       : ClassifierArbreDecision
        @y          : numpy.ndarray
        return      : bool|string
        '''
        Y2 = np.unique(Y)
        return False if len(Y2) > 1 else Y2[0]

    def gain(self,Y, list_X,list_y):
        '''
        Fonction qui calcule le gain en fonction des données courantes y, list_X et list_y.

        list_X  : list des données X diviser en fonction de l'attribut courant.
        list_y  : list des données y diviser en fonction de l'attribut courant.
        Y       :  N lignes correspondants aux labels pour chaque exemples.

        @self    : ClassifierArbreDecision
        @list_X  : list[numpy.ndarray]
        @list_y  : list[numpy.ndarray]
        @Y       : numpy.ndarray
        return   : float
        '''
        poids = np.array([len(X) for X in list_X]) / len(Y)
        return self.entropy(Y) - sum([ poids[i]*self.entropy(list_y[i]) for i in range(len(list_X))])

    def shannon(self,array_proba):
        '''
        Fonction qui calcule shannon sur nos données.

        array_proba   :  Tableaux de probabilités des différentent classes.

        @self         : ClassifierArbreDecision
        @array_proba  : numpy.ndarray
        return        : float
        '''
        return -1 * (array_proba @ np.log(array_proba))

    def entropy(self, Y):
        '''
        Fonction qui calcule entropy sur nos données.

        Y      :  N lignes correspondants aux labels pour chaque exemples.
        
        @self  : ClassifierArbreDecision
        @Y     : numpy.ndarray
        return : float
        '''
        return 0 if len(Y) == 0 else self.shannon(np.unique(Y, return_counts=True)[1] /len(Y))
    
    def classeMajoritaire(self,Y):
        '''
        Fonction qui retourne la classe majoritaire.

        Y      :  N lignes correspondants aux labels pour chaque exemples.
        
        @self  : ClassifierArbreDecision
        @Y     : numpy.ndarray
        return : string
        '''
        return np.bincount(Y).argmax()
    
    def indexValeurContinue(self,value,threshold):
        '''
        Fonction qui retourne l'index en fonction de la valeur threshold.

        value     : Valeur courante à comparer.
        threshold : Valeur de threshold.
        
        @self      : ClassifierArbreDecision
        @value     : float
        @threshold : float
        return     : int
        '''
        return 0 if value <= threshold else 1

    def indexValeurDiscrete(self,node,value):
        '''
        Fonction qui retourne l'index en fonction de la valeur value.

        node  : Noeud courant.
        value : Valeur courante à comparer.
        
        @self  : ClassifierArbreDecision
        @node  : Node
        @value : string
        return : int
        '''
        for i in range(len(node.list_values)) :
            if node.list_values[i] == value :
                return i
        return 0 
    
    def predict(self,Xi):
        '''
        Prédit la classe de la donnée en paramètre.

        Xi  : 1 lignes correspondants aux nombre d'exemples.
              D colonnes correspondants aux différentes caractéristiques de nos données.
        
        @self  : ClassifierArbreDecision
        @Xi    : Node
        return : string
        '''
        return self.__predict( self.tree ,Xi)
    
    def __predict(self, node,Xi):
        '''
        Prédit la classe de la donnée en paramètre.

        node : Noeud courant
        Xi   : 1 lignes correspondants aux nombre d'exemples.
               D colonnes correspondants aux différentes caractéristiques de nos données.
        
        @self  : ClassifierArbreDecision
        @node  : Node
        @Xi    : numpy.ndarray
        return : string
        ''' 
        if node.estFeuille :
            return node.label
        
        elif node.threshold == None :
            return self.__predict(node.enfants[self.indexValeurDiscrete(node,Xi[self.__index_attribut[node.label]])],Xi)
        
        else:
            try : 
                return self.__predict(node.enfants[self.indexValeurContinue(Xi[self.__index_attribut[node.label]],node.threshold)],Xi)
            except :
                return self.classeMajoritaire([self.__predict(noeud,Xi) for noeud in node.enfants])


    def accuracy(self, X, y):
        """
        Fonction qui va donnée l'accuracy de notre modèle sur les données X et y données en paramètre.

        Hypothese :
        Dimension X : N lignes correspondants aux nombre d'exemples.
                      D colonnes correspondants aux différentes caractéristiques de nos données.
        Dimension y : N lignes correspondants aux labels pour chaque exemples.
                     

        @self  : ClassifierBaseSupervised
        @X     : numpy.ndarray
        @y     : numpy.ndarray
        return : float
        """
        return (y == [self.predict(Xi) for Xi in X]).mean()

