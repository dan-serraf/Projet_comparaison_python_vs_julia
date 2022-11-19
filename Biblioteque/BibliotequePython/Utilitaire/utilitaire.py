import copy
import numpy as np

def extractOneVsOne(X,y,classe1,classe2):
    X2 = copy.deepcopy(X) 
    y2 = copy.deepcopy(y)
    ind1 = np.where(y2 == classe1)[0] 
    ind2 = np.where(y2 == classe2)[0]
    ind3 = np.append(ind1,ind2)
    y2[ind1] = 1 
    y2[ind2] = -1
    return X2[ind3],y2[ind3]

def extractOneVsAll(X,y,classe1):
    X2 = copy.deepcopy(X) 
    y2 = copy.deepcopy(y)
    ind1 = np.where(y2 == classe1)[0] 
    ind2 = np.where(y2 != classe1)[0]
    ind3 = np.append(ind1,ind2)
    y2[ind1] = 1 
    y2[ind2] = -1
    return X2[ind3],y2[ind3]


def dict_attribut(y):

    dico = dict()
    unique_y = np.unique(y)
    for i in range(len(unique_y)):
        dico[unique_y[i]] = i 
    return dico


def convert_categoriel_to_numerique(dico,y) :
    return [dico[y[i]] for i in  range(len(y))]

def dict_continus_discret(attributes,type_attribute):
    '''
    Fonction qui va entrainé notre modèle.

    attributes       : List des attributs pas encore traité.
    type_attribute   : List des types d'attributs.   

    @attributes      : list[string]
    @type_attribute  : numpy.ndarray[type]
    return           : dict[string->int]
    '''
    attribut_valeurs = {}
    for i in range(len(attributes)) :
        if 'float' in str(type_attribute[i]) :
            attribut_valeurs[attributes[i]] = 1 # "continuous"
        else :
            attribut_valeurs[attributes[i]] = 0 #"discret"   
    return attribut_valeurs