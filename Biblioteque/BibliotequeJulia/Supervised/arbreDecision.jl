import CSV as csv
import DataFrames as fr
import Plots as plt
import StatsBase as stb

mutable struct Node
    """
    Classe abstraite qui represente notre classe de base pour un noeud de l'arbre.

    label       : Représente le nom de la valeur de la colonne sur lesquel on a divisé les données.
    threshold   : Correspond a nothing si la valeur est catégorielle et un nombre flotant si les 
                  valeurs sont numériques continues
    estFeuille  : Mis a True si le noeud est une feuille et False sinon
    enfants     : List des noeuds enfants du noeud courant.
    list_values : List des valeurs des noeuds enfants du noeud courant.

    @label       : String
    @threshold   : nothing / float
    @estFeuille  : bool
    @enfants     : Vector{Node}
    @list_values : Vector{String}
    """
    estFeuille
    label
    threshold
    enfants
    list_values
end

mutable struct ClassifierArbreDecision
    """
    X                : N lignes correspondants aux nombre d'exemples.
                        D colonnes correspondants aux différentes caractéristiques de nos données.
    y                : N lignes correspondants aux labels pour chaque exemples.
    tree             : A l'initialisation nothing, puis a la fin de l'éxécution corresponds.
                        à l'arbre construit.
    attribut_valeurs : List des noms d'attributs.
    type_attribute   : List des types d'attributs.
    attributes       : List des attributs pas encore traité.
    __index_attribut : Dictionnaire clé : attribut de nom de colonne 
                                    valeur : index dans la list
    profondeur       : Profondeur maximal de l'arbre supérieur à 0.
    func_threshold   : Actuellement prends la valeur 'mean' ou 'median' sinon  

    @X                 : Matrix{Any}
    @y                 : Vector{Any}
    @tree              : Node / nothing
    @attribut_valeurs  : Vector{string}
    @type_attribute    : Vector{type}
    @attributes        : Vector{string}
    @__index_attribut  : dict{string -> int}
    @profondeur        : int
    @func_threshold    : string 
    """
    X
    y
    tree
    attribut_valeurs
    attributes
    type_attribute
    index_attribut
    profondeur
    func_threshold
end

function initArbreDecision(X,y,attribut_valeurs,type_attribute,attributes,profondeur,func_threshold)
    """
    Inialise une structure ClassifierArbreDecision.

    X                : N lignes correspondants aux nombre d'exemples.
                        D colonnes correspondants aux différentes caractéristiques de nos données.
    y                : N lignes correspondants aux labels pour chaque exemples.
    attribut_valeurs : List des noms d'attributs.
    type_attribute   : List des types d'attributs.
    attributes       : List des attributs pas encore traité.
    profondeur       : Profondeur maximal de l'arbre supérieur à 0.
    func_threshold   : Actuellement prends la valeur 'mean' ou 'median' sinon  

    @X                 : Matrix{Any}
    @y                 : Vector{Any}
    @attribut_valeurs  : Vector{string}
    @type_attribute    : Vector{type}
    @attributes        : Vector[string]
    @profondeur        : int
    @func_threshold    : string 
    return             : ClassifierArbreDecision
    """
    tree = Node(false,-1.0,-1.0,[],[])
    func_threshold2 = func_threshold == "mean" ? st.mean : st.median
    return ClassifierArbreDecision(X,y,tree,attribut_valeurs,attributes,type_attribute,Dict(),profondeur,func_threshold2)
end

function afficheArbre(arbre)
    """
    Affiche depuis la racine, l'arbre de décision construit suite à l'appel de fonction fit .

    @arbre  : ClassifierArbreDecision
    return  : nothing
    """
    afficheNoeud(arbre,arbre.tree,"")
end

function afficheNoeud(arbre, node, indent)
    """
    Affiche depuis un noeud, l'arbre de décision construit suite à l'appel de fonction fit.

    @arbre   : ClassifierArbreDecision
    @node   : Node
    @indent : String
    return  : nothing
    """
    if ! node.estFeuille
        
    
        if isnothing(node.threshold) 
            #discrete
            
            for (index,child) in enumerate(node.enfants)   
                if child.estFeuille
                    k = node.list_values[index]
                    print(indent)
                    print(node.label)
                    print(" = ")
                    print(k)
                    print(" : ")
                    print(child.label)
                    print("\n")
                else
                    k = node.list_values[index]
                    print(indent)
                    print(node.label)
                    print(" = ")
                    print(k)
    
                    print(" : ")
                     print("\n")
                afficheNoeud(arbre, child, string(indent , "	"))
                end
            end
        else
            #numerical
            leftChild = node.enfants[1]
            rightChild = node.enfants[2]
            
            if leftChild.estFeuille
                print(indent)
                print(node.label)
                print(" <= ")
                print(node.threshold)
                print(" : ")
                print(leftChild.label)
                print("\n")
            else
                print(indent)
                print(node.label)
                print(" <= ")
                print(node.threshold)
                print(" : ")
                print("\n")
                afficheNoeud(arbre, leftChild, string(indent , "	"))
            end
        
            if rightChild.estFeuille
                print(indent)
                print(node.label)
                print(" > ")
                print(node.threshold)
                print(" : ")
                print(rightChild.label)
                 print("\n")
            else
                print(indent)
                print(node.label)
                print(" > ")
                print(node.threshold)
                print(" : ")
                print("\n")
                afficheNoeud(arbre , rightChild ,  string(indent , "	"))
            end
        end
    end
end

function fit!(arbre,X,y,attribut_valeurs,attributes,type_attribute)
    """
    Fonction qui va entrainé notre modèle.

    X                : N lignes correspondants aux nombre d'exemples.
                        D colonnes correspondants aux différentes caractéristiques de nos données.
    y                : N lignes correspondants aux labels pour chaque exemples.
    tree             : A l'initialisation nothing, puis a la fin de l'éxécution corresponds.
                        à l'arbre construit.
    attribut_valeurs : List des noms d'attributs.
    type_attribute   : List des types d'attributs.
    attributes       : List des attributs pas encore traité.
    __index_attribut : Dictionnaire clé : attribut de nom de colonne 
                                    valeur : index dans la list
    profondeur       : Profondeur maximal de l'arbre supérieur à 0.
    func_threshold   : Actuellement prends la valeur 'mean' ou 'median' sinon  

    @arbre              : ClassifierArbreDecision
    @X                 : Matrix{Any}
    @y                 : Vector{Any}
    @tree              : Node / nothing
    @attribut_valeurs  : Vector{string}
    @type_attribute    : Vector{type}
    @attributes        : Vector{string}
    @__index_attribut  : dict[string -> int]
    @profondeur        : int
    @func_threshold    : string 
    return             : ClassifierArbreDecision
    """
    arbre.X = X
    arbre.y = y
    arbre.attribut_valeurs = attribut_valeurs
    arbre.type_attribute = type_attribute
    arbre.attributes = attributes
    arbre.index_attribut = Dict(val => i for (i,val) in enumerate(arbre.attributes)) 
    arbre.tree = construitArbreDecision(arbre,arbre.X,arbre.y, arbre.attributes)
    
    return arbre
end

function construitArbreDecision(arbre,X,y ,attributes)
    """
    Fonction qui va construire notre arbre de décision.

    X          : N lignes correspondants aux nombre d'exemples.
                 D colonnes correspondants aux différentes caractéristiques de nos données.
    y          : N lignes correspondants aux labels pour chaque exemples.
    attributes : List des attributs pas encore traité.

    @arbre       : ClassifierArbreDecision
    @X          : Matrix{Any}
    @y          : Vector{Any}
    @attributes : Vector{string}
    return      : Node
    """
    
    bool_unique = classeUnique(arbre,y)
    if bool_unique != false #return a node with that class
        return Node(true, bool_unique, nothing,[],[])

    elseif length(attributes) == 0 || ((length(arbre.type_attribute) - length(attributes)) >= arbre.profondeur)
        return Node(true, classeMajoritaire(arbre,y), nothing,[],[])
    else
        (best,best_threshold,list_X,list_y,attribut_valeurs) = discretiseAttribute(arbre,X,y, attributes)
        
        attribut_modifier = copy(attributes)
        filter!( val -> val != best, attribut_modifier)
       
        node = Node(false, best, best_threshold,[],[])
        node.list_values = attribut_valeurs
        node.enfants = [construitArbreDecision(arbre,X,y, attribut_modifier) for (X,y) in zip(list_X,list_y) if length(y) > 0]
        return node
    end
end


function discretiseAttribute(arbre,X,y ,attributes)
    """
    Fonction qui va discrétiser nos données X et y.

    X          : N lignes correspondants aux nombre d'exemples.
                 D colonnes correspondants aux différentes caractéristiques de nos données.
    y          : N lignes correspondants aux labels pour chaque exemples.
    attributes : List des attributs pas encore traité.

    @arbre       : ClassifierArbreDecision
    @X          : Matrix{Any}
    @y          : Vector{Any}
    @attributes : Vector{string}
    return      : Tuple(String * (None|float) * Vector[Matrix{Any}] * Vector[Matrix{Any}] * Vector[string])
    """
    splitted_X = []
    splitted_y = []
    max_entropy = -Inf
    best_attribute = -1
    best_threshold = nothing # None -> attributs discrets , threshold -> attributs continus 
    best_attribute_list = []

    for attribute in attributes
        index = findfirst(x -> x == attribute, arbre.attributes)
        
        attribut_valeurs = unique( X[:,index]) # #liste des valeurs  prises par l'attribut
        
        if isAttrDiscrete(arbre,attribute)
            
            list_X,list_y = discretiseAttributeDiscret(arbre, X,y, index,attribut_valeurs)
            e = gain(arbre,y, list_X,list_y)
            
            if e >= max_entropy
                
                max_entropy,splitted_X,splitted_y,best_attribute,best_attribute_list = e,list_X,list_y,attribute,attribut_valeurs
            end
        else

#             arr = sort(X[:,index])
            
#             for j in collect(1:1:length(arr)-1)
#                 if arr[j] != arr[j+1] 
#                     threshold = (arr[j] + arr[j+1]) / 2
            threshold = arbre.func_threshold(X[:,index])
            list_X,list_y = discretiseAttributeContinue(arbre, X,y, index,threshold)
            e = gain(arbre,y, list_X,list_y)
            if e >= max_entropy
                max_entropy,splitted_X,splitted_y,best_attribute,best_threshold,best_attribute_list = e,list_X,list_y,attribute,threshold,attribut_valeurs
            end
#                 end
#             end
        end
    end
    return (best_attribute,best_threshold,splitted_X,splitted_y,best_attribute_list)
       
end

function discretiseAttributeDiscret(arbre,X,y, index,attributs_valeurs) 
    """
    Fonction qui va discrétiser nos données X et y discrete.

    X                  : N lignes correspondants aux nombre d'exemples.
                         D colonnes correspondants aux différentes caractéristiques de nos données.
    y                  : N lignes correspondants aux labels pour chaque exemples.
    index              : Numéro de l'index de l'attribut courant.
    attributs_valeurs  : List des noms d'attributs.

    @arbre              : ClassifierArbreDecision
    @X                 : Matrix{Any}
    @y                 : Vector{Any}
    @index             : int
    @attribut_valeurs  : Vector{string}
    return             : Tuple(Vector[Matrix{Any}] * Vector[Matrix{Any}])
    """
    list_X = []
    list_y = []
  
    for v in attributs_valeurs 
        ind = findall(x -> x == v , X[:,index])
        append!(list_X,[X[ind,:]])
        append!(list_y,[y[ind]])
    end
    
    return list_X,list_y
end

function discretiseAttributeContinue(arbre,X,y, index,threshold)
    """
    Fonction qui va discrétiser nos données X et y continue.

    X          : N lignes correspondants aux nombre d'exemples.
                 D colonnes correspondants aux différentes caractéristiques de nos données.
    y          : N lignes correspondants aux labels pour chaque exemples.
    index      : Numéro de l'index de l'attribut courant.
    threshold  : Valeur qui va séparer nos donnés.

    @arbre     : ClassifierArbreDecision
    @X         : Matrix{Any}
    @y         : Vector{Any}
    @index     : int
    @threshold : float
    return     : Tuple(Vector[Matrix{Any}] * Vector[Matrix{Any}])
    """
    ind1,ind2 = findall(x -> x <= threshold , X[:,index]),findall(x -> x > threshold , X[:,index])
    return [X[ind1,:],X[ind2,:]],[y[ind1],y[ind2]]
end

function isAttrDiscrete(arbre, attribute)
    """
    Fonction qui retourne True si l'attribut est discret et False sinon.

    attributes : List des attributs pas encore traité.

    @arbre       : ClassifierArbreDecision
    @attributes : string
    return      : bool
    """
    return arbre.attribut_valeurs[attribute] == 1 ?  false : true
end

function classeUnique(arbre, Y)
    """
    Fonction qui retourne False s'il ne reste pas une unique classe si la classe 
    est unique on retourne le nom de la classe.

    y          : N lignes correspondants aux labels pour chaque exemples.
    
    @arbre      : ClassifierArbreDecision
    @y          : Vector{Any}
    return      : bool|string
    """
    Y2 = unique(Y)
    return length(Y2) > 1 ? false : Y2[1]
end

function gain(arbre,Y, list_X,list_y)
    """
    Fonction qui calcule le gain en fonction des données courantes y, list_X et list_y.

    list_X  : list des données X diviser en fonction de l'attribut courant.
    list_y  : list des données y diviser en fonction de l'attribut courant.
    Y       :  N lignes correspondants aux labels pour chaque exemples.

    @arbre   : ClassifierArbreDecision
    @list_X  : Vector[Matrix{Any}]
    @list_y  : Vector[Matrix{Any}]
    @Y       : Vector{Any}
    return   : float
    """
    poids = [length(X) for X in list_X] / length(Y)
    return entropy(arbre,Y) .- sum([ poids[i].*entropy(arbre,list_y[i]) for i in collect(1:1:size(list_X)[1])])
end

function shannon(arbre,array_proba)
    """
    Fonction qui calcule shannon sur nos données.

    array_proba   :  Tableaux de probabilités des différentent classes.

    @arbre        : ClassifierArbreDecision
    @array_proba  : Vector{float}
    return        : float
    """
    return -1 * sum(array_proba .* log.(array_proba))
end


function entropy(arbre, Y)
    """
    Fonction qui calcule entropy sur nos données.

    Y      :  N lignes correspondants aux labels pour chaque exemples.
    
    @arbre : ClassifierArbreDecision
    @Y     : Vector{Any}
    return : float
    """
    return length(Y) == 0 ? 0 : shannon(arbre,[i for i in values(stb.countmap(Y))] /length(Y)) 
end



function classeMajoritaire(arbre,Y)
    """
    Fonction qui retourne la classe majoritaire.

    Y      :  N lignes correspondants aux labels pour chaque exemples.
    
    @arbre : ClassifierArbreDecision
    @Y     : Vector{Any}
    return : string
    """
    return stb.mode(Y)
end

function indexValeurContinue(arbre,value,threshold)
    """
    Fonction qui retourne l'index en fonction de la valeur threshold.

    value     : Valeur courante à comparer.
    threshold : Valeur de threshold.
    
    @arbre      : ClassifierArbreDecision
    @value     : float
    @threshold : float
    return     : int
    """
    return value <= threshold ? 1 : 2
end

function indexValeurDiscrete(arbre,node,val)
    """
    Fonction qui retourne l'index en fonction de la valeur value.

    node  : Noeud courant.
    value : Valeur courante à comparer.
    
    @arbre  : ClassifierArbreDecision
    @node  : Node
    @value : string
    return : int
    """
    for i in collect(1:1:length(node.list_values)) 
        if node.list_values[i] == val 
            return i
        end
    end
    return 1
end


function predict(arbre,Xi)
    """
    Prédit la classe de la donnée en paramètre.

    Xi  : 1 lignes correspondants aux nombre d'exemples.
          D colonnes correspondants aux différentes caractéristiques de nos données.
    
    @arbre  : ClassifierArbreDecision
    @Xi    : Node
    return : string
    """
    return __predict( arbre,arbre.tree ,Xi)
end

function __predict(arbre, node,Xi) 
    """
    Prédit la classe de la donnée en paramètre.

    node : Noeud courant
    Xi   : 1 lignes correspondants aux nombre d'exemples.
            D colonnes correspondants aux différentes caractéristiques de nos données.
    
    @arbre  : ClassifierArbreDecision
    @node  : Node
    @Xi    : Vector{Any}
    return : string
    """
    if node.estFeuille 
        return node.label
    
    elseif isnothing(node.threshold) 
        tmp = indexValeurDiscrete(arbre,node,Xi[arbre.index_attribut[node.label]])
        
        return __predict(arbre,node.enfants[tmp],Xi)
    
    else
        try 
            tmp = indexValeurContinue(arbre,Xi[arbre.index_attribut[node.label]],node.threshold)
            
            return __predict(arbre,node.enfants[tmp],Xi)
        catch
            return classeMajoritaire(arbre,[__predict(arbre,noeud,Xi) for noeud in node.enfants])
        end
    end
       
end

function accuracy(arbre, X, y)
    """
    Fonction qui va donnée l'accuracy de notre modèle sur les données X et y données en paramètre.

    Hypothese :
    Dimension X : N lignes correspondants aux nombre d'exemples.
                    D colonnes correspondants aux différentes caractéristiques de nos données.
    Dimension y : N lignes correspondants aux labels pour chaque exemples.
                    

    @arbre : ClassifierArbreDecision
    @X     : Matrix{Any}
    @y     : Vector{Any}
    return : float
    """
    return size(X)[1] == 0 ? 0 : st.mean([(predict(arbre,X[i,:]) .== y[i])  for i in 1:size(X)[1] ])
end


