import LinearAlgebra as la
import Statistics as st
import Random as rd
import Plots as plt
import CSV as csv
import DataFrames as fr
import StatsBase as stb
using ProgressBars

mutable struct ClassifierKmeans
    """
    Structure ClassifierKmens.

    Hypothese :
    X                 : N lignes correspondants aux nombre d'exemples.
                        D colonnes correspondants aux différentes caractéristiques de nos données.
    y                 : N lignes correspondants aux labels pour chaque exemples.
    learning_rate > 0 : représente le pas avec lesquel on considere que l'algorithme a convergé.
    k >= 1            : k représente le nombre de cluster que l'on obtiendra .
    p >= 1            : p représente la variable dans la distance de Minkowski, par défault k=2 (distance euclidienne).
    iter_max > 0      : iter_max représente le nombre d'itération maximale pour le calcule des centroides.

    @X             : Matrix{float}
    @y             : Vector{Any}
    @learning_rate : float
    @k             : int
    @p             : int
    @iter_max      : int
    @centroides    : numpy.ndarray
    @cluster       : dict[int->list[int]]
    @inerties      : numpy.ndarray
    return         : ClassifierKmens
    """
    X::Matrix{Float64}
    y::Vector{Int64}
    k::Int64
    p::Int64
    learning_rate::Float64
    iter_max::Int64
    centroides::Matrix{Float64}
    cluster::Dict{Int64,Vector{Int64}}
    inerties::Vector{Float64}
end

function initKmeans( X::Matrix{Float64} , y::Vector{Int64} , k::Int64 , p::Int64 ,learning_rate::Float64 , iter_max::Int64)
    """
    Inialise une structure ClassifierKmens.

    Hypothese :
    X                 : N lignes correspondants aux nombre d'exemples.
                        D colonnes correspondants aux différentes caractéristiques de nos données.
    y                 : N lignes correspondants aux labels pour chaque exemples.
    learning_rate > 0 : représente le pas avec lesquel on considere que l'algorithme a convergé.
    k >= 1            : k représente le nombre de cluster que l'on obtiendra .
    p >= 1            : p représente la variable dans la distance de Minkowski, par défault k=2 (distance euclidienne).
    iter_max > 0      : iter_max représente le nombre d'itération maximale pour le calcule des centroides.

    @X             : Matrix{float}
    @y             : Vector{Any}
    @learning_rate : float
    @k             : int
    @p             : int
    @iter_max      : int
    @centroides    : numpy.ndarray
    @cluster       : dict[int->list[int]]
    @inerties      : numpy.ndarray
    return         : ClassifierKmens
    """
    return ClassifierKmeans(X,y,k,p,learning_rate,iter_max,Matrix{Float64}(undef,2,size(X)[2]),Dict{Int64,Vector{Int64}}(),Vector{Float64}(undef,0))
end

function normalisation(kmeans::ClassifierKmeans)
    """
    Fonction qui va normaliser les données.

    Hypothese :
    return    : N lignes correspondants aux nombre d'exemples.
                D colonnes correspondants aux différentes caractéristiques de notre donnée.
    
    @self     : ClassifierKmens
    return    : numpy.ndarray

    """
    mini = minimum(kmeans.X,dims=2) #En julia dims = 1 pour lignes dims=2 pour colonne
    maxi = maximum(kmeans.X,dims=2)
    return (kmeans.X .- mini) ./ (maxi - mini)
end

function dist_vect(kmeans::ClassifierKmeans,Xi::Vector{Float64},Xj::Vector{Float64})
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
    return sum(abs.(Xi .- Xj) .^ kmeans.p ) .^ (1 / kmeans.p)  #Distance Minkowski
end

function dist_vect_cosine(kmeans::ClassifierKmeans,Xi::Vector{Float64},Xj::Vector{Float64})
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
    return (Xi.* Xj) ./ (la.norm(Xi)*la.norm(Xj))
end

function centroide(kmeans::ClassifierKmeans,X::Matrix{Float64})
    """
    Fonction qui va calculer les centroides.

    Hypothese :
    Dimension Xi : 1 lignes correspondants a un exemple.
                    D colonnes correspondants aux différentes caractéristiques de notre donnée.
            
    @self   : ClassifierKmens
    @Xi     : numpy.ndarray
    return  : float

    """
    return st.mean(X, dims=1)
end

function init_centroide(kmeans::ClassifierKmeans)
    """
    Fonction qui va initialiser les centroides.

    Hypothese :     
    return    : k lignes correspondant aux différént centroide.
                D colonnes correspondants aux différentes caractéristiques de notre donnée.
        
    @self     : ClassifierKmens   
    return    : numpy.ndarray

    """
    kmeans.centroides = kmeans.X[rd.rand(1:size(kmeans.X)[1], kmeans.k),:]
    return kmeans.centroides
end

function arg_min(kmeans::ClassifierKmeans,Xi::Matrix{Float64})
    ind::Int64 = 1
    mini::Float64 = Xi[1]
    
    for i in 1:size(Xi)[1]
        if Xi[i] < mini 
            mini = Xi[i]
            ind = i
        end
    end
    return ind
end

function plus_proche(kmeans::ClassifierKmeans,Xi::Vector{Float64}) 
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
    return arg_min(kmeans,sum(abs.(kmeans.centroides .-  reshape(Xi,1,size(kmeans.X)[2])) .^ kmeans.p,dims=2))
end

function affecte_cluster!(kmeans::ClassifierKmeans)
    """
    Fonction qui va affecter les différents points de notre donnés X aux centroides les plus proche.      
        
    @self   : ClassifierKmens
        
    return  : dict[int ->  list[int]] : 
            [numero_cluster -> list des points qui appartienne a ce clusters]
            L'indice des points correspondents aux indices de ligne dans X.

    """
    kmeans.cluster = Dict{Int64,Vector{Int64}}()
    
    for i in 1:size(kmeans.X)[1]
        
        t = plus_proche(kmeans,kmeans.X[i, :])
        try 
            append!(kmeans.cluster[t],i)
        catch
            kmeans.cluster[t] = Vector([i])
        end
    end
    return kmeans.cluster
end

function nouveaux_centroides!(kmeans::ClassifierKmeans)
    """
    Fonction qui va mettre à jour les différents points de notre donnés X aux centroides les plus proche.

    Hypothese :
    return      : k lignes correspondant aux différént centroide.
                    D colonnes correspondants aux différentes caractéristiques de notre donnée.
        
    @self    : ClassifierKmens
    return   : numpy.ndarray
    
    """
    kmeans.centroides = Matrix{Float64}(undef,0,size(kmeans.X)[2])
    for valeur in values(kmeans.cluster)
            kmeans.centroides = vcat(kmeans.centroides,centroide(kmeans,kmeans.X[valeur,:]))
    end
    return kmeans.centroides
end

function inertie_globale(kmeans::ClassifierKmeans)
    """
    Fonction qui va calculer l'inertie globale du modèle.
    
    @self   : ClassifierKmens  
    return  : float

    """
    return sum([ inertie_cluster(kmeans,kmeans.X[valeur,:]) for valeur in values(kmeans.cluster)])
end 

function fit!(kmeans::ClassifierKmeans,X::Matrix{Float64},verbose::Bool)
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
    kmeans.X = X
    init_centroide(kmeans)
    affecte_cluster!(kmeans)
    nouveaux_centroides!(kmeans)
    append!(kmeans.inerties,inertie_globale(kmeans))

    for iter in 2:kmeans.iter_max
        affecte_cluster!(kmeans)
        nouveaux_centroides!(kmeans)
        append!(kmeans.inerties,inertie_globale(kmeans))
        temps = abs(kmeans.inerties[iter] - kmeans.inerties[iter-1])
        # if  temps < kmeans.learning_rate 
        #     break
        # end
        if verbose
            println("iteration ", iter, " Inertie : ", kmeans.inerties[iter], " Difference: ",  temps)
        end
    end
    return kmeans
end

function affiche_resultat(kmeans::ClassifierKmeans,save::Bool,name::String)
    """
    Fonction qui va afficher les points avec leurs différents centroides.

    @self  : ClassifierKmens
    @save  : bool
    @name  : String
    return : None
    """
    plot = plt.scatter()
    for liste_point in values(kmeans.cluster)
        array = kmeans.X[liste_point,:]
        plot = plt.scatter!(array[:,1],array[:,2])
    plot = plt.scatter!(kmeans.centroides[:,1],kmeans.centroides[:,2],seriescolor="red")#
    end
    display(plot)
    if save 
        plt.savefig(name)
    end
end

function inertie_cluster(kmeans::ClassifierKmeans,X::Matrix{Float64})
    """
    Fonction qui retourne l'inertie d'un cluster.

    Hypothese :
    Dimension Xi  : k lignes correspondants aux différents donnés appartnant a un centroides.
                    D colonnes correspondants aux différentes caractéristiques de notre donnée.
        
    @self  : ClassifierKmens
    @Xi    : numpy.ndarray 
    return : int

    """
    return sum(sqrt(sum((X .- centroide(kmeans,X)).^2)).^2)
end
