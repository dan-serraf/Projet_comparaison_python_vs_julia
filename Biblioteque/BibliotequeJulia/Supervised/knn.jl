import CSV as csv
import DataFrames as fr
import Plots as plt

# mutable struct ClassifierKNN
#     X::Matrix{Float64}
#     y::Vector{Int64}
#     k::Int64 
#     p::Int64  
# end

# function initKnn(X::Matrix{Float64} , y::Vector{Int64} , k::Int64 ,p::Int64  )
#     return ClassifierKNN(X, y, k, p)
# end

# function fit!(Knn::ClassifierKNN,  X::Matrix{Float64}, y::Vector{Int64})
#     Knn.X = X
#     Knn.y = y
#     return Knn
# end

# function getDistance(Knn::ClassifierKNN, X::Vector{Float64})
#     return [ sum(  abs.(X .- Knn.X[i,:]) .^ Knn.p ) .^ (1 / Knn.p) for i in 1:size(Knn.X)[1] ]  #Distance Minkowski
# end

# function getVoisin(Knn::ClassifierKNN, X::Vector{Float64})
#     return sortperm(getDistance(Knn,X))[1:Knn.k]
# end

# function score(Knn::ClassifierKNN, X::Vector{Float64})
#     return sum(Knn.y[getVoisin(Knn,X)] .== 1 ) / Knn.k
# end 

# function predict(Knn::ClassifierKNN, X::Vector{Float64})
#     return score(Knn, X) >= 0.5 ? 1 : -1
# end

# function accuracy(Knn::ClassifierKNN, X::Matrix{Float64}, y::Vector{Int64})
#     return size(X)[1] == 0 ? 0 : sum([(predict(Knn,X[i,:]) .== y[i]) / size(X)[1] for i in 1:size(X)[1] ])
# end


mutable struct ClassifierKNN
    X::Matrix{Float64}
    y::Vector{Int64}
    k::Int64 
end

function initKnn(X::Matrix{Float64} , y::Vector{Int64} , k::Int64   )
    """
    Inialise une structure ClassifierKNN.

    Hypothese :
    Dimension X : N lignes correspondants aux nombre d'exemples.
                  D colonnes correspondants aux différentes caractéristiques de nos données.
    Dimension y : N lignes correspondants aux labels pour chaque exemples.
    k >= 1      : k représente le nombre de voisin que l'on va voir pour prédire, par défaut k=5 .
    
    @X     : Matrix{Float64}
    @y     : Vector{Int64}
    @k     : Int64
    return : ClassifierKNN
    """

    return ClassifierKNN(X, y, k)
end

function fit!(Knn::ClassifierKNN,  X::Matrix{Float64}, y::Vector{Int64})
    """
    Fonction qui va entrainé notre modèle.

    Hypothese :
    Dimension X : N lignes correspondants aux nombre d'exemples.
                  D colonnes correspondants aux différentes caractéristiques de nos données.
    Dimension y : N lignes correspondants aux labels pour chaque exemples.
                    
    @Knn   : ClassifierKNN
    @X     : Matrix{Float64}
    @y     : Vector{Int64}
    @k     : Int64
    return : ClassifierKNN
    """
    Knn.X = X
    Knn.y = y
    return Knn
end

function getDistance(Knn::ClassifierKNN, X::Vector{Float64})
    """
    Fonction qui retourne un Vector qui corresponds au distance entre X donnée en paramètre et 
    tous les autres données apprit précedemment lors de l'appel de fonction fit.

    Hypothese :
    Dimension X : 1 lignes correspondant a un exemples.
              D colonnes correspondants aux différentes caractéristiques de notre donnée.
         return : N lignes correspondant entre X et les différents données apprit précédemment.

        @Knn   : ClassifierKNN
        @X     : Vector{Float64}
        return : ClassifierKNN
    """
    dist = sum(x -> x^2, Knn.X .- reshape(X,1,size(X)[1]); dims=2)
    dist .= sqrt.(dist)
    return dist[:]
end

function getVoisin(Knn::ClassifierKNN, X::Vector{Float64})
    """
        Fonction qui retourne un array qui corresponds au k plus proche voisin entre X donnée en paramètre et 
        tous les autres données apprit précedemment lors de l'appel de fonction fit.

        Hypothese :
        Dimension X : 1 lignes correspondant a un exemples.
                      D colonnes correspondants aux différentes caractéristiques de notre donnée.
             return : k lignes correspondant aux k plus proche voisins de X données en paramètre.

             @Knn  : ClassifierKNN
             @X     : Vector{Float64}
             return : ClassifierKNN
    """
    return Knn.y[sortperm(getDistance(Knn,X))[1:Knn.k]]
end

function predict(Knn::ClassifierKNN, X::Vector{Float64})
    """
    Fonction qui va prédire de quel classe appartient notre données X.

    Hypothese :
    Dimension X : 1 lignes correspondant a un exemples.
              D colonnes correspondants aux différentes caractéristiques de notre donnée.

    @Knn  : ClassifierKNN
    @X     : Vector{Float64}
    return : ClassifierKNN
    """
    y_pred = getVoisin(Knn,X)
    return sb.mode(y_pred)
end

function accuracy(Knn::ClassifierKNN, X::Matrix{Float64}, y::Vector{Int64})
    """
    Fonction qui va donnée l'accuracy de notre modèle sur les données X et y données en paramètre.

    Hypothese :
    Dimension X : N lignes correspondants aux nombre d'exemples.
                    D colonnes correspondants aux différentes caractéristiques de nos données.
    Dimension y : N lignes correspondants aux labels pour chaque exemples.
                    

    @Knn   : ClassifierKNN
    @X     : Matrix{Float64}
    @y     : Vector{Int64}
    return : ClassifierKNN
    """
    return size(X)[1] == 0 ? 0 : st.mean([(predict(Knn,X[i,:]) .== y[i])  for i in 1:size(X)[1] ])
end

