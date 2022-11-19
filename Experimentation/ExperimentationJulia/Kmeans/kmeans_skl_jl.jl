import LinearAlgebra as la
import Statistics as st
import Random as rd
import Plots as plt
import CSV as csv
import DataFrames as fr
import StatsBase as stb
using ProgressBars
import PyCall as pc
km = pc.pyimport("sklearn.cluster")


k=10
p=2
nb_iter = collect(1:1:3)
nom_fichier = "./mnist_split/"

list_data = []
for iter in tqdm(nb_iter )
    train_x = Matrix{Float64}(csv.read(nom_fichier* "train_x" * string(iter-1) * ".csv",fr.DataFrame))[1:100,:]
    test_x =  Matrix{Float64}(csv.read(nom_fichier* "test_x" * string(iter-1) * ".csv",fr.DataFrame))[1:100,:]
    train_y = Vector{Int64}(Matrix(csv.read(nom_fichier* "train_y" * string(iter-1) * ".csv",fr.DataFrame))[:])[1:100]
    test_y =  Vector{Int64}(Matrix(csv.read(nom_fichier* "test_y" * string(iter-1) * ".csv",fr.DataFrame))[:])[1:100]
    append!(list_data,(train_x,test_x,train_y,test_y))
end



temps_time = Float64[]
temps_acc = Float64[]

learning_rate=0.005
iter_max = 100

for i in tqdm(nb_iter)
    
    train_x = list_data[(i-1)*4+1]
    test_x =  list_data[(i-1)*4+2]
    train_y = list_data[(i-1)*4+3]
    test_y =  list_data[(i-1)*4+4]
    
    debut = time()
    kmeans = km.KMeans(n_clusters=10, random_state=0,max_iter=iter_max).fit(train_x)
   
    a = sort([i for i in values(stb.countmap(train_y))])
    b = sort([i for i in values(stb.countmap(kmeans.labels_))])

    while length(a) !=  length(b) 
        append!(b,0)
    end
    sort!(b)
    
    acc = (1 - (sum( abs.(a-b))./sum(a)))
    fin = time()
    append!(temps_time,[fin-debut])
    append!(temps_acc,[acc])

end

nn1 = "./save/timekmeans_julia.png"
nn2 = "./save/acckmeans_julia.png"
plt.scatter(nb_iter,temps_time,title="Temps exécution  kmeans k=10 julia sklearn",xlabel="Nombre itération validation croisée",ylabel="Temps d'exécution en seconde")

plt.savefig(nn1)


plt.plot(nb_iter,temps_acc,title="Julia ",xlabel="Nombre de k",ylabel="Accuracy")
plt.savefig(nn2)

f = open("./save/kmeans_julia.txt","w+")
write(f,"temps_acc mean -> ")
write(f,string(st.mean(temps_acc)))
write(f,"\n")
write(f,"temps_acc std -> ")
write(f,string( st.std(temps_acc)))
write(f,"\n")
write(f,"temps_time mean -> ")
write(f,string(st.mean(temps_time)))
write(f,"\n")
write(f,"temps_time std -> ")
write(f,string(st.std(temps_time)))
write(f,"\n")
close(f)