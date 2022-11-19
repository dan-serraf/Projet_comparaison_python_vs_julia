import CSV as csv
import DataFrames as fr
import Plots as plt
import StatsBase as sb
import Statistics as st
import LinearAlgebra as la
using ProgressBars

include("../../../Biblioteque/BibliotequeJulia/Supervised/knn.jl")
include("../../../Biblioteque/BibliotequeJulia/Utilitaire/utilitaire.jl")

k=5
nb_iter = collect(1:1:3)
nom_fichier = "../../../mnist_split/"

list_data = []
for iter in tqdm(nb_iter )
    train_x = Matrix{Float64}(csv.read(nom_fichier* "train_x" * string(iter-1) * ".csv",fr.DataFrame))[1:100,:]
    test_x =  Matrix{Float64}(csv.read(nom_fichier* "test_x" * string(iter-1) * ".csv",fr.DataFrame))[1:100,:]
    train_y = Vector{Int64}(Matrix(csv.read(nom_fichier* "train_y" * string(iter-1) * ".csv",fr.DataFrame))[1:100])
    test_y =  Vector{Int64}(Matrix(csv.read(nom_fichier* "test_y" * string(iter-1) * ".csv",fr.DataFrame))[1:100])
    append!(list_data,(train_x,test_x,train_y,test_y))
end



temps_time = Float64[]
temps_acc = Float64[]

for i in tqdm(nb_iter)
    
    train_x = list_data[(i-1)*4+1]
    test_x =  list_data[(i-1)*4+2]
    train_y = list_data[(i-1)*4+3]
    test_y =  list_data[(i-1)*4+4]

    debut = time()
    knn =  initKnn(train_x, train_y, k)
    knn = fit!(knn,train_x,train_y)
    acc = accuracy(knn,test_x,test_y)
    fin = time()
    append!(temps_time,[fin-debut])
    append!(temps_acc,[acc])

end

nn1 = "../../../Image/ImageJulia/KPlusProcheVoisin/timeknn_julia.png"
nn2 = "../../../Image/ImageJulia/KPlusProcheVoisin/accknn_julia.png"
plt.scatter(nb_iter,temps_time,title="Julia ",xlabel="Nombre de k",ylabel="Temps seconde")
plt.savefig(nn1)


plt.scatter(nb_iter,temps_acc,title="Julia ",xlabel="Nombre de k",ylabel="Accuracy")
plt.savefig(nn2)

f = open("../../../Image/ImageJulia/KPlusProcheVoisin/knn_julia.txt","w+")
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