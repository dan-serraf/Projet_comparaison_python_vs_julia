
import CSV as csv
import DataFrames as fr
import Plots as plt
import StatsBase as stb
import Statistics as st
using ProgressBars
import PyCall as pc
arb = pc.pyimport("sklearn.tree")

profondeur=50
nb_iter = collect(1:1:3)
nom_fichier = "../../../mnist_split/"

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


for i in tqdm(nb_iter)

    train_x = list_data[(i-1)*4+1]
    test_x =  list_data[(i-1)*4+2]
    train_y = list_data[(i-1)*4+3]
    test_y =  list_data[(i-1)*4+4]
    

    debut = time()
    clf = arb.DecisionTreeClassifier(random_state=0)
    clf.fit(train_x,train_y)
    y2 = clf.predict(test_x)
    acc = st.mean(y2 .== test_y)
    fin = time()
    append!(temps_time,[fin-debut])
    append!(temps_acc,[acc])

end

nn1 = "../../../Image/ImageJulia/ArbreDecision/timearb_skl_julia.png"
nn2 = "../../../Image/ImageJulia/ArbreDecision/accarb_skl_julia.png"

plt.scatter(nb_iter,temps_time,title="Temps exécution arbre de decision profondeur=5 julia sklearn",xlabel="Nombre itération validation croisée",ylabel="Temps d'exécution en seconde")
plt.savefig(nn1)


plt.plot(nb_iter,temps_acc,title="Julia ",xlabel="Nombre de profondeur",ylabel="Accuracy")
plt.savefig(nn2)

f = open("../../../Image/ImageJulia/ArbreDecision/arb_skl_julia.txt","w+")
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