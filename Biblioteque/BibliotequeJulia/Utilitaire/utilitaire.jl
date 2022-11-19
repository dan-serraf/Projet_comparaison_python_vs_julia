function extractOneVsOne(X::Matrix{Float64},y::Vector{Int64},classe1::Int64 ,classe2::Int64 )
    X2 = deepcopy(X) 
    y2 = deepcopy(y)
    ind1 = findall( x -> x == classe1,y2)
    ind2 = findall( x -> x == classe2,y2)
    ind3 = deepcopy(ind1)
    append!(ind3,ind2)
    y4 = Int64[]
    for i in collect(1:1:length(y2))
        if i in ind1 
            append!(y4,1)
        elseif i in ind2
            append!(y4,-1)
        else
            append!(y4,0)
        end
    end
    return X2[ind3,:],y4[ind3]
end

function extractOneVsAll(X::Matrix{Float64},y::Vector{Int64},classe1::Int64 )
    X2 = deepcopy(X) 
    y2 = deepcopy(y)
    ind1 = findall( x -> x == classe1,y2)
    ind2 = findall( x -> x != classe1,y2)
    ind3 = deepcopy(ind1)
    append!(ind3,ind2)
    y4 = Int64[]
    for i in collect(1:1:length(y2))
        if i in ind1 
            append!(y4,1)
        else
            append!(y4,-1)
        end
    end
    return X2[ind3,:],y4[ind3]
end

function dict_continus_discret(attributes::Vector{String},type_attribute::Vector{DataType})

    attribut_valeurs = Dict{Any,Int64}()

    for i in collect(1:1:length(attributes)) 
        if type_attribute[i] <: AbstractFloat 
            attribut_valeurs[attributes[i]] = 1 # "continuous"
            
        else 
            attribut_valeurs[attributes[i]] = 0 #"discret"   
        end
    end
    return attribut_valeurs

end

function convert_categoriel_to_numerique(dico,y)
    y2 = Int64[]
    for i in 1:length(y)
        append!(y2,dico[y[i]] )
    end
    return y2
end

function dict_attribut(y)
    dico = Dict{String,Int64}()
    unique_y = unique(y)
    for i in 1:length(unique_y)
        dico[unique_y[i]] = i 
    end
    return dico
end