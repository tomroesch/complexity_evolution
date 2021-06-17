using CSV, DataFrames, Dates, LinearAlgebra, Distributions, DelimitedFiles, SharedArrays, Plots
gr()

import Base.Threads.@spawn
# Custom package
using Jevo


# Get date to append to output file
date = Dates.format(Dates.today(), "yyyy_mm_dd")

# Define Parameters
l0 = 10
ϵ0 = 0.7
n = 4
N = 100
fl = 1/2N
f0 = 400/2N
κ_arr = [0, 0.2, 0.5, 1, 2, 5, 10, 20]#, 20, 45] 
F = Jevo.fermi_fitness(epsilon=ϵ0, f0=f0, fl=fl, l0=l0)
emat = ϵ0* (ones(n, n) - Matrix{Float64}(I, n, n))


function make_histogram(data)
    x = unique(data)
    y = [sum([d == t for d in data]) for t in x]
    return x, y/sum(y)
end


function Run(N, l, emat, F, κ, l0, steps)
    E_arr = zeros(Float64, steps)
    L_arr = zeros(Float64, steps)
    pop = Jevo.mono_pop(N=N, l=l)
    Jevo.initiate!(pop, emat)
    for r in 1:steps
        Jevo.bp_substitution!(pop, emat, F)
        if rand() < κ/N
            Jevo.driver_mutation!(pop)
        end
        if rand() < 1/10
            Jevo.l_substitution!(pop, emat, F)
        end
        E_arr[r] = Jevo.get_energy(pop, emat)
        L_arr[r] = length(pop.seqs)
        if E_arr[r]/(L_arr[r]*ϵ0) > (Jevo.γ_0(n))
            Jevo.initiate!(pop, emat)
        end 
    end
    return E_arr, L_arr
end


# Predefine output arrays
k_list = Array{Array{Float64, 1}, 1}()
l_list = Array{Array{Float64, 1}, 1}()
Q_list = Array{Array{Float64, 1}, 1}()
κ_list = Array{Array{Float64, 1}, 1}()

for i in 1:length(κ_arr)
    push!(l_list, Float64[])
    push!(k_list, Float64[])
    push!(Q_list, Float64[])
    push!(κ_list, Float64[])
end

# uses all availabe cores
Threads.@threads for i in 1:length(κ_arr)
    E, L = Run(N, 40, emat, F, κ_arr[i], l0, 10^8)
    pairs, hist = make_histogram([(x, y) for (x, y) in zip(E[10^6:10^8], L[10^6:10^8])])
    k_list[i] = [pair[1] for pair in pairs]
    l_list[i] = [pair[2] for pair in pairs]
    Q_list[i] = hist
    κ_list[i] = ones(length(pairs)) .* κ_arr[i]
    println("Run κ = $(κ_arr[i]) done.")
end


df = DataFrame(Q=[(Q_list...)...], k=[(k_list...)...], l=[(l_list...)...], kappa=[(κ_list...)...])

CSV.write(date * "_results.csv", df)