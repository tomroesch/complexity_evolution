using CSV, DataFrames, Dates, LinearAlgebra, Distributions, DelimitedFiles, SharedArrays

import Base.Threads.@spawn
# Custom package
using Jevo


# Get date to append to output file
date = Dates.format(Dates.today(), "yyyy_mm_dd")




gap = 10
l_0 = 20
n = 4
N = 100
steps = 1*10^5
fl = 0.6l_0/2N
f0 = 200l_0/2N
κ_arr = [0, 1, 2, 3, 4, 5, 10, 20, 40]
F = Jevo.num_fermi(n, l_0, gap, f0, fl)
emat = gap/l_0 * (ones(n, n) - Matrix{Float64}(I, n, n))

function make_histogram(data)
    x = sort(unique(data))
    y = [sum(data .== t) for t in x]
    return x, y/sum(y)
end

# Run one rep
function Run(N, l, emat, F, κ, l_0, gap, steps)
    ad_steps = Int(floor((steps*(1+κ)/2)))
    E_arr = zeros(Float64, ad_steps)
    L_arr = zeros(Float64, ad_steps)
    pop = Jevo.mono_pop(N=N, l=l)
    Jevo.initiate!(pop, emat)
    for r in 1:ad_steps
        Jevo.bp_substitution!(pop, emat, F)
        if rand() < κ/N
            Jevo.driver_mutation!(pop)
        end
        if rand() < 1/(10N * (1+κ))
            Jevo.l_substitution!(pop, emat, F)
        end
        E_arr[r] = Jevo.get_energy(pop, emat)
        L_arr[r] = length(pop.seqs)
        if E_arr[r]/(L_arr[r] * gap/l_0) > (Jevo.γ_1(l, n, l_0) + 0.15)
            Jevo.initiate!(pop, emat)
        end 
    end
    return E_arr, L_arr
end


x_list = Array{Array{Float64, 1}, 1}()
y_list = Array{Array{Float64, 1}, 1}()
κ_list = Array{Array{Float64, 1}, 1}()
# Predefine output arrays
for i in 1:length(κ_arr)
    push!(x_list, Float64[])
    push!(y_list, Float64[])
    push!(κ_list, Float64[])
end

Threads.@threads for i in 1:length(κ_arr)
    E, L = Run(N, 100, emat, F, κ_arr[i], l_0, gap, 10)
    x, y = make_histogram(L)
    x_list[i] = x
    y_list[i] = y
    κ_list[i] = ones(length(x)) .* κ_arr[i]
    println("Run $i done.")
end


df = DataFrame(p_l=[(y_list...)...], l=[(x_list...)...], kappa=[(κ_list...)...])
#=
CSV.write(date * "_results.csv", df)
    

open(date*"_METADATA.txt", "a") do io
    write(io, "gap=$gap\n")
    write(io, "l_0=$l_0\n")
    write(io, "f0=$f0\n")
    write(io, "fl=$fl\n")
    write(io, "kappa=$κ_arr\n")
    write(io, "l=$l_arr\n")
    write(io, "n=$n\n")
    write(io, "N=$N\n")
    write(io, "steps=$steps\n")
end

=#
