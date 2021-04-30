using CSV, DataFrames, Distributed, Dates, LinearAlgebra, Distributions, DelimitedFiles, SharedArrays

# Custom package
using Jevo


# Get date to append to output file
date = Dates.format(Dates.today(), "yyyy_mm_dd")

# Get number of workers as a script argument
if length(ARGS) == 1
    addprocs(parse(Int64, ARGS[1]))
elseif length(ARGS) > 1
    throw(ArgumentError("Only one command line argument (cores)."))
end

# Import packages needed for all workers
@everywhere  begin
    using Jevo
    using Distributions
    using DelimitedFiles
    using SharedArrays
    using LinearAlgebra
end


# Parameters
@everywhere begin
    gap = 10
    l_0 = 20
    fl = 0#.5l_0
    f0 = 100l_0
    κ_arr = 0:2:20
    l_arr = 50:200
    n = 4
    N = 100
    steps = 1*10^5

    F = Jevo.num_fermi(n, l_0, gap, f0/2N, fl/2N)
    emat = gap/l_0 * (ones(n, n) - Matrix{Float64}(I, n, n))
end

@everywhere function make_histogram(data)
    x = sort(unique(data))
    y = [sum(data .== t) for t in x]
    return x, y/sum(y)
end

@everywhere function Run(N, l, emat, F, κ, l_0, gap, steps)
    E_arr = zeros(Float64, steps)
    pop = Jevo.mono_pop(N=N, l=l)
    Jevo.initiate!(pop, emat)
    for r in 1:steps
        Jevo.bp_substitution!(pop, emat, F)
        if rand() < κ/N
            Jevo.driver_mutation!(pop)
        end
        E_arr[r] = Jevo.get_energy(pop, emat)
    end
    return make_histogram(E_arr[5000:end])
end


# Store Metadata
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

E_results = SharedArray{Float64, 3}(length(κ_arr), length(l_arr), reps)
l_list = SharedArray{Float64, 3}(length(κ_arr), length(l_arr), reps)
kappa_list = SharedArray{Float64, 3}(length(κ_arr), length(l_arr), reps)

# Run simulations and enjoy speed
for κ in κ_arr
    @sync @distributed for l in l_arr
        x, y = Run(N, l, emat, F, κ, l_0, gap, steps)
    end
    println("Run $j done.")
end
df = DataFrame(gamma=[(E_results...)...], l=[(l_list...)...], kappa=[(kappa_list...)...])
CSV.write(date * "_results.csv", df)

