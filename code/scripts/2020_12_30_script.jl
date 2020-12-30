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
    fl = .7l_0
    f0 = 20l_0
    κ_arr = 0:2:20
    l_arr = 70:10:200
    n = 4
    N = 1000
    steps = 10^6
    reps = 100
    F = Jevo.num_fermi(n, l_0, gap, f0/2N, fl/2N)
    emat = gap/l_0 * (ones(n, n) - Matrix{Float64}(I, n, n))
end


# Run one rep
@everywhere function run(N, l, emat, F, κ, l_0, gap, steps)
    pop = Jevo.mono_pop(N=N, l=l)
    Jevo.initiate!(pop, emat)
    for i in 1:steps
        Jevo.bp_substitution!(pop, emat, F)
        if rand() < κ/N
            Jevo.driver_mutation!(pop)
        end
    end
    return Jevo.get_energy(pop, emat) * l_0/l/gap
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
    write(io, "reps=$reps")
end

E_results = SharedArray{Float64, 3}(length(κ_arr), length(l_arr), reps)
l_list = SharedArray{Float64, 3}(length(κ_arr), length(l_arr), reps)
kappa_list = SharedArray{Float64, 3}(length(κ_arr), length(l_arr), reps)

# Run simulations and enjoy speed
@sync @distributed for j in 1:reps
    for (i1, κ) in enumerate(κ_arr)
        for (i2, l) in enumerate(l_arr)
            E = run(N, l, emat, F, κ, l_0, gap, steps)
            E_results[i1, i2, j] = E
            l_list[i1, i2, j] = l
            rho_list[i1, i2, j] = κ
        end
    end
    println("Run $j done.")
end
df = DataFrame(gamma=[(E_results...)...], l=[(l_results...)...], rho=[(rho_list...)...])
CSV.write(date * "_results.csv", df)

