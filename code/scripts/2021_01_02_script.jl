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
    n = 4
    N = 1000
    steps = 10^8
    reps = 200
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
        if (rand() < 1/10N) && (Jevo.get_energy(pop, emat)*l_0/length(pop.seqs)/gap < Jevo.γ_0(n))
            Jevo.l_substitution!(pop, emat, F)
	elseif (Jevo.get_energy(pop, emat)*l_0/length(pop.seqs)/gap > Jevo.γ_0(n))
            pop = Jevo.mono_pop(N=N, l=length(pop.seqs))
	    Jevo.initiate!(pop, emat)
	end
    end
    return Jevo.get_energy(pop, emat) * l_0/length(pop.seqs)/gap, length(pop.seqs)
end


# Store Metadata
open(date*"_METADATA.txt", "a") do io
    write(io, "gap=$gap\n")
    write(io, "l_0=$l_0\n")
    write(io, "f0=$f0\n")
    write(io, "fl=$fl\n")
    write(io, "kappa=$κ_arr\n")
    write(io, "n=$n\n")
    write(io, "N=$N\n")
    write(io, "steps=$steps\n")
    write(io, "reps=$reps")
end

E_results = SharedArray{Float64, 2}(length(κ_arr), reps)
l_list = SharedArray{Float64, 2}(length(κ_arr), reps)
kappa_list = SharedArray{Float64, 2}(length(κ_arr), reps)

# Run simulations and enjoy speed
@sync @distributed for j in 1:reps
    for (i1, κ) in enumerate(κ_arr)
        E, l= run(N, 150, emat, F, κ, l_0, gap, steps)
        E_results[i1, j] = E
        l_list[i1, j] = l
        kappa_list[i1, j] = κ
    end
    println("Run $j done.")
end
df = DataFrame(gamma=[(E_results...)...], l=[(l_list...)...], kappa=[(kappa_list...)...])
CSV.write(date * "_results.csv", df)

