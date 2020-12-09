using CSV, DataFrames, Distributed, Dates, LinearAlgebra, Jevo, Distributions, DelimitedFiles, LambertW
using SharedArrays, TimerOutputs
# Get date to append to output file
date = Dates.format(Dates.today(), "yyyy_mm_dd")

# Get number of workers
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
    using LambertW
    using LinearAlgebra
end

# Parameters
@everywhere begin
    reps = 100  
    steps = 2 * 10^6
    rho = [0, 0.1, 0.5, 1., 2, 5]
    l_0 = 10
    N = 100
    nu = 1/N

    f0 = 200/2N
    ϵ = 2
    n = 4
    emat = ϵ * (ones(4, 4) - Matrix{Float64}(I, 4, 4))

    l0_kappa(kappa, l) = 1/2 * lambertw(2 * ϵ^2 * N * l * f0 * (n-1)/n^2 * exp(10)/(1+kappa))
    fl(l_opt) = l0_kappa(0, 10)/l_opt^2 * n^2 / (n-1) * 1/ϵ

    
end


E_results = SharedArray{Float64, 2}(length(rho), reps)
l_results = SharedArray{Float64, 2}(length(rho), reps)
rho_list = SharedArray{Float64, 2}(length(rho), reps)


# Function to run one simulation
@everywhere function run(N, rho, nu, l_0, emat, steps)
    # Initiate population
    pop = Jevo.mono_pop(N=100, l=l_0)
    Jevo.initiate!(pop, opt=true)
    F = Jevo.fermi_fitness(f0=f0, fl=fl(10)/2N)
    for i in 1:steps
        Jevo.bp_substitution!(pop, emat, F)
        if rand() < rho/N
            Jevo.driver_mutation!(pop)
        end
        if rand() < nu
            Jevo.l_substitution!(pop, emat, F)
        end
        # Recover lost sites
        if length(pop.seqs) < 7
            Jevo.initiate!(pop, opt=true)
        end
    end 
    return Jevo.get_energy(pop, emat), length(pop.seqs)
end

# Write Metadata
open(date*"_supp5_script_METADATA.txt", "a") do io
    write(io, "N=$N\n")
    write(io, "f0=$f0\n")
    write(io, "fl=$fl\n")
    write(io, "repetitions=$reps\n")
    write(io, "steps=$steps\n")
    write(io, "rho=$rho\n")
    write(io, "nu=$nu\n")
    write(io, "l_0=$l_0\n")
end


# Run simulations and enjoy speed
@sync @distributed for j in 1:reps
    for r in 1:length(rho)
        E, L = run(N, rho[r], nu, l_0, emat, steps)
        E_results[r, j] = E
        l_results[r, j] = L
        rho_list[r, j] = rho[r]
    end
    println("Run $j done.")
end


df = DataFrame(gamma=[(E_results...)...], l=[(l_results...)...], rho=[(rho_list...)...])
CSV.write(date * "_supp5_script.csv", df)
