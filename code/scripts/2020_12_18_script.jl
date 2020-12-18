using CSV, DataFrames, Distributed, Dates, LinearAlgebra, Distributions, DelimitedFiles, LambertW, SharedArrays

# Custom package
using Jevo

#=
In this script we are testing a variety of parameters to identify the range of
non-equilibrium ratios to which theoretical prediction are accurate. 
=#


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
    using LambertW
    using LinearAlgebra
end

# Parameters
@everywhere begin
    reps = 2  
    steps = 2
    rho = [0, 0.1, 0.5, 1., 2, 3]
    l_0 = 10
    N = 100
    nu = 1/N^2

    f0_eps = [(50, 2.5), (100, 2), (200, 1.5), (500, 1)]

    n = 4
    
    # Length scale
    l0(ϵ, n, f0=f0) = 1/ϵ * lambertw(2 * ϵ^2 * N * 10 * f0 * (n-1)/n^2* exp(5 * ϵ))
    l0(κ, l, ϵ, n, f0=f0) = 1/ϵ * lambertw(2 * ϵ^2 * N * l * f0 * (n-1)/n^2 * exp(5 * ϵ)/(1 + κ))

    # Find length cost to set length in equilibrium (derived from genetic load)
    #lambda(l_opt) = l0()^2/l_opt^2 * n^2 / (n-1) * 1/ϵ

    # Find length cost to set length in equilibrium (derived from genetic load modified by length dynamics)
    lambda_new(l_opt, ϵ, n, f0=f0) = l0(ϵ, n, f0)^3 / l_opt^2 * n^2 / (n-1) * 1/2

    #l_opt_static(κ, ϵ, n) = l0(ϵ, n) * sqrt(n^2 / ((n - 1) * ϵ * lambda(10)) * (1+κ))
    l_opt_dynamic(κ, ϵ, n, f0=f0) = l0(ϵ, n, f0)^(3/2) * sqrt(n^2 / ((n - 1) * 2 * lambda_new(10, ϵ, n, f0)) * (1+κ))
end


# Function to run one simulation
@everywhere function run(N, rho, nu, l_0, steps, f0, ϵ)
    # Initiate population
    pop = Jevo.mono_pop(N=100, l=l_0)
    Jevo.initiate!(pop)
    emat = ϵ * (ones(4, 4) - Matrix{Float64}(I, 4, 4))
    F = Jevo.fermi_fitness(f0=f0, fl=lambda_new(10, ϵ, n, f0)/(2N*l0(ϵ, n, f0)), epsilon=ϵ)
    for i in 1:steps
        Jevo.bp_substitution!(pop, emat, F)
        if rand() < rho/N
            Jevo.driver_mutation!(pop)
        end
        if rand() < nu
            Jevo.l_substitution!(pop, emat, F)
        end
        # Recover lost sites
        if length(pop.seqs) < 6
            Jevo.initiate!(pop)
        end
    end 
    return Jevo.get_energy(pop, emat), length(pop.seqs)
end

for (_f0, ϵ) in f0_eps
    println("Running parameters f0=$_f0 and epsilon=$ϵ:")
    f0 = _f0 / 2N
    # Write Metadata
    open(date*"sub_$(f0)_$(ϵ)_METADATA.txt", "a") do io
        write(io, "N=$N\n")
        write(io, "f0=$f0\n")
        write(io, "repetitions=$reps\n")
        write(io, "steps=$steps\n")
        write(io, "rho=$rho\n")
        write(io, "nu=$nu\n")
        write(io, "l_0=$l_0\n")
        write(io, "epsilon=$ϵ")
    end
    E_results = SharedArray{Float64, 2}(length(rho), reps)
    l_results = SharedArray{Float64, 2}(length(rho), reps)
    rho_list = SharedArray{Float64, 2}(length(rho), reps)
    # Run simulations and enjoy speed
    @sync @distributed for j in 1:reps
        for r in 1:length(rho)
            E, L = run(N, rho[r], nu, l_0, steps, f0, ϵ)
            E_results[r, j] = E
            l_results[r, j] = L
            rho_list[r, j] = rho[r]
        end
        println("Run $j done.")
    end
    df = DataFrame(gamma=[(E_results...)...], l=[(l_results...)...], rho=[(rho_list...)...])
    CSV.write(date * "sub_$(f0)_$(ϵ)_results.csv", df)
end



