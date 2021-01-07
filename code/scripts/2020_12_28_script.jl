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
    reps = 44*8  
    steps = 10^7
    rho = Float64[0, 0.1, 0.5, 1., 2, 3]
    l_0 = 10
    N = 100
    nu = 1/N^2

    f0_eps = [(50, 2.5), (100, 2), (200, 1.5), (500, 1), (300, 2)]

    n = 4
    
    # Length scale
    l0(n, F) = 1/F.epsilon * lambertw(2 * F.epsilon^2 * N * 10 * F.f0 * (n-1)/n^2* exp(5 * F.epsilon))
    l0(κ, l, n, F) = 1/F.epsilon * lambertw(2 * F.epsilon^2 * N * l * F.f0 * (n-1)/n^2 * exp(5 * F.epsilon)/(1 + κ))

    # Find length cost to set length in equilibrium (derived from genetic load modified by length dynamics)
    lambda(l_opt, n, F) = l0(n, F)^2 / l_opt^2 * n^2 / (n-1) * 1/F.epsilon
    lambda_new(l_opt, n, F) = l0(n, F)^3 / l_opt^2 * n^2 / (n-1) * 1/2
    lambda_new(l_opt, κ, l, n, F) = l0(κ, l, n, F)^3 / l_opt^2 * n^2 / (n-1) * 1/2

    #l_opt_static(κ, ϵ, n) = l0(ϵ, n) * sqrt(n^2 / ((n - 1) * ϵ * lambda(10)) * (1+κ))
    l_opt_dynamic(κ, n, F) = l0(n, F)^(3/2) * sqrt(n^2 / ((n - 1) * 2 * lambda_new(10, n, F)) * (1+κ))
end


# Function to run one simulation
@everywhere function run(N::Int64, rho::Float64, nu::Float64, l_0::Int64, steps::Int64, F::Jevo.fitness_functions)
    # Initiate population
    pop = Jevo.mono_pop(N=100, l=l_0)
    emat = F.epsilon * (ones(4, 4) - Matrix{Float64}(I, 4, 4))
    Jevo.initiate!(pop, emat)
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
            Jevo.initiate!(pop, emat)
        end
    end 
    return Jevo.get_energy(pop, emat), length(pop.seqs)
end

for (_f0, ϵ) in f0_eps
    println("Running parameters f0=$(_f0) and epsilon=$ϵ:")
    F = Jevo.fermi_fitness()
    F.f0 = _f0 / 2N
    F.epsilon = ϵ
    F.fl = lambda_new(10, n, F)/(2N*l0(n, F))
    # Write Metadata
    open(date*"sub_$(_f0)_$(ϵ)_METADATA.txt", "a") do io
        write(io, "N=$N\n")
        write(io, "f0=$(F.f0)\n")
        write(io, "repetitions=$reps\n")
        write(io, "steps=$steps\n")
        write(io, "rho=$rho\n")
        write(io, "nu=$nu\n")
        write(io, "l_0=$l_0\n")
        write(io, "epsilon=$ϵ\n")
	write(io, "n=$n")
    end
    E_results = SharedArray{Float64, 2}(length(rho), reps)
    l_results = SharedArray{Float64, 2}(length(rho), reps)
    rho_list = SharedArray{Float64, 2}(length(rho), reps)
    # Run simulations and enjoy speed
    @sync @distributed for j in 1:reps
        for r in 1:length(rho)
            E, L = run(N, rho[r], nu, l_0, steps, F)
            E_results[r, j] = E
            l_results[r, j] = L
            rho_list[r, j] = rho[r]
        end
        println("Run $j done.")
    end
    df = DataFrame(gamma=[(E_results...)...], l=[(l_results...)...], rho=[(rho_list...)...])
    CSV.write(date * "sub_$(_f0)_$(ϵ)_results.csv", df)
end



