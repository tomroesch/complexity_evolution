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
    steps = 1 * 10^6
    N = 100
    nu = 1/N
    ϵ = 2
    n = 4
    emat = ϵ * (ones(4, 4) - Matrix{Float64}(I, 4, 4))
    rho_array = [0, 0.1, 0.5, 1, 2, 5]
    l_array = [10]
    f0 = 200 / 2N 
end

rescue = true


@everywhere l0_kappa(kappa, l, ϵ=2, n=4) = 1/2 * lambertw(2 * ϵ^2 * N * l * f0 * (n-1)/n^2 * exp(10)/(1+kappa))
@everywhere fl(l_opt) = l0_kappa(0, 10)/l_opt^2 * n^2 / (n-1) * 1/ϵ



Gamma_results = SharedArray{Float64, 4}(length(rho_array), length(l_array), 1, reps)
RHO = deepcopy(Gamma_results)
L = deepcopy(Gamma_results)
F0 = deepcopy(Gamma_results)


# Function to run one simulation
@everywhere function run(N, f0, fl, rho, nu, l_0, emat, steps, rescue)
    # Initiate population
    pop = Jevo.mono_pop(N=N, l=l_0)
    Jevo.initiate!(pop, opt=true)
    f = Jevo.fermi_fitness(f0=f0, fl=fl(10)/2N)

    for i in 1:steps
        Jevo.bp_substitution!(pop, emat, f)
        if rand() < rho/N
            Jevo.driver_mutation!(pop)
        end
        if rand() < nu
            Jevo.l_substitution!(pop, emat, f)
        end
        # Recover lost sites
        if length(pop.seqs) < 7
            Jevo.initiate!(pop, opt=true)
        end
        if rescue && (Jevo.get_energy(pop, emat) > 3/4*l_0 - 5/2)
            Jevo.initiate!(pop, opt=true)
        end
    end
    return Jevo.get_energy(pop, emat), length(pop.seqs)
end

# Write Metadata
open(date*"_script1_sub_METADATA.txt", "a") do io
    write(io, "N=$N\n")
    write(io, "f0=$f0\n")
    write(io, "fl=$fl\n")
    write(io, "repetitions=$reps\n")
    write(io, "steps=$steps\n")
    write(io, "rho=$rho_array\n")
    write(io, "nu=$nu\n")
    write(io, "l=$l_array\n")
    write(io, "rescue=$rescue\n")
end



# Run simulation once for Julia
@sync @distributed for j in 1:reps
    for i in 1:length(rho_array)
        for l in 1:length(l_array) 
            for r in 1:1
                f = Jevo.fermi_fitness(f0=f0, l=l_array[l])
                Gamma_results[i, l, r, j], L[i, l, r, j] = run(N, f0, fl, rho_array[i], nu, l_array[l], emat, steps, rescue)
                RHO[i, l, r, j] = rho_array[i]
                F0[i, l, r, j] = f0
            end
        end
    end
    println("Run $j done.")
end




df = DataFrame(gamma=[(Gamma_results...)...], l=[(L...)...], rho=[(RHO...)...], f0=[(F0...)...])
CSV.write(date * "_script1_sub.csv", df)