using CSV, DataFrames, Distributed, Dates, LinearAlgebra

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
    using Jedi
    using Distributions
    using DelimitedFiles
end

# Parameters
reps = 200
steps = 5 * 10^8
rho = [0, 0.1, 0.5, 1., 2]
l_0 = 15
N = 1000
nu = 1/N
emat = 2 * (ones(4, 4) - Matrix{Float64}(I, 4, 4))
f0 = 50/2N
fl = 0.3/2N


# Function to run one simulation
@everywhere function run(N, f0, fl, rho, nu, l_0, emat, steps)
    # Initiate population
    pop = mono_pop(N=1000, l=15)
    initiate!(pop, opt=true)
    rand_rho = rand(steps)
    rand_nu = rand(steps)
    f = fermi_fitness(f0=f0, fl=fl)
    Gamma = zeros(Float64, 1000)
    l_arr = zeros(Int64, 1000)
    for i in 1:steps
        bp_substitution!(pop, emat, f)
        if rand_rho[i] < rho/N
            driver_mutation!(pop)
        end
        if rand_nu[i] < 0.001
            l_substitution!(pop, emat, f)
        end
        # Recover lost sites
        if length(pop.seqs) < 7
            initiate!(pop, opt=true)
        end
        if i % 200000 == 0
            Gamma[i ÷ 200000] = get_energy(pop, emat)
            l_arr[i ÷ 200000] = length(pop.seqs)  
        end
    end
    return Gamma, l_arr
end
#=
# Run simulations on all available workers
@sync @distributed for j in 1:reps
    for r in 1:length(rho)
        E[r, j, :], L[r, j, :] = run(N, f0, fl, rho[r], nu, l_0, emat, steps)
        RHO[r, j, :] = rho[r] .* ones(Float64, 1000)
    end
    println("Run $j done.")
end

# Save results
df = DataFrame(gamma=[(E...)...], l=[(L...)...], rho=[(RHO...)...])
CSV.write(date*"_script4_results.csv", df)
=#
# Write Metadata
open(date*"_script4_results_METADATA.txt", "a") do io
    write(io, "N=$N\n")
    write(io, "f0=$f0\n")
    write(io, "fl=$fl\n")
    write(io, "repetitions=$reps\n")
    write(io, "steps=$steps\n")
    write(io, "rho=$rho\n")
    write(io, "nu=$nu\n")
    write(io, "l_0=$l_0\n")
end

open(date*"_script4_results.csv", "w") do io
   write(io, "rho\tl\tGamma\n")
end

# Run simulations on all available workers
@sync @distributed for j in 1:reps
    for r in 1:length(rho)
        E, L = run(N, f0, fl, rho[r], nu, l_0, emat, steps)
        RHO = rho[r] .* ones(Float64, 1000)
        open(date*"_script4_results.csv", "a") do io
            writedlm(io, [RHO L E])
        end
    end
    println("Run $j done.")
end