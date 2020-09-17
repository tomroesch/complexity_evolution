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
    using SharedArrays
end

# Parameters
reps = 200
steps = 10^8
rho = [0, 0.1, 0.5, 1., 2]
l_0 = 15
N = 1000
nu = 1/N
emat = 2 * (ones(4, 4) - Matrix{Float64}(I, 4, 4))
f0 = 25/2N
fl = 0.25/2N

# Arrays for results
E = SharedArray{Float64, 2}(length(rho), reps)
L = SharedArray{Float64, 2}(length(rho), reps)
RHO = SharedArray{Float64, 2}(length(rho), reps)

# Function to run one simulation
@everywhere function run(N, f0, fl, rho, nu, l_0, emat, steps)
    # Initiate population
    pop = driver_trailer_l(N=N, l_0=l_0, L=50)
    initiate_opt!(pop)
    # Pregenerate random numbers
    rand_rho = rand(steps)
    rand_nu = rand(steps)
    # Initiate fitness landscape
    f = fermi_fitness(f0=f0, fl=fl)

    for i in 1:steps
        bp_substitution!(pop, emat, f)
        if rand_rho[i] < rho/N
            driver_mutation!(pop)
        end
        if rand_nu[i] < nu
            l_substitution!(pop, emat, f)
        end
        # Recover lost sites
        if pop.l[1] < 7
            initiate_opt!(pop)
        end
    end
    Gamma = get_energy(pop, emat)[1]
    l_arr = pop.l[1]
    return Gamma, l_arr
end

# Run simulations on all available workers
@sync @distributed for j in 1:reps
    for r in 1:length(rho)
        E[r, j], L[r, j] = run(N, f0, fl, rho[r], nu, l_0, emat, steps)
        RHO[r, j] = rho[r]
    end
    println("Run $j done.")
end

# Save results
df = DataFrame(gamma=[(E...)...], l=[(L...)...], rho=[(RHO...)...])
CSV.write(date*"_script2_results.csv", df)

# Write Metadata
open(date*"_script2_results_METADATA.txt", "a") do io
    write(io, "N=$N\n")
    write(io, "f0=$f0\n")
    write(io, "fl=$fl\n")
    write(io, "repetitions=$reps\n")
    write(io, "steps=$steps\n")
    write(io, "rho=$rho\n")
    write(io, "nu=$nu\n")
    write(io, "l_0=$l_0\n")
end
