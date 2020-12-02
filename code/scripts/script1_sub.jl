using CSV, DataFrames, Distributed, Dates, LinearAlgebra, Jevo, Distributions, DelimitedFiles
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
end

# Parameters
reps = 1000
steps = 1 * 10^6
N = 1000
nu = 0
emat = 2 * (ones(4, 4) - Matrix{Float64}(I, 4, 4))
rho_array = [0, 0.1, 0.5, 1]      # Driver mutation rates
l_array = [10, 15, 20]           # Binding site lengths
f0_array = [20, 50, 100] ./ 2N    # Fitness scales
fl = 0



Gamma_results = SharedArray{Float64, 4}(length(rho_array), length(l_array), length(f0_array), reps)
RHO = deepcopy(Gamma_results)
L = deepcopy(Gamma_results)
F0 = deepcopy(Gamma_results)


# Function to run one simulation
@everywhere function run(N, f0, fl, rho, nu, l_0, emat, steps)
    # Initiate population
    pop = Jevo.mono_pop(N=1000, l=l_0)
    Jevo.initiate!(pop, opt=true)
    f = Jevo.fermi_fitness(f0=f0, fl=fl)
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
    end
    return Jevo.get_energy(pop, emat)
end

# Write Metadata
open(date*"_script1_sub_METADATA.txt", "a") do io
    write(io, "N=$N\n")
    write(io, "f0=$f0_array\n")
    write(io, "fl=$fl\n")
    write(io, "repetitions=$reps\n")
    write(io, "steps=$steps\n")
    write(io, "rho=$rho_array\n")
    write(io, "nu=$nu\n")
    write(io, "l=$l_array\n")
end



# Run simulation once for Julia
@sync @distributed for j in 1:reps
    for i in 1:length(rho_array)
        for l in 1:length(l_array)
            for r in 1:length(f0_array)
                f = Jevo.fermi_fitness(f0=f0_array[r], l=l_array[l])
                Gamma_results[i, l, r, j] = run(N, f0_array[r], fl, rho_array[i], nu, l_array[l], emat, steps)
                RHO[i, l, r, j] = rho_array[i]
                L[i, l, r, j] = l_array[l]
                F0[i, l, r, j] = f0_array[r]
            end
        end
    end
    println("Run $j done.")
end




df = DataFrame(gamma=[(Gamma_results...)...], l=[(L...)...], rho=[(RHO...)...], f0=[(F0...)...])
CSV.write(date * "_script1_sub.csv", df)