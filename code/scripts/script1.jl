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
    using LinearAlgebra
    emat = 2 * (ones(4,4) - Matrix{Float64}(I, 4, 4))
    generations = 2 * 10^6
end

# Simulation parameters
N = 1000
reps = 100                        # Repetitions
rho_array = [0, 0.1, 0.5, 1]      # Driver mutation rates
l_array = collect(5:30)           # Binding site lengths
f0_array = [20, 50, 100] ./ 2N    # Fitness scales
fl = 0



Gamma_results = SharedArray{Float64, 4}(length(rho_array), length(l_array), length(f0_array), reps)
RHO = deepcopy(Gamma_results)
L = deepcopy(Gamma_results)
F0 = deepcopy(Gamma_results)


@everywhere function run(rho, N, f, l0, generations)
    Gamma_arr = zeros(Float64, 26)
    pop = Jevo.driver_trailer(N=N, L=l0, l=l0)
    Jevo.initiate!(pop, 20)
    for j in 1:generations
        for m in 1:rand(Poisson(1), 1)[1]
            Jevo.mutation!(pop)
        end
        for m in 1:rand(Poisson(rho / N), 1)[1]
            Jevo.driver_mutation!(pop)
        end
        Jevo.sample_gen!(pop, f, emat; remove=true)
    end
    return sum(Jevo.get_energy(pop, emat) .* pop.freqs) / pop.N / l0
end


# Write Metadata
open(date*"_script1_METADATA.txt", "a") do io
    write(io, "N=$N\n")
    write(io, "f0=$f0_array\n")
    write(io, "fl=$fl\n")
    write(io, "repetitions=$reps\n")
    write(io, "generations=$generations\n")
    write(io, "rho=$rho_array\n")
    write(io, "l_0=$l_array\n")
end


# Run simulations 
@sync @distributed for j in 1:reps
    for i in 1:length(rho_array)
        for l in 1:length(l_array)
            for r in 1:length(f0_array)
                f = Jevo.fermi_fitness(f0=f0_array[r])
                Gamma_results[i, l, r, j] = run(rho_array[i], N, f, l_array[l], generations)
                RHO[i, l, r, j] = rho_array[i]
                L[i, l, r, j] = l_array[l]
                F0[i, l, r, j] = f0_array[r]
            end
        end
    end
    println("Run $j done.")
end

df = DataFrame(gamma=[(Gamma_results...)...], l=[(L...)...], rho=[(RHO...)...], f0=[(F0...)...])
CSV.write(date * "_script1.csv", df)
