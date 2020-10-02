using CSV, DataFrames, Distributed, Dates


date = Dates.format(Dates.today(), "yyyy_mm_dd")

if length(ARGS) == 1
    addprocs(parse(Int64, ARGS[1]))
elseif length(ARGS) > 1
    throw(ArgumentError("Only one command line argument (cores)."))
end

# Load necessary packages on all workers
@everywhere  begin
    using Jedi
    using LinearAlgebra
    using Distributions
    using SharedArrays
    emat = 2 * (ones(4,4) - Matrix{Float64}(I, 4, 4))
    N = 1000
    f0 = 25/2N
end

# Driver mutation rates
rho = [0, 0.1, 0.5, 1., 2, 4]

# Lengths
l_arr = collect(8:40)

# Shared arrays to store results
E = SharedArray{Float64, 3}(length(rho), length(l_arr), 1000)
L = SharedArray{Float64, 3}(length(rho), length(l_arr), 1000)
RHO = SharedArray{Float64, 3}(length(rho), length(l_arr), 1000)


# Loop with most reps outside to maximize use of multiple cores
@sync @distributed for j in 1:1000
    for k in eachindex(l_arr)
        f = fermi_fitness(f0=f0, l=l_arr[k])
        pop = driver_trailer(N=1000, l=l_arr[k], L=l_arr[k])
        for r in eachindex(rho)
            # Reset population
            initiate_rand!(pop, 1, overwrite=true)
            # Precompute random numbers
            driver_muts = rand(500000)
            for i in 1:500000
                Jedi.bp_substitution!(pop, emat, f)
                if driver_muts[i] < rho[r]/N
                    driver_mutation!(pop)
                end
            end
            E[r, k, j] = get_energy(pop, emat)[1]
            L[r, k, j] = l_arr[k]
            RHO[r, k, j] = rho[r]
        end
    end
    println("Run $j done.")
end

# Store results in a DataFrame
df = DataFrame(gamma=[(E...)...], l=[(L...)...], rho=[(RHO...)...])
CSV.write(date*"_script1_sub_results.csv", df)