using Distributed, DataFrames, CSV
addprocs(33)

@everywhere  begin
    using Jedi
    using LinearAlgebra
    using Distributions
    using SharedArrays
    emat = 2 * (ones(4, 4) - Matrix{Float64}(I, 4, 4))
    N = 1000
    f0 = 20/2N
end
rho = [0, 0.1, 0.5, 1., 2]
E = SharedArray{Float64, 3}(length(rho),33, 1000)
L = SharedArray{Float64, 3}(length(rho),33, 1000)
RHO = SharedArray{Float64, 3}(length(rho),33, 1000)

l_arr = collect(8:40)

@sync @distributed for k in 1:33
    for r in 1:length(rho)
        f = fermi_fitness(f0=f0, l=l_arr[k])
        for j in 1:1000
            pop = driver_trailer(N=1000, l=l_arr[k], L=l_arr[k])
            initiate_rand!(pop, 1)
            for i in 1:500000
                Jedi.bp_substitution!(pop, emat, f)
                if rand() < rho[r]/N
                    driver_mutation!(pop)
                end
            end
            E[r, k, j] = get_energy(pop, emat)[1]
            L[r, k, j] = l_arr[k]
            RHO[r, k, j] = rho[r]
        end
        println("Rho $(rho[r]), length $(l_arr[k]) done.")
    end
end

df = DataFrame(gamma=[(E...)...], l=[(L...)...], rho=[(RHO...)...])
CSV.write("script3_results.csv", df)
