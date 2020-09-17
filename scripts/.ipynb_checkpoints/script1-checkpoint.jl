
using CSV, DataFrames, Distributed

rho = [0, 0.000005, 0.00001, 0.00005, 0.0001, 0.0005, 0.001]
addprocs(length(rho))

@everywhere  begin
    using Jedi
    using LinearAlgebra
    using Distributions
    using SharedArrays
    emat = 2 * (ones(4,4) - Matrix{Float64}(I, 4, 4))
    N = 1000
    f = fermi_fitness(f0=100/2N)
end


results = SharedArray{Float64, 2}(length(rho), 26)
@everywhere function run(rho, N)
    Gamma_arr = zeros(Float64, 26)
    for (i,l) in enumerate(5:30)
        for k in 1:100
            pop = driver_trailer(N=N, L=l, l=l)
            initiate_rand!(pop, 20)
            mut_arr = rand(Poisson(1), 2000000)
            rho_arr = rand(Poisson(rho), 2000000)
            for j in 1:2000000
                for m in 1:mut_arr[j]
                    mutation!(pop)
                end
                for m in 1:rho_arr[j]
                    driver_mutation!(pop)
                end
                sample_gen!(pop, f, emat; remove=true)
            end
            Gamma_arr[i] += sum(get_energy(pop, emat) .* pop.freqs) / pop.N / l
            println("run $k done")
	end
        println("Length $l done. Gamma= $(Gamma_arr[i]/100)")
    end
    return Gamma_arr
end


@sync @distributed for r in 1:length(rho)
    results[r, :] = run(rho[r], N) ./ 100
end

df_results = vcat(results'...)
df_l = vcat((collect(5:30) * ones(length(rho))')...)
df_rho = vcat((rho * ones(26)')'...)

df = DataFrame(rho=df_rho, l=df_l, gamma=df_results)
CSV.write("script1_results.csv", df)
