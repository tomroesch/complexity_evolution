using CSV, DataFrames
@everywhere  begin
    using Jedi
    using LinearAlgebra
    using Distributions
    using SharedArrays
    emat = Matrix{Float64}(I, 4, 4)
    f = fermi_fitness(f0=100/20000)
end

results = SharedArray{Float64, 2}(4, 26)
@everywhere function run(rho)
    Gamma_arr = zeros(Float64, 26)
    for (i,l) in enumerate(5:30)
        for k in 1:100
            pop = driver_trailer(N=10000, L=30, l=l)
            initiate_rand!(pop, 20)
            mut_arr = rand(Poisson(1), 20000000)
            rho_arr = rand(Poisson(rho), 20000000)
            for j in 1:20000000
                for m in 1:mut_arr[j]
                    mutation!(pop)
                end
                for m in 1:rho_arr[j]
                    driver_mutation!(pop)
                end
                sample_gen!(pop, f, emat; remove=true)
            end
            Gamma_arr[i] += sum(get_energy(pop, emat) .* pop.freqs) / pop.N / l
        end
        println("Length $l done. Gamma= $(Gamma_arr[i]/100)")
    end
    return Gamma_arr
end

rho = [0, 0.00001, 0.0001, 0.0005]
@sync @distributed for r in 1:4
    results[r, :] = run(rho[r]) ./ 100
end

df_results = vcat(results'...)
df_l = vcat((collect(5:30) * ones(4)')...)
df_rho = vcat((rho * ones(26)')'...)

df = DataFrame(rho=df_rho, l=df_l, gamma=df_results)
CSV.write("script1_results.csv", df)
