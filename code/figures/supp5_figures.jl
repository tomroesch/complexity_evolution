using Jedi, Plots, LambertW, Measures, LaTeXStrings, JLD, CSV, DataFrames, Statistics, Jevo

# Find figure folder
path = @__FILE__
figure_directory = splitpath(path)[1:end-3] |> x->joinpath(x..., "figures/")
dir = @__DIR__

# 
#df = CSV.read("outputs/2020_10_01supp5_script_results.csv")
#gdf = groupby(df, :rho)
#cdf = combine(gdf, :l => mean)

# Set default plotting style
Jevo.default_pyplot!()


# Compute mutation selection balance
function theoretical_msb(rho, l::Array{Int64, 1}=collect(8:40))
    Z = lambertw.(4/3 * N * f0 * l * exp(10) / (1 + rho)) ./2
    k_msb = -Z .+ (l .* (3/4))
    return k_msb
end

# Compute dynamical substitution rates for length mutations
function length_rates(rho::Real, F::Jevo.fitness_functions, l::Array{Int, 1}=collect(8:40))
    k = theoretical_msb(rho, l.+1)
    
    s_plus_match = Jevo.fitness.(k * 2, l .+ 1, F) .- Jevo.fitness.(k .* 2, l, F)
    s_plus_mismatch = Jevo.fitness.((k .+ 1) * 2, l .+ 1, F) .- Jevo.fitness.(k .* 2, l, F)


    k = theoretical_msb(rho, l .+ 1)
    s_minus_match = Jevo.fitness.(k .* 2, l, F) .- Jevo.fitness.(k .* 2, l .+ 1, F)
    s_minus_mismatch = Jevo.fitness.((k .- 1) * 2, l, F) .- Jevo.fitness.(k .* 2, l .+ 1, F)
    
    up = Jevo.kimura_prob.(((1/4 .* s_plus_match) .+ (3/4 .* s_plus_mismatch)), N)
    down = Jevo.kimura_prob.(((k ./ (l .+ 1) .* s_minus_mismatch) .+ ((1 .- k ./ (l .+1 )) .* s_minus_match)), N)
    return up, down
end


# Compute optimal length as mean of length distribution, which is computed from the rates
function dynamical_optimum(F::Jevo.fitness_functions)
    l_opt = []
    for rho in 0:0.01:2
        up, down = length_rates(rho, F, collect(8:60))
        p_l = zeros(length(up))
        p_l[1] = 0.1

        # detailed balance
        for i in 2:length(up)
            p_l[i] = p_l[i-1] * up[i-1] / down[i]
        end

        # normalize
        p_l /= sum(p_l)
        #push!(l_opt, sum(p_l .* collect(8:60)))#argmax(p_l) + 7)
        push!(l_opt, argmax(p_l) + 7)
    end
    return l_opt
end

# Set fitness parameters
N = 1000
f0 = 0.1*2000 / 2N
fl = .2 / 2N
F = Jevo.fermi_fitness(f0=f0, fl=fl)

# Compute optimal length
l_opt = dynamical_optimum(F)

# Compute optimal load from minimizing genetic load
l0 = lambertw(4/3 * f0 * N * 10 * exp(10))/2
lambda(fl) = 2N * fl * l0
l_opt_load = [l0 *  sqrt((1+x)/lambda(fl) * 3)  for x in 0:0.01:2]

# Plot log-log-scale
p1 = plot(
    collect(0.01:0.01:2) .+ 1, 
    l_opt[2:end], 
    xlabel=L"$1+\kappa$", 
    ylabel=L"$l_{opt}$", 
    label="Dynamical", 
    size=(500,400),
    legend=:topleft
    )

l_opt_load = load(dir * "/l_opt_from_load.jld")["l_opt"]

plot!(
    p1, collect(0.01:0.01:2) .+ 1, 
    l_opt_load[2:end], 
    label="Load Minimization", 
    xscale=:ln, 
    yscale=:ln
    )
l_0 = lambertw(4/3 * N * f0 * 10 * exp(10)) / 2

plot!(
    p1, 
    collect(0.01:0.01:2) .+ 1, 
    l_opt_load[2:end] .* sqrt(l_0), 
    label=L"Load Minimization Rescaled by $\sqrt{l_0}$"
    )


savefig(p1, figure_directory * "supp5_l_opt_comparison_loglog.pdf")
plot!()


# Plot normal scale
p2 = plot(
    collect(0:0.01:2), 
    l_opt, 
    xlabel=L"$\kappa$", 
    ylabel=L"$l_{opt}$", 
    label="Dynamical", 
    size=(500,400),
    legend=:topleft
    )

l_opt_load = load(dir * "/l_opt_from_load.jld")["l_opt"]

l_0 = lambertw(4/3 * N * f0 * 10 * exp(10)) / 2

plot!(
    p2, 
    collect(0:0.01:2), 
    l_opt_load .* sqrt(l_0), 
    label="Rescaled Load Minimization"
    )

plot!(
    p2, 
    collect(0:0.01:2), 
    l_opt_load, 
    label="Load Minimization"
    )
#=
scatter!(
    p2,
    cdf.rho,
    cdf.l_mean
)    
=#
savefig(p2, figure_directory * "supp5_l_opt_comparison.pdf")



# Redo plots with l0 scaling with kappa
# Plot log-log-scale
p3 = plot(
    collect(0.01:0.01:2) .+ 1, 
    l_opt[2:end], 
    xlabel=L"$1+\kappa$", 
    ylabel=L"$l_{opt}$", 
    label="Dynamical", 
    size=(500,400),
    legend=:topleft
    )

l_opt_load = load(dir * "/l_opt_from_load_w_kappa.jld")["l_opt"]

plot!(
    p3, collect(0.01:0.01:2) .+ 1, 
    l_opt_load[2:end], 
    label="Load Minimization", 
    xscale=:ln, 
    yscale=:ln
    )
l_0 = [lambertw(4/3 * f0 * N * 10 * exp(10))/2 / (1+rho)^(1/14) for rho in 0:0.01:2]

plot!(
    p3, 
    collect(0.01:0.01:2) .+ 1, 
    l_opt_load[2:end] .* sqrt.(l_0[2:end]), 
    label=L"Load Minimization Rescaled by $\sqrt{l_0}$"
    )

savefig(p3, figure_directory * "supp5_l_opt_comparison_loglog_w_kappa.pdf")
plot!()


# Plot normal scale
p4 = plot(
    collect(0:0.01:2), 
    l_opt, 
    xlabel=L"$\kappa$", 
    ylabel=L"$l_{opt}$", 
    label="Dynamical", 
    size=(500,400),
    legend=:topleft
    )

l_opt_load_kappa = load(dir * "/l_opt_from_load_w_kappa.jld")["l_opt"]

plot!(
    p4, 
    collect(0:0.01:2), 
    l_opt_load_kappa, 
    label="Load Minimization"
    )


plot!(
    p4, 
    collect(0:0.01:2), 
    l_opt_load .* sqrt.(l_0), 
    label="Rescaled Load Minimization"
    )

savefig(p4, figure_directory * "supp5_l_opt_comparison_w_kappa.pdf")
plot!()


# Set fitness parameters

fl = 1.4 / 2N
F = Jevo.fermi_fitness(f0=f0, fl=fl)

l_opt_load = [l0 *  sqrt((1+x)/lambda(fl) * 3)  for x in 0:0.01:2]

# Compute optimal length
l_opt = dynamical_optimum(F)

# Plot normal scale
p5 = plot(
    collect(0:0.01:2), 
    l_opt, 
    xlabel=L"$\kappa$", 
    ylabel=L"$l_{opt}$", 
    label="Dynamical", 
    size=(500,400),
    legend=:topleft,
    #linestyle=:dash
    )

l_0 = lambertw(4/3 * N * f0 * 10 * exp(10)) / 2

plot!(
    p5, 
    collect(0:0.01:2), 
    l_opt_load .* sqrt(l_0), 
    label="Rescaled Load Minimization",
    #linestyle=:dash
    )


savefig(p5, figure_directory * "supp5_l_opt_comparison_high_cost.pdf")