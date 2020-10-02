using Jedi, Plots, LambertW, Measures, LaTeXStrings, JLD

# Find figure folder
path = @__FILE__
figure_directory = splitpath(path)[1:end-3] |> x->joinpath(x..., "figures/")

# Set default plotting style
default_pyplot!()

# Set fitness parameters
N = 1000
f0 = 50 / 2N
fl = .2 / 2N
F = fermi_fitness(f0=f0, fl=fl)


function theoretical_msb(rho, l::Array{Int64, 1}=collect(8:40))
    Z = lambertw.(4/3 * N * f0 * l * exp(10) / (1 + rho)) ./2
    k_msb = -Z .+ (l .* (3/4))
    return k_msb
end


function length_rates(rho::Real, l::Array{Int, 1}=collect(8:40))
    k = theoretical_msb(rho, l.+1)
    
    s_plus_match = fitness.(k * 2, l .+ 1, F) .- fitness.(k .* 2, l, F)
    s_plus_mismatch = fitness.((k .+ 1) * 2, l .+ 1, F) .- fitness.(k .* 2, l, F)


    k = theoretical_msb(rho, l .+ 1)
    s_minus_match = fitness.(k .* 2, l, F) .- fitness.(k .* 2, l .+ 1, F)
    s_minus_mismatch = fitness.((k .- 1) * 2, l, F) .- fitness.(k .* 2, l .+ 1, F)
    
    up = kimura_prob.(((1/4 .* s_plus_match) .+ (3/4 .* s_plus_mismatch)), N)
    down = kimura_prob.(((k ./ (l .+ 1) .* s_minus_mismatch) .+ ((1 .- k ./ (l .+1 )) .* s_minus_match)), N)
    return up, down
end



l_opt = []
for rho in 0:0.01:2
    up, down = length_rates(rho, collect(8:60))
    p_l = zeros(length(up))
    p_l[1] = 0.1

    # detailed balance
    for i in 2:length(up)
        p_l[i] = p_l[i-1] * up[i-1] / down[i]
    end

    # normalize
    p_l /= sum(p_l)
    push!(l_opt, sum(p_l .* collect(8:60)))#argmax(p_l) + 7)
end


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

plot!(p1, collect(0.01:0.01:2) .+ 1, l_opt_load[2:end] .* sqrt(l_0), label=L"Load Minimization Rescaled by $\sqrt{l_0}$")

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

plot!(
    p2, 
    collect(0:0.01:2), 
    l_opt_load, 
    label="Load Minimization"
    )

l_0 = lambertw(4/3 * N * f0 * 10 * exp(10)) / 2

plot!(
    p2, 
    collect(0:0.01:2), 
    l_opt_load .* sqrt(l_0), 
    label=L"Load Minimization Rescaled by $\sqrt{l_0}$"
    )

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
    label=L"Load Minimization Rescaled by $\sqrt{l_0}$"
    )

savefig(p4, figure_directory * "supp5_l_opt_comparison_w_kappa.pdf")
plot!()