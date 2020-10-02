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

# Plot ProductLog scaling with length
p1 = plot(
    8:0.1:30, 
    [1/2l*lambertw(4/3 * f0 * N * l * exp(10)) for l in 8:0.1:30], 
    label=L"$\dfrac{1}{2l}\times \mathrm{ProductLog(l)}$", 
    xlabel=L"$l$",
    ylabel=L"$\gamma - \gamma_0$",
    linewidth=2,
    legend=:topright,
    size=(300, 250),
    fmt=:pdf
)

plot!(
    p1,
    8:0.1:30, 
    [1/2l * lambertw(4/3 * f0 * N * 10 * exp(10)) for l in 8:0.1:30], 
    label=L"$\dfrac{1}{2l}\times \mathrm{const.}$", 
    linewidth=2,
    linestyle=:dash
)

savefig(p1, figure_directory * "supp4_prodlog_lengthscaling.pdf")

# Plot ProductLog scaling with non-equilibrium
y = [1/20 * lambertw(4/3 * f0*N * 10 * exp(10)/(1+rho)) for rho in 0.01:0.01:2]
p2 = plot(
    0.01:0.01:2, 
    [1/20 * lambertw(4/3 * f0*N * 10 * exp(10)/(1+rho)) for rho in 0.01:0.01:2],
    xlabel=L"$\kappa$",
    ylabel=L"$\gamma - \gamma_0$",
    linewidth=2,
    legend=:topright,
    size=(300, 250),
    fmt=:pdf,
    label=L"ProductLog($\kappa$)"
)
plot!(
    p2,
    0.01:0.01:2, 
    [y[1]/(1+rho)^(1/14) for rho in 0.01:0.01:2], 
    label=L"$1/(1+\kappa)^{1/14}$", 
    linewidth=2,
    linestyle=:dash
)

savefig(p2, figure_directory * "supp4_prodlog_kappascaling.pdf")


# Plot optimal length (including productlog scaling with kappa)
# Rescaled length cost/
l0 = [lambertw(4/3 * f0 * N * 10 * exp(10)) / 2 / (1+rho)^(1/14) for rho in 0:0.01:2] 
lambda = 2N .* fl .* l0
l_opt = [l0[i] *  sqrt((1+x)/lambda[i] * 3)  for (i, x) in enumerate(0:0.01:2)]
p3 = plot(
    0:0.01:2, 
    l_opt,
    size=(400,300),
    xlabel=L"$\kappa$",
    ylabel=L"$l_{opt}$",
    linewidth=2,
    fmt=:pdf
    )

savefig(p3, figure_directory * "supp4_opt_l_with_kappa.pdf")
dir = @__DIR__
save(dir * "/l_opt_from_load_w_kappa.jld", "l_opt", l_opt)



# Plot optimal length
# Rescaled length cost
l0 = lambertw(4/3 * f0 * N * 10 * exp(10))/2
lambda = 2N * fl * l0
l_opt = [l0 *  sqrt((1+x)/lambda * 3)  for x in 0:0.01:2]
p3 = plot(
    0:0.01:2, 
    l_opt,
    size=(400,300),
    xlabel=L"$\kappa$",
    ylabel=L"$l_{opt}$",
    linewidth=2,
    fmt=:pdf
    )

savefig(p3, figure_directory * "supp4_opt_l.pdf")
dir = @__DIR__
save(dir * "/l_opt_from_load.jld", "l_opt", l_opt)
plot!()
