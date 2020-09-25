using Plots, CSV, DataFrames, Jedi, Statistics, Measures, LaTeXStrings

# Find file path
path = @__FILE__
directory = splitdir(path)[1]

# Set default plotting style
default_gr!()

# Set fitness parameters
N = 1000
f0 = 50 / 2N
fl = 0.3 / 2N
F = fermi_fitness(f0=f0, fl=fl)

# Generate 2D grid
g_arr = 0:0.01:1 |> collect
l_arr = 8:40 |> collect
f_arr = zeros(Float64,  length(g_arr), length(l_arr))

# Compute fitness
for (i, g) in enumerate(g_arr)
    for (j, l) in enumerate(l_arr)
        f_arr[i, j] = fitness(2l * g, l, F)
    end
end

# Plot 2D landscape
p2D = contourf(
    l_arr, 
    g_arr, 
    f_arr, 
    color=:viridis, 
    xlabel=L"$l$", 
    ylabel=L"$\gamma$", 
    colorbarticks=([],[]), 
    size=(400,300),
    frame=true,
    fmt=:pdf
)

savefig(p2D, directory*"../../figures/2Dfitness.pdf")

# Compute exponential approximation
g = collect(0:0.01:1)
f_exp = f0 .* (1 .- exp.(2g.*10 .- (6/4 *10) .+ 10))

# Plot 1D Landscape
p1D = plot(
    0:0.01:1,
    f_arr[:, 3].+ fl*10,
    linewidth=2,
    ylim=(-f0 * 0.01, f0*1.01),
    yticks=([f0], [L"$f_0$"]),
    xlabel=L"$\gamma$",
    ylabel="Fitness",
    size=(400,300),
    label="Sigmoid",
    legend=(0.65, 0.8)
    )

plot!(
    p1D,
    0:0.01:1,
    f_exp,
    linewidth=2,
    label="Exponential"
    )

savefig(p2D, directory*"../../figures/1Dfitness.pdf")