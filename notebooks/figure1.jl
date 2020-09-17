using Plots, CSV, DataFrames, Jedi

plotlyjs()

# set plotting style
Jedi.default_plotlyjs!()

# Read data file
data = CSV.read("script1_results.csv")

# Find unique mutation rates
rho = data.rho |> unique

plot(xlabel="l", ylabel="γ")

for x in rho
    l = data[data.rho .== x, :l]
    gamma = data[data.rho .== x, :gamma]
    plot!(l, gamma, label=x, linewidth=2)
end
