using DataFrames, CSV, Jevo, Plots, Statistics, Jedi

Jevo.default_gr!()

#=
df = CSV.read("../../outputs/2020_10_06_supp5_script.csv")
gdf = groupby(df, :rho)
cdf = combine(gdf, :l => mean)

df2 = CSV.read("../../outputs/2020_10_01supp5_script_results.csv")
gdf2 = groupby(df2, :rho)
cdf2 = combine(gdf2, :l => mean)
=#

df2 = CSV.read("../../outputs/2020_10_01supp5_script_results.csv")
gdf2 = groupby(df2, :rho)
cdf2 = combine(gdf2, :l => mean)

p1 = Jedi.histogram(data=Array{Int64, 1}(gdf[4].l))
p2 = scatter(cdf.rho, cdf.l_mean, label="Simulation")
scatter!(p2, [0, 0.1, 0.5, 1, 2], [15, 20, 25, 30, 35], label="Initial condition")
p3 = scatter(cdf2.rho, cdf2.l_mean, label="Simulation")
scatter!(p3, [0, 0.1, 0.5, 1, 2], [15, 15, 15, 15, 15], label="Initial condition", er)
plot(p1, p2, p3)