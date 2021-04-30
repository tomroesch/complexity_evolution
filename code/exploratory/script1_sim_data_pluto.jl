### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ bcbc1fc2-13c2-11eb-0207-71c0fc51bb1e
using Jedi, CSV, DataFrames, Jevo, StatsBase, Plots

# ╔═╡ 979e9718-14bc-11eb-1562-ad4346a3a6a8
Jevo.default_plotlyjs!()

# ╔═╡ d4b3083e-13c2-11eb-32f7-f92b52232542
# Read metadata
parameters = Jevo.parse_metadata("../../outputs/2020_10_22_script1_METADATA.txt")

# ╔═╡ daff459e-13c3-11eb-0649-77b227aff60c
# Read simulation results
df = CSV.read("../../outputs/2020_10_22_script1.csv")

# ╔═╡ 0c189536-13c4-11eb-1f14-2306e2570d46
# Group by parameters
gdf = groupby(df, [:l, :rho, :f0])

# ╔═╡ 8214be90-1478-11eb-3f50-4d788c30b5c4
Jedi.histogram(data=df[(df.rho .== 0) .& (df.l .== 30) .& (df.f0 .== 0.025), :gamma] ./ 2)

# ╔═╡ 70c8e7cc-14bb-11eb-3d7f-f9bf9c35e891
function theoretical_dist(l, rho, f0, N)
	gamma = collect(0:0.01:1)
	F = Jevo.fermi_fitness(f0=f0, l=l)
	Q = exp.(-1/2 * l * (gamma .- 1/4*3) .^2 / (3/16)) .* exp.(2N / (1 + rho) * Jevo.fitness.(2 .* gamma .* l , F))
	return gamma, Q/sum(Q)*0.01
end

# ╔═╡ bd0fbde8-14bd-11eb-2831-47c0f2c17d94
begin 
	l=10
	f0=0.05
	rho=0
end

# ╔═╡ 5663e3f6-14bc-11eb-1a6b-73e553307a37
plot(theoretical_dist(l, rho, f0, 1000))

# ╔═╡ b2599208-14bc-11eb-1415-1b276b860117
begin
	p = fit(Histogram, df[(df.rho .== rho) .& (df.l .== l) .& (df.f0 .== f0), :gamma] ./ 2, 0:0.01:1)
	x = p.edges[1] |> collect
	x = [(x[i+1] + x[i]) / 2 for i in 1:length(x)-1]
	scatter!(x, p.weights ./ sum(p.weights) * 0.01)
end

# ╔═╡ Cell order:
# ╠═bcbc1fc2-13c2-11eb-0207-71c0fc51bb1e
# ╠═979e9718-14bc-11eb-1562-ad4346a3a6a8
# ╠═d4b3083e-13c2-11eb-32f7-f92b52232542
# ╠═daff459e-13c3-11eb-0649-77b227aff60c
# ╠═0c189536-13c4-11eb-1f14-2306e2570d46
# ╠═8214be90-1478-11eb-3f50-4d788c30b5c4
# ╠═70c8e7cc-14bb-11eb-3d7f-f9bf9c35e891
# ╠═bd0fbde8-14bd-11eb-2831-47c0f2c17d94
# ╠═5663e3f6-14bc-11eb-1a6b-73e553307a37
# ╠═b2599208-14bc-11eb-1415-1b276b860117
