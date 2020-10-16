### A Pluto.jl notebook ###
# v0.12.4

using Markdown
using InteractiveUtils

# ╔═╡ 35a1d126-0f21-11eb-1135-7d1e637e169c
using Revise

# ╔═╡ b7babfb0-0f12-11eb-12cf-b7fa0b773f0c
using DataFrames, CSV, Jevo, Plots, Statistics, Jedi, JLD, LaTeXStrings, LambertW

# ╔═╡ e0d93548-0f12-11eb-04e4-1b88ffc544e7
begin
	Jevo.default_gr!()
	gr(size=(600, 600))
end

# ╔═╡ 11c502fe-0f13-11eb-33f0-716ee8ec84f0
begin
	df = CSV.read("../../outputs/2020_10_06_supp5_script.csv")
	gdf = groupby(df, :rho)
	cdf = combine(:l => x -> (l_mean=mean(x), l_std=std(x)), gdf)
end

# ╔═╡ 20bc41a0-0f1d-11eb-053c-6f8910bcc881
parameter_dict = Jevo.parse_metadata("../../outputs/2020_10_06_supp5_script_METADATA.txt")

# ╔═╡ bfb48da0-0f4d-11eb-25c1-730df6671614
parameter_dict_2 = Jevo.parse_metadata("../../outputs/2020_10_01_script4_results_METADATA.txt")

# ╔═╡ 1e346566-0f13-11eb-11d2-e75056c08c1b
begin 
	df2 = CSV.read("../../outputs/2020_10_01supp5_script_results.csv")
	gdf2 = groupby(df2, :rho)
	cdf2 = combine(:l => x -> (l_mean=mean(x), l_std=std(x)), gdf2)
end

# ╔═╡ 28ce58ae-0f13-11eb-1fe9-df5f18a71c5e
begin
	p1 = Jedi.histogram(
		data=Array{Int64, 1}(gdf[4].l),
		line_kwargs=Dict{Any,Any}(:xlabel=>"l")
	)
	p2 = scatter(
		cdf.rho, 
		cdf.l_mean, 
		label="Simulation",
		yerror=cdf.l_std,
		xlabel="ρ",
		ylabel="l"
	)
	scatter!(
		p2, 
		[0, 0.1, 0.5, 1, 2], 
		[15, 20, 25, 30, 35], 
		label="Initial condition",
		legend=:topleft
	)
	p3 = scatter(
		cdf2.rho, 
		cdf2.l_mean, 
		label="Simulation",
		yerror=cdf2.l_std,
		xlabel=L"$\rho$",
		ylabel=L"$\ell$"
	)
	scatter!(
		p3, 
		[0, 0.1, 0.5, 1, 2], 
		[15, 15, 15, 15, 15], 
		label="Initial condition",
		legend=(0.6,0.5)
	)
	plot(p1, p2, p3)
end

# ╔═╡ 8c46f78e-0f14-11eb-31db-51d04837e276
begin
	N = 1000
	f0 = 50 / 2N
	fl = .2 / 2N
	F = Jevo.fermi_fitness(f0=f0, fl=fl)
	l_opt_dynamical = load("../figures/l_opt_dynamical.jld")["l_opt"]
	plot!(
		p2, 
		collect(0:0.01:2),
		l_opt_dynamical,
		label="Theory"
		)
	plot!(
		p3, 
		collect(0:0.01:2),
		l_opt_dynamical,
		label="Theory"
		)
	plot(p1, p2, p3)
end

# ╔═╡ ab41f29a-0f16-11eb-2e87-8db9e4c3684b
begin
	adapted_df = df[df.gamma .<= 2 .* (3 .* df.l ./ 4 .- 5), :] 
	adapted_gdf = groupby(adapted_df, :rho)
	adapted_cdf = combine(:l => x -> (l_mean=mean(x), l_std=std(x)), adapted_gdf)
	
	p4 = scatter(
		adapted_cdf.rho, 
		adapted_cdf.l_mean, 
		label="Functional sites",
		yerror=adapted_cdf.l_std,
		xlabel="ρ",
		ylabel="l",
		legend=:topleft
	)
	
	plot!(
		p4, 
		collect(0:0.01:2),
		l_opt_dynamical,
		label="Theory"
	)
	plot(p1, p2, p3, p4)
end


# ╔═╡ 725c3862-0f18-11eb-3681-6b72bf8c008b


# ╔═╡ Cell order:
# ╠═35a1d126-0f21-11eb-1135-7d1e637e169c
# ╠═b7babfb0-0f12-11eb-12cf-b7fa0b773f0c
# ╠═e0d93548-0f12-11eb-04e4-1b88ffc544e7
# ╠═11c502fe-0f13-11eb-33f0-716ee8ec84f0
# ╠═20bc41a0-0f1d-11eb-053c-6f8910bcc881
# ╠═bfb48da0-0f4d-11eb-25c1-730df6671614
# ╠═1e346566-0f13-11eb-11d2-e75056c08c1b
# ╠═28ce58ae-0f13-11eb-1fe9-df5f18a71c5e
# ╠═8c46f78e-0f14-11eb-31db-51d04837e276
# ╠═ab41f29a-0f16-11eb-2e87-8db9e4c3684b
# ╠═725c3862-0f18-11eb-3681-6b72bf8c008b
