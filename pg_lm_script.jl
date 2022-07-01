using Random, Distributions, PolyaGammaSamplers, LinearAlgebra, PDMats, Plots, DataFrames;
include("src/log_sum_exp.jl")
include("src/softmax.jl")
include("src/eta_to_pi.jl")
include("src/calc_Mi.jl")
include("src/calc_kappa.jl")
#include("src/polyagamma.jl")
include("src/pg_lm.jl")


Random.seed!(2022);

N = 1000;
p = 2;
J = 3;
X = rand(Normal(0, 1), N, p-1);
X = hcat(ones(N), X);
size(X)

beta = rand(Normal(0, 1), p, J-1);
eta = X * beta;
size(eta)


eta_to_pi(X[1, ])
pi = map(eta_to_pi, eachrow(eta));
size(pi)
totals = [sum(v) for v in pi]

Y = Array{Union{Missing, Integer}}(undef, N, J)

Ni = rand(Poisson(500), N)
for i in 1:N
    Y[i, :] = rand(Multinomial(Ni[i], pi[i]), 1)
end

# add in some missing values
Y[2,:] .= missing

Y[4,:] .= missing

params = Dict{String, Int64}("n_adapt" => 100, "n_mcmc" => 100, "n_thin" => 1, "n_message" => 50)

out = pg_lm(Y, X, params);

mean(select(out, r"beta"), dims=1)
mean(out["beta"], dims=1)
beta

out |> @df plot(^(1:nrow(out)), cols(1:4))
out |> @df plot(^(1:nrow(out)), cols("beta[1, 1]"))


@df out plot(^(1:nrow(out)), :r"beta")







