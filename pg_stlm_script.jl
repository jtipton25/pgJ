using Random, Distributions, LinearAlgebra, PDMats, Plots;
using DataFrames, Distances, GaussianRandomFields;
using GaussianProcesses, ThreadsX, Dates;
# using PolyaGammaSamplers
include("src/log_sum_exp.jl")
include("src/softmax.jl")
include("src/eta_to_pi.jl")
include("src/calc_Mi.jl")
include("src/calc_kappa.jl")
include("src/polyagamma.jl")
include("src/update_tuning.jl")
include("src/pg_stlm.jl")

Threads.nthreads()



Random.seed!(2022);

N = 33^2;
#N = 100^2;
p = 2;
J = 6;
#n_time = 5
n_time = 30;
#locs = collect(product(
#     collect(range(0, stop=1, length=isqrt(N))),
#     collect(range(0, stop=1, length=isqrt(N)))))

locs = vec(collect(Iterators.product(
     range(0, stop=1, length=isqrt(N)),
     range(0, stop=1, length=isqrt(N)))))

locs = Matrix(reshape(reinterpret(Float64, locs), (2,:))')

D = pairwise(Distances.Euclidean(), locs, locs, dims=1)

tau2 = 4.5 * ones(J-1)
theta = 0.5 * ones(J-1)
rho = 0.9 * ones(J-1)

#cov_fun = GaussianRandomFields.Exponential(theta)

#Sigma = tau2 * apply(cov_fun, D)
# TODO: figure out how to do the GP kernels later, start by hand with exponential kernel
R = [exp.(-D / v) for v in theta]
R_chol = [cholesky(v) for v in R]
Sigma = [tau2[j] * R[j] for j in 1:(J-1)]
Sigma_chol = [sqrt(tau2[j]) * R_chol[j].U for j in 1:(J-1)]
Sigma_chol2 = [cholesky(v) for v in Sigma]

Sigma_chol3 = R_chol
for j in 1:(J-1)
    Sigma_chol3[j].U .*= sqrt(tau2[j])
end

all(isapprox.(Sigma_chol2[1].U, Sigma_chol3[1].U))

@time Sigma_inv = [inv(v) for v in Sigma_chol];
@time Sigma_inv2 = [inv(v) for v in Sigma_chol2];

j=1

@time rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol[j])), 1);  # expect an error
@time rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol2[j])), 1);
@time rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol3[j])), 1);

psi = Array{Float64}(undef, (N, J-1, n_time));

for j in 1:(J-1)
    # first psi
    psi[:, j, 1] = rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol3[j])), 1);
    for t in 2:n_time
        psi[:, j, t] = rand(MvNormal(rho[j] * psi[:, j, t-1], PDMat(Sigma[j], Sigma_chol3[j])), 1);
    end
end    

# setup the fixed effects
X = rand(Normal(0, 1), N, p-1);
X = hcat(ones(N), X);
size(X)

beta = rand(Normal(0, 1), p, J-1);

eta = Array{Float64}(undef, (N, J-1, n_time));
pi = Array{Float64}(undef, (N, J, n_time));

for t in 1:n_time
    eta[:, :, t] = X * beta + psi[:, :, t]
    pi[:, :, t] = reduce(hcat, map(eta_to_pi, eachrow(eta[:, :, t])))'
end

size(pi)
sum(pi[1, :, 1])
sum(pi[1, :, 10])
sum(pi[41, :, 10])


Y = Array{Union{Missing, Integer}}(undef, (N, J, n_time))

Ni = rand(Poisson(500), (N, n_time))
for i in 1:N
    for t in 1:n_time
        Y[i, :, t] = rand(Multinomial(Ni[i, t], pi[i, :, t]), 1)
    end
end

# add in some missing values
Y[2,:, 1] .= missing

Y[4,:, 3] .= missing

params = Dict{String, Int64}("n_adapt" => 100, "n_mcmc" => 100, "n_thin" => 1, "n_message" => 50, "mean_range" => 0, "sd_range" => 10, "alpha_tau" => 1, "beta_tau" => 1)

priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(10.0 .* ones(p)),
       	 "mean_range" => 0, "sd_range" => 10,
	 "alpha_tau" => 1, "beta_tau" => 1,
 	 "alpha_rho" => 1, "beta_rho" => 1)


tic = now()
out = pg_stlm(Y, X, locs, params, priors); # 32 minutes for 200 iterations -- can this be sped up more through parallelization?
# parallelization for omega running time of 20 minutes for 200 iterations
toc = now()

mean(select(out, r"beta"), dims=1)
mean(out["beta"], dims=1)
beta

out |> @df plot(^(1:nrow(out)), cols(1:4))
out |> @df plot(^(1:nrow(out)), cols("beta[1, 1]"))


@df out plot(^(1:nrow(out)), :r"beta")







