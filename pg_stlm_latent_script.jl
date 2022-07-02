# bash command to run script is 
# julia -t 32 pg_stlm_latent_script.jl > latent_out.txt &
using Random, Distributions, LinearAlgebra, PDMats, Plots;
using DataFrames, Distances, GaussianRandomFields;
#using GaussianProcesses;
using ThreadsX, Dates;
using JLD2, FileIO;
using StatsBase;
# using PolyaGammaSamplers
include("src/log_sum_exp.jl")
include("src/softmax.jl")
include("src/eta_to_pi.jl")
include("src/calc_Mi.jl")
include("src/calc_kappa.jl")
#include("src/polyagamma.jl")
include("src/update_tuning.jl")
include("src/pg_stlm_latent.jl")

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
     range(0, stop=1, length=isqrt(N)))));

locs = Matrix(reshape(reinterpret(Float64, locs), (2,:))');

D = pairwise(Distances.Euclidean(), locs, locs, dims=1);

I = Diagonal(ones(N));
sigma = 2.2 * ones(J-1);
tau = 4.5 * ones(J-1);
theta = 0.5 * ones(J-1);
rho = 0.9 * ones(J-1);

#cov_fun = GaussianRandomFields.Exponential(theta)

#Sigma = tau * apply(cov_fun, D)
# TODO: figure out how to do the GP kernels later, start by hand with exponential kernel
R = [exp.(-D / v) for v in theta];
R_chol = [cholesky(v) for v in R];
Sigma = [tau[j]^2 * R[j] for j in 1:(J-1)];
Sigma_chol = [cholesky(v) for v in Sigma];

Sigma_inv = [inv(v) for v in Sigma_chol];

psi = Array{Float64}(undef, (N, J-1, n_time));

for j in 1:(J-1)
    # first psi
    psi[:, j, 1] = rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol[j])), 1);
    for t in 2:n_time
        psi[:, j, t] = rand(MvNormal(rho[j] * psi[:, j, t-1], PDMat(Sigma[j], Sigma_chol[j])), 1);
    end
end    

# setup the fixed effects
X = rand(Normal(0, 1), N, p-1);
X = hcat(ones(N), X);
size(X);

beta = rand(Normal(0, 1), p, J-1);

eta = Array{Float64}(undef, (N, J-1, n_time));
pi = Array{Float64}(undef, (N, J, n_time));

for t in 1:n_time
    eta[:, :, t] = reduce(hcat, [X * beta[:, j] + psi[:, j, t] + rand(Normal(0, sigma[j]), N) for j in 1:(J-1)]);
    pi[:, :, t] = reduce(hcat, map(eta_to_pi, eachrow(eta[:, :, t])))';
end

size(pi)
sum(pi[1, :, 1])
sum(pi[1, :, 10])
sum(pi[41, :, 10])


Y = Array{Union{Missing, Integer}}(undef, (N, J, n_time));

Ni = rand(Poisson(500), (N, n_time));
for i in 1:N
    for t in 1:n_time
        Y[i, :, t] = rand(Multinomial(Ni[i, t], pi[i, :, t]), 1);
    end
end

# add in some missing values
missing_idx = StatsBase.sample([false, true], ProbabilityWeights([0.8, 0.2]), (N, n_time));
for i in 1:N
    for t in 1:n_time
        Y[i, :, t] .= missing;
    end
end


dat_sim = Dict{String, Any}("N" => N, "Y" => Y, "X" => X, "Ni" => Ni, "missing_idx" => missing_idx,
            "beta" => beta, "p" => p, "J" => J, "n_time" => n_time,
            "locs" => locs, "sigma" => sigma, "tau" => tau, "theta" => theta,
            "sigma" => sigma, "rho" => rho, "psi" => psi, "eta" => eta, "pi" =>pi);
save("output/latent_sim_data.jld", "data", dat_sim);            


params = Dict{String, Int64}("n_adapt" => 2000, "n_mcmc" => 5000, "n_thin" => 5, "n_message" => 50, "mean_range" => 0, "sd_range" => 10, "alpha_tau" => 1, "beta_tau" => 1);

priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(10.0 .* ones(p)),
       	"mean_range" => 0, "sd_range" => 10,
        "alpha_tau" => 1, "beta_tau" => 1,
        "alpha_sigma" => 1, "beta_sigma" => 1,
 	    "alpha_rho" => 1, "beta_rho" => 1);

if (!isfile("output/latent_sim_fit.jld"))
    BLAS.set_num_threads(32);
    tic = now();
    out = pg_stlm_latent(Y, X, locs, params, priors); # XX minutes for 200 iterations -- can this be sped up more through parallelization?
    # parallelization for omega running time of 20 mexinutes for 200 iterations on macbook
    # parallelization with 64 threads takes 19 minutes for 200 iterations on statszilla
    # parallelization with 32 threads takes 18 minutes for 200 iterations on statszilla
    toc = now();
    
    save("output/latent_sim_fit.jld", "data", out);
end    




#mean(select(out, r"beta"), dims=1)
#mean(out["beta"], dims=1)
#beta

#out |> @df plot(^(1:nrow(out)), cols(1:4))
#out |> @df plot(^(1:nrow(out)), cols("beta[1, 1]"))


#@df out plot(^(1:nrow(out)), :r"beta")


#Threads.nthreads()
#BLAS.get_num_threads()
#Sys.CPU_THREADS




