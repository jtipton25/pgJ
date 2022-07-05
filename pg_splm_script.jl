# bash command to run script is 
# julia -t 32 pg_splm_script.jl > pg_splm_out.txt &
using Random, Distributions, LinearAlgebra, PDMats, Plots;
using DataFrames, Distances, GaussianRandomFields;
#using GaussianProcesses;
using ThreadsX, Dates;
#using JLD2, FileIO;
using JLD, HDF5;
using RCall;
using StatsBase;
using Plots;
# using PolyaGammaSamplers
include("src/log_sum_exp.jl")
include("src/softmax.jl")
include("src/eta_to_pi.jl")
include("src/calc_Mi.jl")
include("src/calc_kappa.jl")
include("src/polyagamma.jl")
include("src/update_tuning.jl")
include("src/pg_splm.jl")

Threads.nthreads()



Random.seed!(2022);

N = 33^2;
#N = 100^2;
p = 2;
J = 6;
#locs = collect(product(
#     collect(range(0, stop=1, length=isqrt(N))),
#     collect(range(0, stop=1, length=isqrt(N)))))

locs = vec(collect(Iterators.product(
     range(0, stop=1, length=isqrt(N)),
     range(0, stop=1, length=isqrt(N)))));

locs = Matrix(reshape(reinterpret(Float64, locs), (2,:))');

D = pairwise(Distances.Euclidean(), locs, locs, dims=1);

tau = range(0.25, 1, length=J-1) .* ones(J-1);
#theta = exp.(rand(Uniform(-1.5, 0.5), J-1)); 
theta = exp.(rand(Uniform(-1.5, 0.5), J-1)); 

#cov_fun = GaussianRandomFields.Exponential(theta)

#Sigma = tau * apply(cov_fun, D)
# TODO: figure out how to do the GP kernels later, start by hand with exponential kernel
R = [exp.(-D / v) for v in theta];
R_chol = [cholesky(v) for v in R];
Sigma = [tau[j]^2 * R[j] for j in 1:(J-1)];
Sigma_chol = [tau[j] * R_chol[j].U for j in 1:(J-1)];
Sigma_chol2 = [cholesky(v) for v in Sigma];

Sigma_chol3 = R_chol;
for j in 1:(J-1)
    Sigma_chol3[j].U .*= tau[j];
end

all(isapprox.(Sigma_chol2[1].U, Sigma_chol3[1].U));

@time Sigma_inv = [inv(v) for v in Sigma_chol];
@time Sigma_inv2 = [inv(v) for v in Sigma_chol2];

j=1;

#@time rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol[j])), 1);  # expect an error
@time rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol2[j])), 1);
@time rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol3[j])), 1);

psi = Array{Float64}(undef, (N, J-1));

for j in 1:(J-1)
    psi[:, j] = rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol3[j])), 1);
end    

# setup the fixed effects
X = rand(Normal(0, 1), N, p-1);
X = hcat(ones(N), X);
# size(X)

beta = Array{Float64}(undef, p, J-1)
beta_0 = range(-2.5, -0.5, length=J-1)
for j in 1:(J-1)
    beta[:, j] = [beta_0[j] rand(Normal(0, 0.25), p-1)];
end
eta = X * beta + psi;
pi = reduce(hcat, map(eta_to_pi, eachrow(eta)))';

size(pi);
sum(pi[1, :]);
sum(pi[41,:]);



Y = Array{Union{Missing, Integer}}(undef, (N, J));

Ni = rand(Poisson(500), N);
for i in 1:N
    Y[i, :] = rand(Multinomial(Ni[i], pi[i, :]), 1);
end

# add in some missing values
missing_idx = StatsBase.sample([false, true], ProbabilityWeights([0.8, 0.2]), N);
Y[missing_idx, :] .= missing;


dat_sim = Dict{String, Any}("N" => N, "Y" => Y, "X" => X, "Ni" => Ni, "missing_idx" => missing_idx,
            "beta" => beta, "p" => p, "J" => J, 
            "locs" => locs, "tau" => tau, "theta" => theta,
            "psi" => psi, "eta" => eta, "pi" =>pi);
save("output/pg_splm_sim_data.jld", "data", dat_sim);            
R"saveRDS($dat_sim, file = 'output/pg_splm_sim_data.RDS')";

params = Dict{String, Int64}("n_adapt" => 2000, "n_mcmc" => 5000, "n_thin" => 5, "n_message" => 50, "mean_range" => 0, "sd_range" => 10, "alpha_tau" => 1, "beta_tau" => 1);

priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(10.0 .* ones(p)),
       	"mean_range" => 0, "sd_range" => 10,
	    "alpha_tau" => 1, "beta_tau" => 1,);

if (!isfile("output/pg_splm_sim_fit.jld"))
    BLAS.set_num_threads(32);
    tic = now();
    out = pg_splm(Y, X, locs, params, priors); # 32 minutes for 200 iterations -- can this be sped up more through parallelization?
    # parallelization for omega running time of 20 minutes for 200 iterations on macbook
    # parallelization with 64 threads takes 19 minutes for 200 iterations on statszilla
    # parallelization with 32 threads takes 18 minutes for 200 iterations on statszilla
    toc = now();

    save("output/pg_splm_sim_fit.jld", "data", out);
    R"saveRDS($out, file = 'output/pg_splm_sim_fit.RDS', compress = FALSE)";
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




#save the output in R format
#dat_sim=load("output/matern_sim_data.jld")["data"];
#R"saveRDS($dat_sim, file = 'output/matern_sim_data.RDS')";
#out=load("output/matern_sim_fit.jld")["data"];
#delete!(out, "runtime"); # remove the runtime which has a corrupted type
#R"saveRDS($out, file = 'output/matern_sim_fit.RDS', compress = FALSE)";