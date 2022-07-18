# bash command to run script is 
# julia -t 32 pg_stlm_overdispersed_script.jl > overdispersed_out.txt &
println("Starting pg_stlm_overdispersed_script")
flush(stdout)
using Random, Distributions, LinearAlgebra, PDMats, Plots;
using DataFrames, Distances, GaussianRandomFields;
#using GaussianProcesses;
using ThreadsX, Dates;
#using JLD2, FileIO;
using JLD, HDF5;
using RCall;
using StatsBase;
# using PolyaGammaSamplers
include("src/log_sum_exp.jl")
include("src/softmax.jl")
include("src/eta_to_pi.jl")
include("src/calc_Mi.jl")
include("src/calc_kappa.jl")
include("src/polyagamma.jl")
include("src/update_tuning.jl")
include("src/pg_stlm_overdispersed.jl")

Threads.nthreads()



Random.seed!(2022);

N = 15^2;
p = 2;
J = 4;
n_time = 10;

# pollen size
#N = 33^2;
#p = 2;
#J = 16;
#n_time = 30;

locs = vec(collect(Iterators.product(
     range(0, stop=1, length=isqrt(N)),
     range(0, stop=1, length=isqrt(N)))));

locs = Matrix(reshape(reinterpret(Float64, locs), (2,:))');

D = pairwise(Distances.Euclidean(), locs, locs, dims=1);

I = Diagonal(ones(N));
sigma = rand(Uniform(0.5, 1.5), J-1);
tau = range(0.25, 1, length=J-1) .* ones(J-1);
#theta = exp.(rand(Uniform(-1.5, 0.5), J-1)); 
theta = exp.(rand(Uniform(-1.5, 0.5), J-1)); 
rho = rand(Uniform(0.8, 0.99), J-1);

#cov_fun = GaussianRandomFields.Exponential(theta)

#Sigma = tau * apply(cov_fun, D)
# TODO: figure out how to do the GP kernels later, start by hand with exponential kernel
R = [exp.(-D / v) for v in theta];
R_chol = [cholesky(v) for v in R];
Sigma = [tau[j]^2 * R[j] + sigma[j]^2 * I for j in 1:(J-1)];
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

beta = Array{Float64}(undef, p, J-1)
beta_0 = range(-2.5, -0.5, length=J-1)
for j in 1:(J-1)
    beta[:, j] = [beta_0[j] rand(Normal(0, 0.25), p-1)];
end

eta = Array{Float64}(undef, (N, J-1, n_time));
pi = Array{Float64}(undef, (N, J, n_time));

for t in 1:n_time
    eta[:, :, t] = X * beta + psi[:, :, t];
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
        if (missing_idx[i, t]) 
            Y[i, :, t] .= missing;
        end
    end
end


dat_sim = Dict{String, Any}("N" => N, "Y" => Y, "X" => X, "Ni" => Ni, "missing_idx" => missing_idx,
            "beta" => beta, "p" => p, "J" => J, "n_time" => n_time,
            "locs" => locs, "sigma" => sigma, "tau" => tau, "theta" => theta,
            "sigma" => sigma, "rho" => rho, "psi" => psi, "eta" => eta, "pi" =>pi);
save("output/overdispersed_sim_data.jld", "data", dat_sim);            
R"saveRDS($dat_sim, file = 'output/overdispersed_sim_data.RDS')";     

params = Dict{String, Int64}("n_adapt" => 1000, "n_mcmc" => 500, "n_thin" => 5, "n_message" => 50, "mean_range" => 0, "sd_range" => 10, "alpha_tau" => 1, "beta_tau" => 1);
# params = Dict{String, Int64}("n_adapt" => 2000, "n_mcmc" => 5000, "n_thin" => 5, "n_message" => 50, "mean_range" => 0, "sd_range" => 10, "alpha_tau" => 1, "beta_tau" => 1);


priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(10.0 .* ones(p)),
       	"mean_range" => 0, "sd_range" => 10,
        "alpha_tau" => 1, "beta_tau" => 1,
        "alpha_sigma" => 1, "beta_sigma" => 1,
 	    "alpha_rho" => 1, "beta_rho" => 1);

if (!isfile("output/overdispersed_sim_fit.jld"))
    BLAS.set_num_threads(32);
    tic = now();
    out = pg_stlm_overdispersed(Y, X, locs, params, priors); # XX minutes for 200 iterations -- can this be sped up more through parallelization?
    # parallelization for omega running time of 20 mexinutes for 200 iterations on macbook
    # parallelization with 64 threads takes 19 minutes for 200 iterations on statszilla
    # parallelization with 32 threads takes 18 minutes for 200 iterations on statszilla
    toc = now();
    
    save("output/overdispersed_sim_fit.jld", "data", out);
    #delete!(out, "runtime"); # remove the runtime which has a corrupted type
    R"saveRDS($out, file = 'output/overdispersed_sim_fit.RDS', compress = FALSE)";

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
#dat_sim=load("output/overdispersed_sim_data.jld")["data"];
#R"saveRDS($dat_sim, file = 'output/overdispersed_sim_data.RDS')";
#out=load("output/overdispersed_sim_fit.jld")["data"];
#delete!(out, "runtime"); # remove the runtime which has a corrupted type
#R"saveRDS($out, file = 'output/overdispersed_sim_fit.RDS', compress = FALSE)";