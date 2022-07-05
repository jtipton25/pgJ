# bash command to run script is 
# julia -t 32 pg_stlm_latent_script.jl > latent_out.txt &
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
include("src/pg_lm.jl")

Threads.nthreads()


Random.seed!(2022);


Random.seed!(2022);

N = 1000;
p = 2;
J = 4;
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
missing_idx = StatsBase.sample([false, true], ProbabilityWeights([0.8, 0.2]), N);
Y[missing_idx, :] .= missing;

dat_sim = Dict{String, Any}("N" => N, "Y" => Y, "X" => X, "Ni" => Ni, "missing_idx" => missing_idx,
            "beta" => beta, "p" => p, "J" => J, "eta" => eta, "pi" => reduce(hcat, pi)');
save("output/pg_lm_sim_data.jld", "data", dat_sim);    
R"saveRDS($dat_sim, file = 'output/pg_lm_sim_data.RDS')";        

params = Dict{String, Int64}("n_adapt" => 2000, "n_mcmc" => 5000, "n_thin" => 5, "n_message" => 50)

if (!isfile("output/pg_lm_sim_fit.jld"))
    BLAS.set_num_threads(32);
    tic = now();
    out = pg_lm(Y, X, params); # XX minutes for 200 iterations -- can this be sped up more through parallelization?
    # parallelization for omega running time of 20 mexinutes for 200 iterations on macbook
    # parallelization with 64 threads takes 19 minutes for 200 iterations on statszilla
    # parallelization with 32 threads takes 18 minutes for 200 iterations on statszilla
    toc = now();
    save("output/pg_lm_sim_fit.jld", "data", out);
    #delete!(out, "runtime"); # remove the runtime which has a corrupted type
    R"saveRDS($out, file = 'output/pg_lm_sim_fit.RDS', compress = FALSE)";
end    



#mean(select(out, r"beta"), dims=1)
#mean(out["beta"], dims=1)
#beta

#out |> @df plot(^(1:nrow(out)), cols(1:4))
#out |> @df plot(^(1:nrow(out)), cols("beta[1, 1]"))


#@df out plot(^(1:nrow(out)), :r"beta")


#save the output in R format
dat_sim=load("output/pg_lm_sim_data.jld")["data"];
R"saveRDS($dat_sim, file = 'output/pg_lm_sim_data.RDS')";
out=load("output/pg_lm_sim_fit.jld")["data"];
#delete!(out, "runtime"); # remove the runtime which has a corrupted type
R"saveRDS($out, file = 'output/pg_lm_sim_fit.RDS', compress = FALSE)";




