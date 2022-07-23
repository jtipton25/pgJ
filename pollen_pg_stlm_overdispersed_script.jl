# bash command to run script is 
# julia -t 32 pollen_pg_stlm_overdispersed_script.jl > pollen_overdispersed_out.txt &
println("Starting pollen_pg_stlm_overdispersed_script")
flush(stdout)
using Random, Distributions, LinearAlgebra, PDMats, Plots;
using DataFrames, Distances, GaussianRandomFields;
#using GaussianProcesses;
using ThreadsX, Dates;
#using JLD2, FileIO;
using JLD, HDF5;
using RCall;
using RData, CodecBzip2;
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
include("src/predict_pg_stlm_overdispersed.jl")

Threads.nthreads()

# load the pollen data

# dat = RData.load("./data/pollen_dat_5.0.RDS", convert=true)
Y = load("./data/pollen_data_5.0.h5")["pollen"];
# convert NaNs to missing
Y = replace(Y, NaN=>missing);
# round to integer -- for this data, this is the same as floor and ceil
Y = round.(Union{Missing, Int64}, Y);
# add in design matrix with intercept only
X = reshape(ones(size(Y)[1]), size(Y)[1], 1);
p = size(X)[2]
# load the location data
locs = load("./data/pollen_locs_5.0.h5")["locs"];
rescale = 1e4
locs = locs / rescale

#params = Dict{String, Int64}("n_adapt" => 200, "n_mcmc" => 100, "n_thin" => 5, "n_message" => 1);
params = Dict{String, Int64}("n_adapt" => 2000, "n_mcmc" => 5000, "n_thin" => 5, "n_message" => 50);

priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(100.0 .* ones(p)),
        "mean_range" => [-2, -2], "sd_range" => [2, 10],
	    "alpha_tau" => 0.1, "beta_tau" => 0.1,
        "alpha_sigma" => 1, "beta_sigma" => 1,
 	    "alpha_rho" => 0.1, "beta_rho" => 0.1);

if (!isfile("output/pollen_overdispersed_fit.jld"))
    BLAS.set_num_threads(32);
    tic = now();
    out = pg_stlm_overdispersed(Y, X, locs, params, priors, corr_fun="matern"); 
    toc = now();

    save("output/pollen_overdispersed_fit.jld", "data", out);
    #delete!(out, "runtime"); # remove the runtime which has a corrupted type
    R"saveRDS($out, file = 'output/pollen_overdispersed_fit.RDS', compress = FALSE)";
end



