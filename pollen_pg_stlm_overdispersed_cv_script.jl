println("Starting pollen_pg_stlm_overdispersed_cv_script")
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

Threads.nthreads()
BLAS.set_num_threads(32);

# load the pollen data

# dat = RData.load("./data/pollen_dat_5.0.RDS", convert=true)
Y = load("./data/pollen_data_5.0.h5")["pollen"];
K = 8
for k in 1:K
    println("Fitting Overdispersed model for fold $k of $K")
    flush(stdout)

    Y = load("output/cross-validate/pollen_5.0_test_fold_$k.rds")
    # round to integer -- for this data, this is the same as floor and ceil
    Y = round.(Union{Missing, Int64}, Y);
    # add in design matrix with intercept only
    X = reshape(ones(size(Y)[1]), size(Y)[1], 1);
    p = size(X)[2]
    # load the location data
    locs = load("./data/pollen_locs_5.0.h5")["locs"];
    rescale = 1e3
    locs = locs / rescale
    params = Dict{String, Int64}("n_adapt" => 5000, "n_mcmc" => 5000, "n_thin" => 5, "n_message" => 50, "n_save" => 200);

    priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(100.0 .* ones(p)),
            "mean_range" => [-2, 0], "sd_range" => [2, 2],
            "alpha_tau" => 0.1, "beta_tau" => 0.1,
            "alpha_sigma" => 1, "beta_sigma" => 1,
            "alpha_rho" => 0.1, "beta_rho" => 0.1);


    out = pg_stlm_overdispersed(Y, X, locs, params, priors, corr_fun="matern", path = "./output/cross-validate/pollen_overdispersed_fit_fold_$k.jld"); 
    if (out["runtime"] / (60*1000) < 120)
        println("Model fitting took ",  out["runtime"]/(60*1000), " minutes")
        flush(stdout)
    else
        println("Model fitting took ",  out["runtime"]/(60*60*1000), " hours")
        flush(stdout)   
    end

    alert("Finished Overdispersed fitting cross-validation fold $k out of $K")

    if (!isfile("./output/cross-validate/pollen_overdispersed_fit_fold_$k.rds"))
        R"saveRDS($out, file = './output/cross-validate/pollen_overdispersed_fit_fold_$k.rds', compress = FALSE)";
    end
end 

