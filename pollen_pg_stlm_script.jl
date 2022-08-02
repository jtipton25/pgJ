# bash command to run script is 
# julia -t 32 pollen_pg_stlm_script.jl > pollen_matern_out.txt &
println("Starting pollen_pg_stlm_script")
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
# include("src/pg_stlm.jl")
include("src/pg_stlm_checkpoint.jl")
include("src/predict_pg_stlm.jl")
include("src/subset_posterior.jl")

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

# params = Dict{String, Int64}("n_adapt" => 200, "n_mcmc" => 100, "n_thin" => 5, "n_message" => 50);
# params = Dict{String, Int64}("n_adapt" => 5000, "n_mcmc" => 5000, "n_thin" => 5, "n_message" => 50);
params = Dict{String, Int64}("n_adapt" => 5000, "n_mcmc" => 5000, "n_thin" => 5, "n_message" => 50, "n_save" => 200);

priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(100.0 .* ones(p)),
       	"mean_range" => [-2, -2], "sd_range" => [2, 2],
	    "alpha_tau" => 0.1, "beta_tau" => 0.1,
 	    "alpha_rho" => 0.1, "beta_rho" => 0.1);


# if (!isfile("output/pollen/pollen_matern_fit.jld"))
    BLAS.set_num_threads(32);
    out = pg_stlm(Y, X, locs, params, priors, corr_fun="matern", path = "./output/pollen/pollen_matern_fit.jld"); 
    if (out["runtime"] / (60*1000) < 120)
        println("Model fitting took ",  out["runtime"]/(60*1000), " minutes")
    else
        println("Model fitting took ",  out["runtime"]/(60*60*1000), " hours")
    end

    # alert("Finished Matern fitting")

    # save("output/pollen/pollen_matern_fit.jld", "data", out);
    #delete!(out, "runtime"); # remove the runtime which has a corrupted type

    # out["Sigma_theta_tune_chol"] = nothing
    # out["Sigma_theta_tune"] = nothing
    setindex!(out, X, "X")
    setindex!(out, Y, "Y")
    setindex!(out, locs, "locs")

    # out["X"] = X
    # out["Y"] = Y
    # out["locs"] = locs
    # R"saveRDS($out, file = 'output/pollen/pollen_matern_fit.rds', compress = FALSE)";

# else
    # println("Loading saved model output from pollen_pg_stlm")
    # flush(stdout)
    # out = load("output/pollen/pollen_matern_fit.jld")["data"];
# end

#
# prediction
#

# subset_posterior(out, "matern")

locs_pred = Matrix(load("./data/grid_5.0.rds"));
locs_pred = locs_pred / rescale;
X_pred = reshape(ones(size(locs_pred)[1]), size(locs_pred)[1], 1);

# if (!isfile("output/pollen/pollen_matern_predictions.jld"))
    BLAS.set_num_threads(32);
    preds = predict_pg_stlm(out, X_pred, locs_pred, n_message = 1, n_save=10); 

    # save("output/pollen/pollen_matern_predictions.jld", "data", preds);
#     #delete!(out, "runtime"); # remove the runtime which has a corrupted type
    R"saveRDS($preds, file = 'output/pollen/pollen_matern_predictions.RDS', compress = FALSE)";

    # alert("Finished Matern predictions")
# end


