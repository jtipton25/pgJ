using HDF5
include("src/log_sum_exp.jl")
include("src/softmax.jl")
include("src/eta_to_pi.jl")
include("src/calc_Mi.jl")
include("src/calc_kappa.jl")
include("src/polyagamma.jl")
include("src/update_tuning.jl")
include("src/pg_stlm_overdispersed.jl")

# load the pollen data
fid = h5open("data/pollen_data_5.0.h5", "r")
pollen = fid["pollen"]
Y = read(pollen)
close(fid)
# convert to Int64
Y = convert(Array{Union{Float64, Missing}, 3}, Y)
Y[isnan.(Y)] .= missing
Y = convert(Array{Union{Int64, Missing}, 3}, Y)

# load the taxa data
fid = h5open("data/pollen_taxa_5.0.h5", "r")
taxa = fid["taxa"]
taxa = read(taxa)
close(fid)

# load the location data
fid = h5open("data/pollen_locs_5.0.h5", "r")
locs = fid["locs"]
locs = read(locs)
close(fid)

# Default fixed effects (mean only)
X = ones(size(Y, 1))
p = size(X, 2)

Threads.nthreads()

# fit the models

# 
# Fit the matern model
#

params = Dict{String, Int64}("n_adapt" => 100, "n_mcmc" => 100, "n_thin" => 5, 
        "n_message" => 50, "mean_range" => 0, "sd_range" => 10, "alpha_tau" => 1, 
        "beta_tau" => 1);
#params = Dict{String, Int64}("n_adapt" => 2000, "n_mcmc" => 5000, "n_thin" => 5,
#        "n_message" => 50, "mean_range" => 0, "sd_range" => 10, "alpha_tau" => 1, 
#        "beta_tau" => 1);

priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(10.0 .* ones(p)),
       	"mean_range" => 0, "sd_range" => 10,
        "alpha_tau" => 1, "beta_tau" => 1,
        "alpha_rho" => 1, "beta_rho" => 1);

if (!isfile("output/matern_pollen_fit.jld"))
    BLAS.set_num_threads(32);
    tic = now();
    out = pg_stlm(Y, X, locs, params, priors); # XX minutes for 200 iterations -- can this be sped up more through parallelization?
    # parallelization for omega running time of 20 mexinutes for 200 iterations on macbook
    # parallelization with 64 threads takes 19 minutes for 200 iterations on statszilla
    # parallelization with 32 threads takes 18 minutes for 200 iterations on statszilla
    toc = now();
end    

save("output/matern_pollen_fit.jld", "data", out);

# 
# Fit the overdispersed model
#

params = Dict{String, Int64}("n_adapt" => 100, "n_mcmc" => 100, "n_thin" => 5, 
        "n_message" => 50, "mean_range" => 0, "sd_range" => 10, "alpha_tau" => 1, 
        "beta_tau" => 1);
#params = Dict{String, Int64}("n_adapt" => 2000, "n_mcmc" => 5000, "n_thin" => 5,
#        "n_message" => 50, "mean_range" => 0, "sd_range" => 10, "alpha_tau" => 1, 
#        "beta_tau" => 1);

priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(10.0 .* ones(p)),
       	"mean_range" => 0, "sd_range" => 10,
        "alpha_tau" => 1, "beta_tau" => 1,
        "alpha_sigma" => 1, "beta_sigma" => 1,
 	    "alpha_rho" => 1, "beta_rho" => 1);

if (!isfile("output/overdispersed_pollen_fit.jld"))
    BLAS.set_num_threads(32);
    tic = now();
    out = pg_stlm_overdispersed(Y, X, locs, params, priors); # XX minutes for 200 iterations -- can this be sped up more through parallelization?
    # parallelization for omega running time of 20 mexinutes for 200 iterations on macbook
    # parallelization with 64 threads takes 19 minutes for 200 iterations on statszilla
    # parallelization with 32 threads takes 18 minutes for 200 iterations on statszilla
    toc = now();
end    

save("output/overdispersed_pollen_fit.jld", "data", out);

