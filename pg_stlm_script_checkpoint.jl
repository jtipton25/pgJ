# bash command to run script is 
# julia -t 32 pg_stlm_script.jl > matern_out.txt &
println("Starting pg_stlm_script")
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
include("src/pg_stlm_checkpoint.jl")
include("src/correlation_function.jl")

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

tau = range(0.25, 1, length=J-1) .* ones(J-1);
#theta = exp.(rand(Uniform(-1.5, 0.5), J-1)); 
theta = exp.(rand(Uniform(-1.5, 0.5), J-1, 2)); 
rho = 0.9 * ones(J-1);

#cov_fun = GaussianRandomFields.Exponential(theta)

#Sigma = tau * apply(cov_fun, D)
# TODO: figure out how to do the GP kernels later, start by hand with exponential kernel
R = [correlation_function.(D, (v, )) for v in theta];
R_chol = [cholesky(v) for v in R];
Sigma = [tau[j]^2 * R[j] for j in 1:(J-1)];
Sigma_chol = [cholesky(v) for v in Sigma];
Sigma_chol2 = [tau[j] * R_chol[j].U for j in 1:(J-1)];

Sigma_chol3 = copy(R_chol);
for j in 1:(J-1)
    Sigma_chol3[j].U .*= tau[j];
end

all(isapprox.(Sigma_chol[1].U, Sigma_chol3[1].U));

@time Sigma_inv = [inv(v) for v in Sigma_chol];
@time Sigma_inv2 = [inv(v) for v in Sigma_chol2];

j=1;

(tau[j]^2 * PDMat(R[j], R_chol[j])) == PDMat(Sigma[j], Sigma_chol[j])
#@time rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol[j])), 1);  # expect an error
using BenchmarkTools
@benchmark rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol[j])), 1)
Sigma_mats = PDMat(Sigma[j], Sigma_chol[j])
@benchmark rand(MvNormal(zeros(N), Sigma_mats), 1)
@benchmark rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol3[j])), 1)
@benchmark rand(MvNormal(zeros(N), tau[j]^2 * PDMat(R[j], R_chol[j])), 1)
Sig_mats = tau[j]^2 * PDMat(R[j], R_chol[j])
@benchmark rand(MvNormal(zeros(N), Sig_mats), 1)

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
# size(X)

beta = Array{Float64}(undef, p, J-1)
beta_0 = range(-2.5, -0.5, length=J-1)
for j in 1:(J-1)
    beta[:, j] = [beta_0[j] rand(Normal(0, 0.25), p-1)];
end

eta = Array{Float64}(undef, (N, J-1, n_time));
pi = Array{Float64}(undef, (N, J, n_time));

for t in 1:n_time
    for j in 1:(J-1)
        if t == 1
            eta[:, j, t] = X * beta[:, j] + rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol3[j])), 1);
        else
            eta[:, j, t] = (1 - rho[j]) * X * beta[:, j] + rho[j] * eta[:, j, t-1] + rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol3[j])), 1);
        end
    end
    pi[:, :, t] = reduce(hcat, map(eta_to_pi, eachrow(eta[:, :, t])))';
end




size(pi);
sum(pi[1, :, 1]);
sum(pi[1, :, 10]);
sum(pi[41, :, 10]);


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
            "locs" => locs, "tau" => tau, "theta" => theta,
            "rho" => rho, "psi" => psi, "eta" => eta, "pi" =>pi);
# save("output/matern_sim_data.jld", "data", dat_sim);     
# R"saveRDS($dat_sim, file = 'output/matern_sim_data.RDS')";       


params = Dict{String, Int64}("n_adapt" => 1000, "n_mcmc" => 500, "n_thin" => 5, "n_message" => 5, "n_save" => 100);
# params = Dict{String, Int64}("n_adapt" => 2000, "n_mcmc" => 5000, "n_thin" => 5, "n_message" => 50);

priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(100.0 .* ones(p)),
       	"mean_range" => [-2, -2], "sd_range" => [2, 10],
	    "alpha_tau" => 1, "beta_tau" => 1,
 	    "alpha_rho" => 1, "beta_rho" => 1);



         # if (!isfile("output/matern_sim_fit.jld"))
    BLAS.set_num_threads(32);
    include("src/pg_stlm_checkpoint.jl")
    out = pg_stlm(Y, X, locs, params, priors, corr_fun="matern", path="matern_tmp.jld", save_full=false); 

    # save("output/matern_sim_fit.jld", "data", out);
    #delete!(out, "runtime"); # remove the runtime which has a corrupted type
    # R"saveRDS($out, file = 'output/matern_sim_fit.RDS', compress = FALSE)";
# end

# need to subset posterior to save as R object
# include("src/subset_posterior.jl")
# subset_posterior(out, "matern")
# R"saveRDS($out, file = 'output/matern_sim_fit.RDS', compress = FALSE)";
# Fit a Matern model

# priors = Dict{String, Any}("mu_beta" => zeros(p), "Sigma_beta" => Diagonal(100.0 .* ones(p)),
#        	"mean_range" => [0, -2], "sd_range" => [10, 2],
# 	    "alpha_tau" => 0.1, "beta_tau" => 0.1,
#  	    "alpha_rho" => 0.1, "beta_rho" => 0.1);

# out = pg_stlm(Y, X, locs, params, priors, corr_fun="matern"); 



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