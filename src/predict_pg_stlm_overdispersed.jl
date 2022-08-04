using SparseArrays
using Graphs
using LinearAlgebra
include("correlation_function.jl")

export predict_pg_stlm_overdispersed

"""
predict_pg_stlm_overdispersed(out, X_pred, locs_pred; posterior_mean_only=false, n_message=50)

Return the predictions at unobserved locations `locs_pred` with covariate values `X_pred` given MCMC model output `out` fitted at observed locations `locs` with observed covariates `X`. For reduced computational effort, only the posterior mean predictions can be made by setting `posterior_mean_only` to `true`.
"""
function predict_pg_stlm_overdispersed(out, X_pred, locs_pred; path="./output/pollen/pollen_overdispersed_predictions.jld", posterior_mean_only=false, n_message = 50, n_save = 50)
    # TODO: add in class/object oriented form
    
    tic = now()

    X = out["X"]
    locs = out["locs"]
    corr_fun = out["corr_fun"]

    beta = out["beta"]
    theta = out["theta"]
    tau2 = out["tau"].^2
    sigma2 = out["sigma"].^2
    eta = out["eta"]
    rho = out["rho"]

    n_samples = size(beta, 1)
    N = size(X, 1)
    n_time = size(eta, 4)
    n_pred = size(X_pred, 1)
    J = size(beta, 3) + 1

    checkpoints = collect(1:n_save:n_samples)
    if checkpoints[end] != n_samples
        push!(checkpoints, n_samples + 1)
    end

    D_obs = pairwise(Euclidean(), locs, locs, dims = 1)
    D_pred = zeros((n_pred, n_pred))
    if !posterior_mean_only
        D_pred = pairwise(Euclidean(), locs_pred, locs_pred, dims = 1)
    end
    D_pred_obs = pairwise(Euclidean(), locs_pred, locs, dims = 1)

        if isfile(path)
        out_pred = load(path);
    else
        out_pred = Dict(
            "k" => Array{Int64}(undef, 0),
            "checkpoint_idx" => Array{Int64}(undef, 0),
            "eta" => Array{Float64}(undef, (n_samples, n_pred, J - 1, n_time)),
            "pi" => Array{Float64}(undef, (n_samples, n_pred, J, n_time)),
            "runtime" => Int(0) # milliseconds runtime as an Int
        )
    end

    if !isempty(out_pred["k"])
        if out_pred["k"][end] == n_samples
            delete!(out_pred, "k")
            delete!(out_pred, "checkpoint_idx")
            return out_pred
        end
    end

    checkpoint_idx = 1
    if !isempty(out_pred["checkpoint_idx"])
        checkpoint_idx = out_pred["checkpoint_idx"][end] + 1
    end
    k_start = 1
    if !isempty(out_pred["k"])
        k_start = out_pred["k"][end] + 1        
    end


    G_time = Graphs.grid((n_time, 1))
    W_time = Graphs.adjacency_matrix(G_time)

    println("Predicting new locations from MCMC. Running for ", n_samples, " iterations starting at iteration ", k_start)
    flush(stdout)

    if (n_pred > 20000) & !posterior_mean_only
       @assert false "Number of prediction points must be less than 20000 if posterior_mean_only=false" 
       flush(stdout)
    end

    # loop over the posterior samples
    for k in k_start:n_samples
        if (mod(k, n_message) == 0)
            println("Prediction iteration ", k, " out of ", n_samples)
		    flush(stdout)	 
        end

        # TODO: add in control for parallelization if matrices are large
        Threads.@threads for j in 1:(J-1)

            Q_time = spdiagm(vcat(1, (1 + rho[k, j]^2) * ones(n_time-2), 1)) - rho[k, j] * W_time
            Sigma_time = Matrix(Hermitian(inv(Matrix(Q_time))))
            Sigma_time_chol = cholesky(Sigma_time)
           
            Sigma = Matrix(Hermitian(covariance_function.(D_obs, tau2[k, j], (exp.(theta[k, j, :]), ), corr_fun=corr_fun) + sigma2[k, j] * I))
            Sigma_pred_obs = Matrix(covariance_function.(D_pred_obs, tau2[k, j], (exp.(theta[k, j, :]), ), corr_fun=corr_fun))
            Sigma_pred = zeros((n_pred, n_pred))
            if !posterior_mean_only                
                Sigma_pred = Matrix(Hermitian(covariance_function.(D_pred, tau2[k, j], (exp.(theta[k, j, :]), ), corr_fun=corr_fun) + sigma2[k, j] * I));
            end
            Sigma_chol = try 
                cholesky(Sigma)
            catch
                cholesky(Sigma + 1e-6 * I)
                @warn "The Covariance matrix has been mildly regularized. If this warning is rare, it should be ok to ignore it."
            end
            # Sigma_inv = inv(Sigma_chol.U)
            Sigma_inv = inv(Sigma_chol)
            Sigma_space_chol = zeros((n_pred, n_pred))
            if !posterior_mean_only
                Sigma_space_chol = try 
                    cholesky(Matrix(Hermitian(Sigma_pred - Sigma_pred_obs * Sigma_inv * Sigma_pred_obs')))                
                catch
                    cholesky(Matrix(Hermitian(Sigma_pred - Sigma_pred_obs * Sigma_inv * Sigma_pred_obs' + 1e-6 * I)))
                    @warn "The Covariance matrix has been mildly regularized. If this warning is rare, it should be ok to ignore it."
                end
            end

            pred_mean = Sigma_pred_obs * Sigma_inv * (eta[k, :, j, :] - repeat(X * beta[k, :, j], inner=(1, n_time))) + repeat(X_pred * beta[k, :, j], inner = (1, n_time))
            if posterior_mean_only
                out_pred["eta"][k, :, j, :] = pred_mean
            else
                out_pred["eta"][k, :, j, :] = pred_mean + Sigma_space_chol.U' * rand(Normal(0.0, 1.0), (n_pred, n_time)) * Sigma_time_chol.U
            end
        end

        if k == (checkpoints[checkpoint_idx+1] - 1)
            append!(out_pred["k"], k)
            append!(out_pred["checkpoint_idx"], checkpoint_idx)
            toc = now()
            out_pred["runtime"] += Int(Dates.value(toc - tic))
            tic = now()
            save(path, out_pred)
            if (k !== n_samples)
                checkpoint_idx += 1
            end
        end
    end

    Threads.@threads for k in 1:n_samples
        for t in 1:n_time
            out_pred["pi"][k, :, :, t] = reduce(hcat, map(eta_to_pi, eachrow(out_pred["eta"][k, :, :, t])))'
        end
    end
    
    toc = now()

    out_pred["runtime"] += Int(Dates.value(toc - tic))
    save(path, out_pred)
    
    return(out)
end