include("update_tuning.jl")
include("correlation_function.jl")

# function pg_stlm(Y, X, locs, params, priors, n_cores)

export pg_stlm_overdispersed

"""
    pg_stlm_overdispersed(Y, X, locs, params, priors; corr_fun="exponential")

Return the MCMC output for a linear model with A 2-D Array of observations of Ints `Y` (with `missing` values), a 2-D Array of covariates `X`, A 2-D Array of locations `locs`, a `Dict` of model parameters `params`, and a `Dict` of prior parameter values `priors` with a correlation function `corr_fun` that can be either `"exponential"` or `"matern"`
"""
function pg_stlm_overdispersed(Y, X, locs, params, priors; corr_fun="exponential", path="./output/pollen/pollen_overdispersed_fit.jld", save_full=false, correct_initial_variance=true)

    tic = now()

    # check input (TODO)
    # check params (TODO)    

    N = size(Y, 1)
    J = size(Y, 2)
    n_time = size(Y, 3)
    p = size(X, 2)

    checkpoints = collect(1:params["n_save"]:(params["n_adapt"]+params["n_mcmc"]))
    if checkpoints[end] != params["n_adapt"] + params["n_mcmc"]
        push!(checkpoints, params["n_adapt"] + params["n_mcmc"] + 1)
    end

    n_save = Int(params["n_mcmc"] / params["n_thin"]) ## convert to an integer

    # load the file if it exists
    beta_init = nothing
    eta_init = nothing
    omega_init = nothing
    theta_init = nothing
    tau_init = nothing
    tau_accept_init = nothing
    tau_tune_init = nothing
    sigma_init = nothing
    sigma_accept_init = nothing
    sigma_tune_init = nothing
    rho_init = nothing
    rho_accept_init = nothing
    rho_tune_init = nothing
    theta_accept_init = nothing
    lambda_theta_init = nothing
    Sigma_theta_tune_init = nothing
    Sigma_theta_tune_chol_init = nothing


    if save_full
        if isfile(path)
            out = load(path)

            # load the last MCMC output values
            beta_init = out["beta"][out["k"][end], :, :]
            eta_init = out["eta"][out["k"][end], :, :, :]
            omega_init = out["omega"][out["k"][end], :, :, :]
            if corr_fun == "matern"
                theta_init = out["theta"][out["k"][end], :, :]'
            else
                theta_init = out["theta"][out["k"][end], :]
            end
            tau_init = out["tau"][out["k"][end], :]
            rho_init = out["rho"][out["k"][end], :]
            sigma_init = out["sigma"][out["k"][end], :]

            # need to recover the current tuning values
            theta_accept_init = out["theta_accept"]
            lambda_theta_init = out["lambda_theta"]
            Sigma_theta_tune_init = out["Sigma_theta_tune"]
            Sigma_theta_tune_chol_init = out["Sigma_theta_tune_chol"]
            rho_accept_init = out["rho_accept"]
            rho_tune_init = out["rho_tune"]
            sigma_accept_init = out["sigma_accept"]
            sigma_tune_init = out["sigma_tune"]
            tau_accept_init = out["tau_accept"]
            tau_tune_init = out["tau_tune"]
        else
            # initialize the Dict
            out = Dict(
                "k" => Array{Int64}(undef, 0),
                "checkpoint_idx" => Array{Int64}(undef, 0),
                "beta" => Array{Float64}(undef, (params["n_adapt"] + params["n_mcmc"], p, J - 1)),
                "eta" => Array{Float64}(undef, (params["n_adapt"] + params["n_mcmc"], N, J - 1, n_time)),
                "omega" => Array{Float64}(undef, (params["n_adapt"] + params["n_mcmc"], N, J - 1, n_time)),
                "theta" => Array{Float64}(undef, (params["n_adapt"] + params["n_mcmc"], J - 1)),
                "tau" => Array{Float64}(undef, (params["n_adapt"] + params["n_mcmc"], J - 1)),
                "sigma" => Array{Float64}(undef, (params["n_adapt"] + params["n_mcmc"], J - 1)),
                "pi" => Array{Float64}(undef, (params["n_adapt"] + params["n_mcmc"], N, J, n_time)),
                "rho" => Array{Float64}(undef, (params["n_adapt"] + params["n_mcmc"], J - 1)),
                "corr_fun" => corr_fun,
                "theta_accept" => 0,
                "lambda_theta" => Array{Float64}(undef, J - 1),
                "Sigma_theta_tune" => [0.01 * (1.8 * diagm([1]) .- 0.8) for j in 1:J-1],
                "rho_accept" => 0,
                "sigma_accept" => 0,
                "tau_accept" => 0,
                "Y" => Y,
                "X" => X,
                "locs" => locs,
                "params" => params,
                "priors" => priors,
                "runtime" => Int(0) # milliseconds runtime as an Int
            )
            out["Sigma_theta_tune_chol"] = [cholesky(Matrix(Hermitian(out["Sigma_theta_tune"][j]))) for j in 1:(J-1)]

            if corr_fun == "matern"
                out["theta"] = Array{Float64}(undef, (params["n_adapt"] + params["n_mcmc"], J - 1, 2))
                out["Sigma_theta_tune"] = [0.01 * (1.8 * diagm(ones(2)) .- 0.8) for j in 1:J-1]
                out["Sigma_theta_tune_chol"] = [cholesky(Matrix(Hermitian(out["Sigma_theta_tune"][j]))) for j in 1:(J-1)]
            end
        end
    else
        # only save the reduces samples
        if isfile(path)
            out = load(path)

            # load the last MCMC output values
            beta_init = out["beta_init"]
            eta_init = out["eta_init"]
            omega_init = out["omega_init"]
            theta_init = out["theta_init"]
            tau_init = out["tau_init"]
            rho_init = out["rho_init"]
            sigma_init = out["sigma_init"]

            # need to recover the current tuning values
            theta_accept_init = out["theta_accept"]
            lambda_theta_init = out["lambda_theta"]
            Sigma_theta_tune_init = out["Sigma_theta_tune"]
            Sigma_theta_tune_chol_init = out["Sigma_theta_tune_chol"]
            rho_accept_init = out["rho_accept"]
            rho_tune_init = out["rho_tune"]
            sigma_accept_init = out["sigma_accept"]
            sigma_tune_init = out["sigma_tune"]
            tau_accept_init = out["tau_accept"]
            tau_tune_init = out["tau_tune"]
        else
            # initialize the Dict
            out = Dict(
                "k" => 0,
                "checkpoint_idx" => Array{Int64}(undef, 0),
                "beta" => Array{Float64}(undef, (n_save, p, J - 1)),
                "eta" => Array{Float64}(undef, (n_save, N, J - 1, n_time)),
                "omega" => Array{Float64}(undef, (n_save, N, J - 1, n_time)),
                "theta" => Array{Float64}(undef, (n_save, J - 1)),
                "tau" => Array{Float64}(undef, (n_save, J - 1)),
                "sigma" => Array{Float64}(undef, (n_save, J - 1)),
                "pi" => Array{Float64}(undef, (n_save, N, J, n_time)),
                "rho" => Array{Float64}(undef, (n_save, J - 1)),
                "corr_fun" => corr_fun,
                "theta_accept" => 0,
                "lambda_theta" => Array{Float64}(undef, J - 1),
                "Sigma_theta_tune" => [0.1 * (1.8 * diagm([1]) .- 0.8) for j in 1:J-1],
                "rho_accept" => 0,
                "Y" => Y,
                "X" => X,
                "locs" => locs,
                "params" => params,
                "priors" => priors,
                "runtime" => Int(0) # milliseconds runtime as an Int
            )
            out["Sigma_theta_tune_chol"] = [cholesky(Matrix(Hermitian(out["Sigma_theta_tune"][j]))) for j in 1:(J-1)]
            if corr_fun == "matern"
                out["theta"] = Array{Float64}(undef, (n_save, J - 1, 2))
                out["Sigma_theta_tune"] = [0.1 * (1.8 * diagm(ones(2)) .- 0.8) for j in 1:J-1]
                out["Sigma_theta_tune_chol"] = [cholesky(Matrix(Hermitian(out["Sigma_theta_tune"][j]))) for j in 1:(J-1)]
            end
        end
    end

    #
    # return the MCMC output if the MCMC has fully run
    #

    if save_full
        if !isempty(out["k"])
            if out["k"][end] == (params["n_adapt"] + params["n_mcmc"])

                delete!(out, "eta_init")
                delete!(out, "beta_init")
                delete!(out, "Sigma_theta_tune_chol")
                delete!(out, "Sigma_theta_tune")
                delete!(out, "rho_init")
                delete!(out, "rho_tune")
                delete!(out, "rho_accept")
                delete!(out, "omega_init")
                delete!(out, "tau_tune")
                delete!(out, "tau_accept")
                delete!(out, "sigma_tune")
                delete!(out, "sigma_accept")
                delete!(out, "k")
                delete!(out, "sigma_init")
                delete!(out, "tau_init")
                delete!(out, "lambda_theta")
                delete!(out, "checkpoint_idx")
                delete!(out, "theta_accept")
                delete!(out, "theta_init")

                return (out)
            end
        end
    else
        if out["k"] == (params["n_adapt"] + params["n_mcmc"])

            delete!(out, "eta_init")
            delete!(out, "beta_init")
            delete!(out, "Sigma_theta_tune_chol")
            delete!(out, "Sigma_theta_tune")
            delete!(out, "rho_init")
            delete!(out, "rho_tune")
            delete!(out, "rho_accept")
            delete!(out, "omega_init")
            delete!(out, "tau_tune")
            delete!(out, "tau_accept")
            delete!(out, "sigma_tune")
            delete!(out, "sigma_accept")
            delete!(out, "k")
            delete!(out, "sigma_init")
            delete!(out, "tau_init")
            delete!(out, "lambda_theta")
            delete!(out, "checkpoint_idx")
            delete!(out, "theta_accept")
            delete!(out, "theta_init")

            return out
        end
    end

    #
    # setup the checkpoints
    #

    checkpoint_idx = 1
    if !isempty(out["checkpoint_idx"])
        checkpoint_idx = out["checkpoint_idx"][end] + 1
    end
    k_start = 1
    if !isempty(out["k"])
        if save_full
            k_start = out["k"][end] + 1
        else
            k_start = out["k"] + 1
        end
    end

    tX = X'

    Mi = Array{Int64}(undef, (N, J - 1, n_time))
    kappa = Array{Float64}(undef, (N, J - 1, n_time))
    missing_idx = Array{Bool}(undef, (N, n_time))
    nonzero_idx = Array{Bool}(undef, (N, J - 1, n_time))
    for t = 1:n_time
        for i = 1:N
            Mi[i, :, t] = calc_Mi(Y[i, :, t])
            kappa[i, :, t] = calc_kappa(Y[i, :, t], Mi[i, :, t])
            missing_idx[i, t] = any(ismissing.(Y[i, :, t]))
            nonzero_idx[i, :, t] = Mi[i, :, t] .!= 0
        end
    end

    print(
        "There are ",
        ifelse(sum(missing_idx) > 0, sum(missing_idx), "no"),
        " observations with missing count vectors \n",
    )
    flush(stdout)

    n_nonzero = sum(nonzero_idx)


    # default priors
    # mu_beta = zeros(p)
    # Sigma_beta = Diagonal(10.0 .* ones(p))

    mu_beta = priors["mu_beta"]
    Sigma_beta = priors["Sigma_beta"]

    # TODO add in custom priors for mu_beta and Sigma_beta

    Sigma_beta_chol = cholesky(Matrix(Hermitian(Sigma_beta)))
    Sigma_beta_inv = inv(Sigma_beta_chol)
    Sigma_beta_inv_mu_beta = Sigma_beta_inv * mu_beta

    # the first case using the pre-computed cholesky is much faster
    # p = 5000
    # mu_beta = zeros(p)
    # A = rand(Normal(0, 1), p, p)
    # A = A * A'
    # Sigma_beta = Diagonal(10.0 .* ones(p)) + A
    # Sigma_beta_chol = cholesky(Sigma_beta)
    #
    # @time beta = rand(MvNormal(mu_beta, PDMat(Sigma_beta, Sigma_beta_chol)), J-1);
    # @time beta = rand(MvNormal(mu_beta, PDMat(Sigma_beta)), J-1);

    # initialize beta
    beta = rand(MvNormal(mu_beta, PDMat(Sigma_beta, Sigma_beta_chol)), J - 1)
    if !isnothing(beta_init)
        beta = copy(beta_init)
    end

    # initialize Xbeta
    Xbeta = X * beta

    # initialize sigma
    sigma2 = rand(Gamma(priors["alpha_sigma"], priors["beta_sigma"]), J - 1)
    sigma2[sigma2.>25] .= 25
    sigma = sqrt.(sigma2)
    if !isnothing(sigma_init)
        sigma = copy(sigma_init)
    end
    sigma2 = sigma.^2

    # initialize theta (log-scale)
    if corr_fun == "exponential"
        @assert length(priors["mean_range"]) == 1 "The \"mean_range\" prior value must be a vector of length 1"
        @assert priors["mean_range"] isa Array{<:Number} "The \"mean_range\" prior value must be a vector of length 1"
        @assert length(priors["sd_range"]) == 1 "The \"sd_range\" prior value must be a vector of positive numbers of length 1"
        @assert all(priors["sd_range"] .> 0) "The \"sd_range\" prior value must be a vector of positive numbers of length 1"
        @assert priors["sd_range"] isa Array{<:Number} "The \"sd_range\" prior value must be a vector of positive numbers of length 1"
    elseif corr_fun == "matern"
        @assert length(priors["mean_range"]) == 2 "The \"mean_range\" prior value must be a vector of length 2"
        @assert priors["mean_range"] isa Array{<:Number} "The \"mean_range\" prior value must be a vector of length 1"
        @assert length(priors["sd_range"]) == 2 "The \"sd_range\" prior value must be a vector of positive numbers of length 1"
        @assert all(priors["sd_range"] .> 0) "The \"sd_range\" prior value must be a vector of positive numbers of length 1"
    else
        @assert false "The correlation function \"corr_fun\" must be either \"exponential\" or \"matern\""
    end
    theta_mean = priors["mean_range"]
    theta_var = priors["sd_range"] .^ 2

    theta = rand(MvNormal(theta_mean, diagm(theta_var)), J - 1)
    theta[theta.<-2] .= -2
    theta[theta.>0.1] .= 0.1
    if !isnothing(theta_init)
        theta = copy(theta_init)
    end

    # initilaize tau
    tau = rand(InverseGamma(priors["alpha_tau"], priors["beta_tau"]), J - 1)
    tau[tau.>10] .= 10
    if !isnothing(tau_init)
        tau = copy(tau_init)
    end

    # initialize rho
    rho = rand(Uniform(0, 1), J - 1)
    if !isnothing(rho_init)
        rho = copy(rho_init)
    end

    # TODO: check if initial values are supplied


    # setup the GP covariance
    D = pairwise(Euclidean(), locs, locs, dims=1)

    # R = [exp.(-D / exp(v)) for v in theta]
    R = [
        Matrix(Hermitian(correlation_function.(D, (exp.(v),), corr_fun=corr_fun))) for
        v in eachcol(theta)
    ] # broadcasting over D but not theta
    Sigma = [Matrix(Hermitian(tau[j]^2 * R[j] + sigma[j]^2 * I)) for j in 1:(J-1)]
    Sigma_chol = [cholesky(v) for v in Sigma]
    Sigma_inv = [inv(v) for v in Sigma_chol]


    # initialize eta
    eta = Array{Float64}(undef, (N, J - 1, n_time))
    for j in 1:(J-1)
        eta[:, j, 1] =
            Xbeta[:, j] + rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol[j])), 1)
        for t = 2:n_time
            eta[:, j, t] =
                Xbeta[:, j] +
                rand(MvNormal(rho[j] * eta[:, j, t-1], PDMat(Sigma[j], Sigma_chol[j])), 1)
        end
    end
    if !isnothing(eta_init)
        eta = copy(eta_init)
    end

    # initialize omega

    omega = zeros(N, J - 1, n_time)
    Mi_nonzero = Mi[nonzero_idx]
    eta_nonzero = eta[nonzero_idx]

    # parallel for loop
    #tmp = Array{Float64}(undef, n_nonzero);
    #@time Threads.@threads for i = 1:n_nonzero
    #    tmp[i] = rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i]))
    #end
    # sequential for loop
    #@time for i in 1:n_nonzero
    #	tmp[i] = rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i]))
    #end
    #@time ThreadsX.collect([rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i in 1:n_nonzero]);
    #@time [rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i in 1:n_nonzero];
    # sequential
    # omega[nonzero_idx] = [rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i in 1:n_nonzero]
    # parallel update
    omega[nonzero_idx] = ThreadsX.collect(
        rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i = 1:n_nonzero
    )
    if !isnothing(omega_init)
        omega = copy(omega_init)
    end


    # TODO: setup config
    sample_beta = true
    sample_omega = true
    sample_rho = true
    sample_tau = true
    sample_sigma = true
    sample_theta = true
    sample_eta = true

    #
    # setup save variables
    #

    beta_save = Array{Float64}(undef, (params["n_save"], p, J - 1))
    tau_save = Array{Float64}(undef, (params["n_save"], J - 1))
    sigma_save = Array{Float64}(undef, (params["n_save"], J - 1))
    rho_save = Array{Float64}(undef, (params["n_save"], J - 1))
    theta_save = Array{Float64}(undef, (params["n_save"], J - 1))
    if corr_fun == "matern"
        theta_save = Array{Float64}(undef, (params["n_save"], J - 1, 2))
    end
    eta_save = Array{Float64}(undef, (params["n_save"], N, J - 1, n_time))
    pi_save = Array{Float64}(undef, (params["n_save"], N, J, n_time))
    omega_save = Array{Float64}(undef, (params["n_save"], N, J - 1, n_time))
    k_vec = zeros(Int64, params["n_save"])
    if !save_full
        beta_save = Array{Float64}(undef, (Int(params["n_save"] / params["n_thin"]), p, J - 1))
        tau_save = Array{Float64}(undef, (Int(params["n_save"] / params["n_thin"]), J - 1))
        sigma_save = Array{Float64}(undef, (Int(params["n_save"] / params["n_thin"]), J - 1))
        rho_save = Array{Float64}(undef, (Int(params["n_save"] / params["n_thin"]), J - 1))
        theta_save = Array{Float64}(undef, (Int(params["n_save"] / params["n_thin"]), J - 1))
        if corr_fun == "matern"
            theta_save = Array{Float64}(undef, (Int(params["n_save"] / params["n_thin"]), J - 1, 2))
        end
        eta_save = Array{Float64}(undef, (Int(params["n_save"] / params["n_thin"]), N, J - 1, n_time))
        pi_save = Array{Float64}(undef, (Int(params["n_save"] / params["n_thin"]), N, J, n_time))
        omega_save = Array{Float64}(undef, (Int(params["n_save"] / params["n_thin"]), N, J - 1, n_time))
        k_vec = zeros(Int64, Int(params["n_save"] / params["n_thin"]))
    end

    #
    # MCMC tuning
    #

    # tuning for theta
    theta_accept = zeros(J - 1)
    if !isnothing(theta_accept_init)
        theta_accept = copy(theta_accept_init)
    end
    lambda_theta = 0.01 * ones(J - 1)
    theta_accept_batch = zeros(J - 1)
    theta_batch = Array{Float64}(undef, 50, J - 1)
    Sigma_theta_tune = [0.01 * (1.8 * diagm([1]) .- 0.8) for j in 1:J-1]
    Sigma_theta_tune_chol = [cholesky(Matrix(Hermitian(Sigma_theta_tune[j]))) for j in 1:(J-1)]

    if corr_fun == "matern"
        theta_batch = Array{Float64}(undef, 50, J - 1, 2)
        Sigma_theta_tune = [0.01 * (1.8 * diagm(ones(2)) .- 0.8) for j in 1:J-1]
        Sigma_theta_tune_chol = [cholesky(Matrix(Hermitian(Sigma_theta_tune[j]))) for j in 1:(J-1)]
    end
    if !isnothing(lambda_theta_init)
        lambda_theta = copy(lambda_theta_init)
    end
    if !isnothing(Sigma_theta_tune_init)
        Sigma_theta_tune = copy(Sigma_theta_tune_init)
    end
    if !isnothing(Sigma_theta_tune_chol_init)
        Sigma_theta_tune_chol = copy(Sigma_theta_tune_chol_init)
    end

    # tuning for rho
    rho_accept = zeros(J - 1)
    if !isnothing(rho_accept_init)
        rho_accept = copy(rho_accept_init)
    end
    rho_accept_batch = zeros(J - 1)
    rho_tune = 0.025 * ones(J - 1)
    if !isnothing(rho_tune_init)
        rho_tune = copy(rho_tune_init)
    end

    # tuning for tau
    tau_accept = zeros(J - 1)
    if !isnothing(tau_accept_init)
        tau_accept = copy(tau_accept_init)
    end
    tau_accept_batch = zeros(J - 1)
    tau_tune = 0.5 * ones(J - 1)
    if !isnothing(tau_tune_init)
        tau_tune = copy(tau_tune_init)
    end

    # tuning for sigma
    sigma_accept = zeros(J - 1)
    if !isnothing(sigma_accept_init)
        sigma_accept = copy(sigma_accept_init)
    end
    sigma_accept_batch = zeros(J - 1)
    sigma_tune = 0.5 * ones(J - 1)
    if !isnothing(sigma_tune_init)
        sigma_tune = copy(sigma_tune_init)
    end


    println(
        "Starting MCMC. Running for ",
        params["n_adapt"],
        " adaptive iterations and ",
        params["n_mcmc"],
        " fitting iterations starting at iteration ",
        k_start
    )
    flush(stdout)



    # MCMC loop
    for k = k_start:(params["n_adapt"]+params["n_mcmc"])

        if (k == params["n_adapt"] + 1)
            println("Starting MCMC fitting. Running for ", params["n_mcmc"], " iterations")
            flush(stdout)
        end
        if (mod(k, params["n_message"]) == 0)
            if (k <= params["n_adapt"])
                println("MCMC adaptation iteration ", k, " out of ", params["n_adapt"])
                flush(stdout)
            else
                println(
                    "MCMC fitting iteration ",
                    k - params["n_adapt"],
                    " out of ",
                    params["n_mcmc"],
                )
                flush(stdout)
            end
        end

        #
        # sample omega
        #

        if (sample_omega)
            eta_nonzero = eta[nonzero_idx]
            #@time omega[nonzero_idx] =
            #    ThreadsX.collect(rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i = 1:n_nonzero);
            #@time omega[nonzero_idx] =
            #    [rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i = 1:n_nonzero];
            omega[nonzero_idx] = ThreadsX.collect(
                rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i = 1:n_nonzero
            )
        end

        #
        # sample beta
        #

        if (sample_beta)
            for j in 1:(J-1)
                tXSigma_inv = tX * Sigma_inv[j]
                if correct_initial_variance
                    A = ((1.0 - rho[j]^2) + (n_time - 1.0) * (1.0 - rho[j])^2) * tXSigma_inv * X + Sigma_beta_inv
                    b = dropdims(
                        (1.0 - rho[j]^2) * tXSigma_inv * eta[:, j, 1] +
                        sum(
                            (1 - rho[j]) *
                            tXSigma_inv *
                            (eta[:, j, 2:n_time] - rho[j] * eta[:, j, 1:(n_time-1)]),
                            dims=2,
                        ) +
                        Sigma_beta_inv_mu_beta,
                        dims=2,
                    )
                    beta[:, j] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
                else
                    A = (1.0 + (n_time - 1.0) * (1.0 - rho[j])^2) * tXSigma_inv * X + Sigma_beta_inv
                    b = dropdims(
                        tXSigma_inv * eta[:, j, 1] +
                        sum(
                            (1 - rho[j]) *
                            tXSigma_inv *
                            (eta[:, j, 2:n_time] - rho[j] * eta[:, j, 1:(n_time-1)]),
                            dims=2,
                        ) +
                        Sigma_beta_inv_mu_beta,
                        dims=2,
                    )
                    beta[:, j] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
                end

            end
        end

        # update Xbeta
        Xbeta = X * beta

        #
        # sample eta
        #

        if (sample_eta)
            Threads.@threads for t = 1:n_time # time first XX seconds 
                if t == 1
                    # initial time
                    for j in 1:(J-1)
                        if correct_initial_variance
                            A = ((1.0 - rho[j]^2) + rho[j]^2) * Sigma_inv[j] + Diagonal(omega[:, j, 1])
                            b =
                                Sigma_inv[j] * (
                                    ((1.0 - rho[j]^2) - rho[j] + rho[j]^2) * Xbeta[:, j] +
                                    rho[j] * eta[:, j, 2]
                                ) + kappa[:, j, 1]
                            eta[:, j, 1] =
                                rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
                        else
                            A = (1.0 + rho[j]^2) * Sigma_inv[j] + Diagonal(omega[:, j, 1])
                            b =
                                Sigma_inv[j] * (
                                    (1.0 - rho[j] + rho[j]^2) * Xbeta[:, j] +
                                    rho[j] * eta[:, j, 2]
                                ) + kappa[:, j, 1]
                            eta[:, j, 1] =
                                rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
                        end
                    end
                elseif t == n_time
                    # final time
                    for j in 1:(J-1)
                        A = Sigma_inv[j] + Diagonal(omega[:, j, n_time])
                        b =
                            Sigma_inv[j] *
                            ((1.0 - rho[j]) * Xbeta[:, j] + rho[j] * eta[:, j, n_time-1]) +
                            kappa[:, j, n_time]
                        eta[:, j, n_time] =
                            rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
                    end
                else
                    # middle times
                    for j in 1:(J-1)
                        A = (1.0 + rho[j]^2) * Sigma_inv[j] + Diagonal(omega[:, j, t])
                        b =
                            Sigma_inv[j] * (
                                (1.0 - rho[j])^2 * Xbeta[:, j] +
                                rho[j] * (eta[:, j, t-1] + eta[:, j, t+1])
                            ) + kappa[:, j, t]
                        eta[:, j, t] =
                            rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
                    end
                end
            end
        end

        #
        # Sample tau
        #

        if (sample_tau)
            Threads.@threads for j in 1:(J-1)
                tau_star = exp(rand(Normal(log(tau[j]), tau_tune[j])))
                Sigma_star = Matrix(Hermitian(tau_star^2 * R[j] + sigma[j]^2 * I))
                Sigma_chol_star = try
                    cholesky(Sigma_star)
                catch
                    println("theta[:,j] = ", theta[:, j], " sigma[j] = ", sigma[j], "tau[j] = ", tau[j], "tau_star = ", tau_star)
                    flush(stdout)
                    @warn "The Covariance matrix for updating tau2 has been mildly regularized. If this warning is rare, it should be ok to ignore it."
                    flush(stderr)
                    cholesky(Matrix(Hermitian(Sigma_star + 1e-6 * I)))
                end

                mh1 =
                    sum([
                        logpdf(
                            MvNormal(
                                (1 - rho[j]) * Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma_star, Sigma_chol_star),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(
                        InverseGamma(priors["alpha_tau"], priors["beta_tau"]),
                        tau_star,
                    ) +
                    log(tau_star)

                mh2 =
                    sum([
                        logpdf(
                            MvNormal(
                                (1 - rho[j]) * Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(InverseGamma(priors["alpha_tau"], priors["beta_tau"]), tau[j]) +
                    log(tau[j])

                if correct_initial_variance
                    mh1 += logpdf(
                        MvNormal(Xbeta[:, j], 1.0 / (1.0 - rho[j]^2) * PDMat(Sigma_star, Sigma_chol_star)),
                        eta[:, j, 1],
                    )
                    mh2 += logpdf(
                        MvNormal(Xbeta[:, j], 1.0 / (1.0 - rho[j]^2) * PDMat(Sigma[j], Sigma_chol[j])),
                        eta[:, j, 1],
                    )
                else
                    mh1 += logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma_star, Sigma_chol_star)),
                        eta[:, j, 1],
                    )
                    mh2 += logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma[j], Sigma_chol[j])),
                        eta[:, j, 1],
                    )
                end

                mh = exp(mh1 - mh2)
                if mh > rand(Uniform(0, 1))
                    tau[j] = tau_star
                    Sigma[j] = Sigma_star
                    Sigma_chol[j] = Sigma_chol_star
                    Sigma_inv[j] = inv(Sigma_chol_star)
                    if k <= params["n_adapt"]
                        tau_accept_batch[j] += 1.0 / 50.0
                    else
                        tau_accept[j] += 1.0 / params["n_mcmc"]
                    end
                end
            end
        end

        # adapt the tuning for tau
        if k <= params["n_adapt"]
            if (mod(k, 50) == 0)
                out_tuning = update_tuning_vec(k, tau_accept_batch, tau_tune)
                tau_accept_batch = out_tuning["accept"]
                tau_tune = out_tuning["tune"]
            end
        end

        #
        # sample theta
        #

        if (sample_theta)
            Threads.@threads for j in 1:(J-1)
                theta_star = rand(
                    MvNormal(
                        theta[:, j],
                        sqrt(lambda_theta[j]) *
                        PDMat(Sigma_theta_tune[j], Sigma_theta_tune_chol[j]),
                    ),
                )
                # if (corr_fun == "matern") & ((theta_star[1] > 4.1) | (theta_star[2] < -6.3))
                #     # eliminate Matern correlation function failure
                #     @warn "The proposal for theta_star was potentially computationally unstable and the MH proposal was discarded. If this warning is rare, it should be ok to ignore it."
                        # flush(stderr)
                #     flush(stdout)
                # else
                    R_star = correlation_function.(D, (exp.(theta_star),), corr_fun=corr_fun) # broadcasting over D but not theta_star
                    # R_star = try
                    #     Matrix(
                    #         Hermitian(
                    #             correlation_function.(D, (exp.(theta_star),), corr_fun=corr_fun),
                    #         ),
                    #     ) # broadcasting over D but not theta_star
                    # catch
                    #     println("theta_star = ", theta_star)
                    #     flush(stdout)
                    #     @warn "The proposal for theta_star was potentially computationally unstable and the MH proposal was discarded. If this warning is rare, it should be ok to ignore it."
                    #     if k <= params["n_adapt"]
                    #             theta_accept_batch[j] -= 1.0 / 50.0
                    #     else
                    #         theta_accept[j] -= 1.0 / params["n_mcmc"]
                    #     end
                    #     theta_star = theta[:, j]
                    #     R[j]
                    # end
                    if (any(isnan.(R_star)))
                        println("theta[:,j] = ", theta[:, j], "theta_star = ", theta_star, " sigma[j] = ", sigma[j], "tau[j] = ", tau[j])
                        flush(stdout)
                        @warn "The proposal for theta_star was potentially computationally unstable and the MH proposal was discarded. If this warning is rare, it should be ok to ignore it."
                        flush(stderr)
                        if k <= params["n_adapt"]
                            theta_accept_batch[j] -= 1.0 / 50.0
                        else
                            theta_accept[j] -= 1.0 / params["n_mcmc"]
                        end
                        theta_star = theta[:, j]
                        R_star = R[j]
                    end

                    Sigma_star = Matrix(Hermitian(tau[j]^2 * R_star + sigma[j]^2 * I))
                    Sigma_chol_star = try
                        cholesky(Sigma_star)
                    catch
                        println("theta[:,j] = ", theta[:, j], "theta_star = ", theta_star, " sigma[j] = ", sigma[j], "tau[j] = ", tau[j])
                        flush(stdout)
                        @warn "The Covariance matrix for updating theta has been mildly regularized. If this warning is rare, it should be ok to ignore it."
                        flush(stderr)
                        cholesky(Matrix(Hermitian(Sigma_star + 1e-6 * I)))
                    end
                    # Sigma_chol_star = try
                    #     cholesky(R_star)
                    # catch
                    #     @warn string("The Covariance matrix for updating theta has been mildly regularized with theta_star = ", theta_star, ". If this warning is rare, it should be ok to ignore it.")
                    # flush(stderr)
                    #     try
                    #          cholesky(Hermitian(Sigma_star + 1e-6 * I))
                    #     catch
                    #         @warn string("The Covariance matrix for updating theta has been moderately regularized with theta_star = ", theta_star, ". If this warning is rare, it should be ok to ignore it.")
                    # flush(stderr)
                    #         try
                    #             cholesky(Hermitian(Sigma_star + 1e-4 * I))
                    #         catch
                    #             @warn string("The Covariance matrix for updating theta has been strongly regularized with theta_star = ", theta_star, ". If this warning is rare, it should be ok to ignore it.")
                    # flush(stderr)
                    #             cholesky(Hermitian(Sigma_star + 1e-2 * I))
                    #         end
                    #     end
                    # end

                    mh1 =
                        sum([
                            logpdf(
                                MvNormal(
                                    (1 - rho[j]) * Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                    PDMat(Sigma_star, Sigma_chol_star),
                                ),
                                eta[:, j, t],
                            ) for t = 2:n_time
                        ]) +
                        logpdf(MvNormal(theta_mean, diagm(theta_var)), theta_star)

                    mh2 =
                        sum([
                            logpdf(
                                MvNormal(
                                    (1 - rho[j]) * Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                    PDMat(Sigma[j], Sigma_chol[j]),
                                ),
                                eta[:, j, t],
                            ) for t = 2:n_time
                        ]) +
                        logpdf(MvNormal(theta_mean, diagm(theta_var)), theta[:, j])

                    if correct_initial_variance
                        mh1 += logpdf(
                            MvNormal(Xbeta[:, j], 1.0 / (1.0 - rho[j]^2) * PDMat(Sigma_star, Sigma_chol_star)),
                            eta[:, j, 1],
                        )
                        mh2 += logpdf(
                            MvNormal(Xbeta[:, j], 1.0 / (1.0 - rho[j]^2) * PDMat(Sigma[j], Sigma_chol[j])),
                            eta[:, j, 1],
                        )
                    else
                        mh1 += logpdf(
                            MvNormal(Xbeta[:, j], PDMat(Sigma_star, Sigma_chol_star)),
                            eta[:, j, 1],
                        )
                        mh2 += logpdf(
                            MvNormal(Xbeta[:, j], PDMat(Sigma[j], Sigma_chol[j])),
                            eta[:, j, 1],
                        )
                    end

                    mh = exp(mh1 - mh2)
                    if mh > rand(Uniform(0, 1))
                        theta[:, j] = theta_star
                        R[j] = R_star
                        Sigma[j] = Sigma_star
                        Sigma_chol[j] = Sigma_chol_star
                        Sigma_inv[j] = inv(Sigma_chol_star)
                        if k <= params["n_adapt"]
                            theta_accept_batch[j] += 1.0 / 50.0
                        else
                            theta_accept[j] += 1.0 / params["n_mcmc"]
                        end
                    end
                # end
            end
        end

        # adapt the tuning for theta
        if k <= params["n_adapt"]
            save_idx = mod(k, 50)
            if mod(k, 50) == 0
                save_idx = 50
            end
            theta_batch[save_idx, :, :] = theta'
            if (mod(k, 50) == 0)
                out_tuning = update_tuning_mv_mat(
                    k,
                    theta_accept_batch,
                    lambda_theta,
                    theta_batch,
                    Sigma_theta_tune,
                    Sigma_theta_tune_chol,
                )
                theta_accept_batch = out_tuning["accept"]
                lambda_theta = out_tuning["lambda"]
                theta_batch = out_tuning["batch_samples"]
                Sigma_theta_tune = out_tuning["Sigma_tune"]
                Sigma_theta_tune_chol = out_tuning["Sigma_tune_chol"]
            end
        end

        #
        # sample rho
        #

        if (sample_rho)
            Threads.@threads for j in 1:(J-1)
                rho_star = rand(Normal(rho[j], rho_tune[j]))
                if ((rho_star < 1) & (rho_star > -1))
                    mh1 =
                        sum([
                            logpdf(
                                MvNormal(
                                    (1 - rho_star) * Xbeta[:, j] +
                                    rho_star * eta[:, j, t-1],
                                    PDMat(Sigma[j], Sigma_chol[j]),
                                ),
                                eta[:, j, t],
                            ) for t = 2:n_time
                        ]) + logpdf(Beta(priors["alpha_rho"], priors["beta_rho"]), rho_star)

                    mh2 =
                        sum([
                            logpdf(
                                MvNormal(
                                    (1 - rho[j]) * Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                    PDMat(Sigma[j], Sigma_chol[j]),
                                ),
                                eta[:, j, t],
                            ) for t = 2:n_time
                        ]) + logpdf(Beta(priors["alpha_rho"], priors["beta_rho"]), rho[j])

                    if correct_initial_variance
                        mh1 += logpdf(
                            MvNormal(Xbeta[:, j], 1.0 / (1.0 - rho_star^2) * PDMat(Sigma[j], Sigma_chol[j])),
                            eta[:, j, 1],
                        )
                        mh2 += logpdf(
                            MvNormal(Xbeta[:, j], 1.0 / (1.0 - rho[j]^2) * PDMat(Sigma[j], Sigma_chol[j])),
                            eta[:, j, 1],
                        )
                    end # no rho in the first eta when initial variance isn't corrected

                    mh = exp(mh1 - mh2)
                    if mh > rand(Uniform(0, 1))
                        rho[j] = rho_star
                        if k <= params["n_adapt"]
                            rho_accept_batch[j] += 1.0 / 50.0
                        else
                            rho_accept[j] += 1.0 / params["n_mcmc"]
                        end
                    end
                end
            end
        end

        # adapt the tuning for rho
        if k <= params["n_adapt"]
            if (mod(k, 50) == 0)
                out_tuning = update_tuning_vec(k, rho_accept_batch, rho_tune)
                rho_accept_batch = out_tuning["accept"]
                rho_tune = out_tuning["tune"]
            end
        end

        #
        # Sample sigma
        #

        if (sample_sigma)
            Threads.@threads for j in 1:(J-1)
                sigma2_star = exp(rand(Normal(log(sigma2[j]), sigma_tune[j])))
                sigma_star = sqrt(sigma2_star)
                Sigma_star = Matrix(Hermitian(tau[j]^2 * R[j] + sigma_star^2 * I))
                Sigma_chol_star = try
                    cholesky(Sigma_star)
                catch
                    println("theta[:,j] = ", theta[:, j], " sigma[j] = ", sigma[j], "sigma_star = ", sigma_star, "tau[j] = ", tau[j])
                    flush(stdout)
                    @warn "The Covariance matrix for updating sigma2 has been mildly regularized. If this warning is rare, it should be ok to ignore it."
                    cholesky(Matrix(Hermitian(Sigma_star + 1e-6 * I)))
                end

                mh1 =
                    sum([
                        logpdf(
                            MvNormal(
                                (1 - rho[j]) * Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma_star, Sigma_chol_star),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(
                        InverseGamma(priors["alpha_sigma"], priors["beta_sigma"]),
                        sigma2_star,
                    ) +
                    log(sigma2_star)

                mh2 =
                    sum([
                        logpdf(
                            MvNormal(
                                (1 - rho[j]) * Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(
                        InverseGamma(priors["alpha_sigma"], priors["beta_sigma"]),
                        sigma2[j],
                    ) +
                    log(sigma2[j])

                if correct_initial_variance
                    mh1 += logpdf(
                        MvNormal(Xbeta[:, j], 1.0 / (1.0 - rho[j]^2) * PDMat(Sigma_star, Sigma_chol_star)),
                        eta[:, j, 1],
                    )
                    mh2 += logpdf(
                        MvNormal(Xbeta[:, j], 1.0 / (1.0 - rho[j]^2) * PDMat(Sigma[j], Sigma_chol[j])),
                        eta[:, j, 1],
                    )
                else
                    mh1 += logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma_star, Sigma_chol_star)),
                        eta[:, j, 1],
                    )
                    mh2 += logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma[j], Sigma_chol[j])),
                        eta[:, j, 1],
                    )
                end

                mh = exp(mh1 - mh2)
                if mh > rand(Uniform(0, 1))
                    sigma2[j] = sigma2_star
                    sigma[j] = sigma_star
                    Sigma[j] = Sigma_star
                    Sigma_chol[j] = Sigma_chol_star
                    Sigma_inv[j] = inv(Sigma_chol_star)
                    if k <= params["n_adapt"]
                        sigma_accept_batch[j] += 1.0 / 50.0
                    else
                        sigma_accept[j] += 1.0 / params["n_mcmc"]
                    end
                end
            end
        end

        # adapt the tuning for sigma
        if k <= params["n_adapt"]
            if (mod(k, 50) == 0)
                out_tuning = update_tuning_vec(k, sigma_accept_batch, sigma_tune)
                sigma_accept_batch = out_tuning["accept"]
                sigma_tune = out_tuning["tune"]
            end
        end

        #
        # Save MCMC parameters
        #

        if save_full
            if ((k >= checkpoints[checkpoint_idx]) & (k < checkpoints[checkpoint_idx+1])) | (k == (params["n_adapt"] + params["n_mcmc"]))
                save_idx = k - checkpoints[checkpoint_idx] + 1
                k_vec[save_idx] = k
                beta_save[save_idx, :, :, :] = beta
                eta_save[save_idx, :, :, :] = eta
                theta_save[save_idx, :, :] = theta'
                tau_save[save_idx, :] = tau
                sigma_save[save_idx, :] = sigma
                rho_save[save_idx, :] = rho
                omega_save[save_idx, :, :, :] = omega
                for t = 1:n_time
                    pi_save[save_idx, :, :, t] =
                        reduce(hcat, map(eta_to_pi, eachrow(eta[:, :, t])))'
                end
            end
        else
            if (k > params["n_adapt"])
                if ((k >= checkpoints[checkpoint_idx]) & (k < checkpoints[checkpoint_idx+1])) | (k == (params["n_adapt"] + params["n_mcmc"]))
                    if mod(k, params["n_thin"]) == 0
                        save_idx = mod(Int((k - params["n_adapt"]) / params["n_thin"]), Int(params["n_save"] / params["n_thin"]))
                        if save_idx == 0
                            save_idx = Int(params["n_save"] / params["n_thin"])
                        end
                        k_vec[save_idx] = k
                        beta_save[save_idx, :, :, :] = beta
                        eta_save[save_idx, :, :, :] = eta
                        theta_save[save_idx, :, :] = theta'
                        tau_save[save_idx, :] = tau
                        sigma_save[save_idx, :] = sigma
                        rho_save[save_idx, :] = rho
                        omega_save[save_idx, :, :, :] = omega
                        for t = 1:n_time
                            pi_save[save_idx, :, :, t] =
                                reduce(hcat, map(eta_to_pi, eachrow(eta[:, :, t])))'
                        end
                    end
                end
            end
        end

        #
        # save as file
        #
        if save_full
            if k == (checkpoints[checkpoint_idx+1] - 1)
                append!(out["k"], k_vec)
                append!(out["checkpoint_idx"], checkpoint_idx)
                out["beta"][k_vec, :, :] = beta_save
                out["eta"][k_vec, :, :, :] = eta_save
                out["omega"][k_vec, :, :, :] = omega_save
                out["theta"][k_vec, :, :] = theta_save
                out["tau"][k_vec, :] = tau_save
                out["sigma"][k_vec, :] = sigma_save
                out["rho"][k_vec, :] = rho_save
                out["pi"][k_vec, :, :, :] = pi_save
                # tuning variables
                out["rho_accept"] = rho_accept
                out["rho_tune"] = rho_tune
                out["sigma_accept"] = sigma_accept
                out["sigma_tune"] = sigma_tune
                out["tau_accept"] = tau_accept
                out["tau_tune"] = tau_tune
                out["theta_accept"] = theta_accept
                out["lambda_theta"] = lambda_theta
                out["Sigma_theta_tune"] = Sigma_theta_tune
                out["Sigma_theta_tune_chol"] = Sigma_theta_tune_chol

                toc = now()
                out["runtime"] += Int(Dates.value(toc - tic))
                tic = now()
                save(path, out)
                if (k !== (params["n_adapt"] + params["n_mcmc"]))
                    checkpoint_idx += 1
                end
            end
        else
            if k == (checkpoints[checkpoint_idx+1] - 1)
                if (k > params["n_adapt"])
                    # add the save variables
                    save_idx = Int.((k_vec .- params["n_adapt"]) ./ params["n_thin"])
                    out["beta"][save_idx, :, :] = beta_save
                    out["eta"][save_idx, :, :, :] = eta_save
                    out["omega"][save_idx, :, :, :] = omega_save
                    out["theta"][save_idx, :, :] = theta_save
                    out["tau"][save_idx, :] = tau_save
                    out["sigma"][save_idx, :] = sigma_save
                    out["rho"][save_idx, :] = rho_save
                    out["pi"][save_idx, :, :, :] = pi_save
                end
                out["beta_init"] = beta
                out["eta_init"] = eta
                out["omega_init"] = omega
                out["theta_init"] = theta
                out["tau_init"] = tau
                out["sigma_init"] = sigma
                out["rho_init"] = rho
                out["k"] = k
                append!(out["checkpoint_idx"], checkpoint_idx)
                out["rho_accept"] = rho_accept
                out["rho_tune"] = rho_tune
                out["tau_accept"] = tau_accept
                out["tau_tune"] = tau_tune
                out["sigma_accept"] = sigma_accept
                out["sigma_tune"] = sigma_tune
                out["theta_accept"] = theta_accept
                out["lambda_theta"] = lambda_theta
                out["Sigma_theta_tune"] = Sigma_theta_tune
                out["Sigma_theta_tune_chol"] = Sigma_theta_tune_chol
                toc = now()
                out["runtime"] += Int(Dates.value(toc - tic))
                tic = now()
                save(path, out)
                if (k !== (params["n_adapt"] + params["n_mcmc"]))
                    checkpoint_idx += 1
                end
            end
        end
    end

    #
    # end MCMC loop
    #

    toc = now()

    if !save_full
        delete!(out, "eta_init")
        delete!(out, "beta_init")
        delete!(out, "Sigma_theta_tune_chol")
        delete!(out, "Sigma_theta_tune")
        delete!(out, "rho_init")
        delete!(out, "rho_tune")
        delete!(out, "rho_accept")
        delete!(out, "omega_init")
        delete!(out, "tau_tune")
        delete!(out, "tau_accept")
        delete!(out, "sigma_tune")
        delete!(out, "sigma_accept")
        delete!(out, "k")
        delete!(out, "tau_init")
        delete!(out, "sigma_init")
        delete!(out, "lambda_theta")
        delete!(out, "checkpoint_idx")
        delete!(out, "theta_accept")
        delete!(out, "theta_init")
    end

    return (out)

end
