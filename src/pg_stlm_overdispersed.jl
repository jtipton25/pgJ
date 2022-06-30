using Random
using Distributions
using LinearAlgebra
using PDMats
using Dates
#include("polyagamma.jl")

# using PolyaGammaSamplers;

# function pg_stlm(Y, X, locs, params, priors, n_cores)

function pg_stlm_overdispersed(Y, X, locs, params, priors)

    tic = now()
    
    # check input (TODO)
    # check params (TODO)    

    N = size(Y, 1)
    J = size(Y, 2)
    n_time = size(Y, 3)
    p = size(X, 2)

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

    Sigma_beta_chol = cholesky(Sigma_beta)
    Sigma_beta_inv = inv(Sigma_beta_chol.U)
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
    # TODO: check if initial values are supplied

    # initialize Xbeta
    Xbeta = X * beta

    # initialize sigma
    sigma = rand(Gamma(priors["alpha_sigma"], priors["beta_sigma"]), J-1)
    sigma[sigma .> 5] .= 5

    # initialize theta (log-scale)
    # TODO add in Matern priors
    theta_mean = priors["mean_range"]
    theta_var = priors["sd_range"]^2

    theta = rand(Normal(theta_mean, sqrt(theta_var)), J - 1)
    theta[theta .< -2] .= -2
    theta[theta .> 0.1] .= 0.1
    # TODO: check if initial values are supplied

    # initilaize tau
    tau = rand(InverseGamma(params["alpha_tau"], params["beta_tau"]), J - 1)
    tau[tau .> 10] .= 10

    # TODO: check if initial values are supplied

    # initialize rho
    rho = rand(Uniform(0, 1), J - 1)

    # TODO: check if initial values are supplied


    # setup the GP covariance
    # TODO: setup Matern covariance
    D = pairwise(Euclidean(), locs, locs, dims = 1)
    I = Diagonal(ones(N))


    R = [exp.(-D / exp(v)) for v in theta]
    Sigma = [tau[j]^2 * R[j] + sigma[j]^2 * I for j = 1:(J-1)]
    Sigma_chol = [cholesky(v) for v in Sigma]
    Sigma_inv = [inv(v) for v in Sigma_chol]


    # initialize eta
    eta = Array{Float64}(undef, (N, J - 1, n_time))
    for j = 1:(J-1)
        eta[:, j, 1] =
            Xbeta[:, j] + rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol[j])), 1)
        for t = 2:n_time
            eta[:, j, t] =
                Xbeta[:, j] +
                rand(MvNormal(rho[j] * eta[:, j, t-1], PDMat(Sigma[j], Sigma_chol[j])), 1)
        end
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

    # TODO: check if initial values for omega are supplied



    # TODO: setup config
    sample_beta = true
    sample_omega = true
    sample_rho = true
    sample_tau = true
    sample_sigma = true
    sample_theta = true
    sample_eta = true
    save_omega = false

    # setup save variables
    # TODO: work on adding in Matern parameters
    n_save = div(params["n_mcmc"], params["n_thin"])
    beta_save = Array{Float64}(undef, (n_save, p, J - 1))
    tau_save = Array{Float64}(undef, (n_save, J - 1))
    sigma_save = Array{Float64}(undef, (n_save, J - 1))
    rho_save = Array{Float64}(undef, (n_save, J - 1))
    theta_save = Array{Float64}(undef, (n_save, J - 1))
    eta_save = Array{Float64}(undef, (n_save, N, J - 1, n_time))
    pi_save = Array{Float64}(undef, (n_save, N, J, n_time))
    if (save_omega)
        omega_save = Array{Float64}(undef, (n_save, N, J - 1, n_time))
    end

    #
    # MCMC tuning
    #

    # tuning for theta
    # TODO: add in Matern
    theta_accept = zeros(J - 1)
    theta_accept_batch = zeros(J - 1)
#    theta_batch = Array{Float64}(undef, 50, J-1)
    theta_tune = 0.5 * mean(D) * ones(J - 1)

    # tuning for rho
    rho_accept = zeros(J - 1)
    rho_accept_batch = zeros(J - 1)
    rho_tune = 0.025 * ones(J - 1)

    # tuning for tau
    tau_accept = zeros(J - 1)
    tau_accept_batch = zeros(J - 1)
    tau_tune = 0.5 * ones(J - 1)
    
    # tuning for sigma
    sigma_accept = zeros(J - 1)
    sigma_accept_batch = zeros(J - 1)
    sigma_tune = 0.5 * ones(J - 1)


    println(
        "Starting MCMC. Running for ",
        params["n_adapt"],
        " adaptive iterations and ",
        params["n_mcmc"],
        " fitting iterations",
    )
    flush(stdout)

    # MCMC loop
    for k = 1:(params["n_adapt"]+params["n_mcmc"])

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
            omega[nonzero_idx] = ThreadsX.collect(rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i = 1:n_nonzero)
        end

        #
        # sample beta
        #

        if (sample_beta)
            for j = 1:(J-1)
                tXSigma_inv = X' * Sigma_inv[j]
                A = n_time * tXSigma_inv * X + Sigma_beta_inv
                b = dropdims(
                    sum(rho[j] * tXSigma_inv * eta[:, j, :], dims = 2) + Sigma_beta_inv_mu_beta,
                    dims = 2,
                )
                beta[:, j] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
            end
        end

        # update Xbeta
        Xbeta = X * beta

        #
        # sample theta
        #

        # TODO: add in Matern
        if (sample_theta)
            for j = 1:(J-1)
                theta_star = rand(Normal(theta[j], theta_tune[j]))
                R_star = exp.(-D ./ exp.(theta_star))
                Sigma_star = tau[j]^2 * R_star + sigma[j]^2 * I
                Sigma_chol_star = cholesky(Sigma_star)                

                mh1 =
                    logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma_star, Sigma_chol_star)),
                        eta[:, j, 1],
                    ) +
                    sum([
                        logpdf(
                            MvNormal(
                                Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma_star, Sigma_chol_star),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(Normal(theta_mean, sqrt(theta_var)), theta_star)

                mh2 =
                    logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma[j], Sigma_chol[j])),
                        eta[:, j, 1],
                    ) +
                    sum([
                        logpdf(
                            MvNormal(
                                Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(Normal(theta_mean, sqrt(theta_var)), theta[j])

                mh = exp(mh1 - mh2)
                if mh > rand(Uniform(0, 1))
                    theta[j] = theta_star[1]
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
            end
        end

        # adapt the tuning for theta
        if k <= params["n_adapt"]
#            save_idx = mod(k, 50)
#            if (mod(k, 50) == 0)
#                save_idx = 50
#            end
#            theta_batch[save_idx, :] = theta
            if (mod(k, 50) == 0)
                out_tuning = update_tuning_vec(k, theta_accept_batch, theta_tune)
                theta_accept_batch = out_tuning["accept"]
                theta_tune = out_tuning["tune"]
            end
        end

        #
        # Sample tau
        #

        if (sample_tau)
            for j = 1:(J-1)

                tau_star = exp(rand(Normal(log(tau[j]), tau_tune[j])))
                Sigma_star = tau_star^2 * R[j] + sigma[j]^2 * I
                Sigma_chol_star = cholesky(Sigma_star)                

                mh1 =
                    logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma_star, Sigma_chol_star)),
                        eta[:, j, 1],
                    ) +
                    sum([
                        logpdf(
                            MvNormal(
                                Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma_star, Sigma_chol_star),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(InverseGamma(priors["alpha_tau"], priors["beta_tau"]), tau_star) +
                    log(tau_star)

                mh2 =
                    logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma[j], Sigma_chol[j])),
                        eta[:, j, 1],
                    ) +
                    sum([
                        logpdf(
                            MvNormal(
                                Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(InverseGamma(priors["alpha_tau"], priors["beta_tau"]), tau[j]) +
                    log(tau[j])

                mh = exp(mh1 - mh2)
                if mh > rand(Uniform(0, 1))
                    tau[j] = tau_star[1]
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
#            save_idx = mod(k, 50)
#            if (mod(k, 50) == 0)
#                save_idx = 50
#            end
#            tau_batch[save_idx, :] = tau
            if (mod(k, 50) == 0)
                out_tuning = update_tuning_vec(k, tau_accept_batch, tau_tune)
                tau_accept_batch = out_tuning["accept"]
                tau_tune = out_tuning["tune"]
            end
        end

        #
        # Sample sigma
        #

        if (sample_sigma)
            for j = 1:(J-1)

                sigma_star = exp(rand(Normal(log(sigma[j]), sigma_tune[j])))
                Sigma_star = tau[j]^2 * R[j] + sigma_star^2 * I
                Sigma_chol_star = cholesky(Sigma_star)                

                mh1 =
                    logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma_star, Sigma_chol_star)),
                        eta[:, j, 1],
                    ) +
                    sum([
                        logpdf(
                            MvNormal(
                                Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma_star, Sigma_chol_star),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(InverseGamma(priors["alpha_sigma"], priors["beta_sigma"]), sigma_star) +
                    log(sigma_star)

                mh2 =
                    logpdf(
                        MvNormal(Xbeta[:, j], PDMat(Sigma[j], Sigma_chol[j])),
                        eta[:, j, 1],
                    ) +
                    sum([
                        logpdf(
                            MvNormal(
                                Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(InverseGamma(priors["alpha_sigma"], priors["beta_sigma"]), sigma[j]) +
                    log(sigma[j])

                mh = exp(mh1 - mh2)
                if mh > rand(Uniform(0, 1))
                    sigma[j] = sigma_star[1]
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
#            save_idx = mod(k, 50)
#            if (mod(k, 50) == 0)
#                save_idx = 50
#            end
#            sigma_batch[save_idx, :] = sigma
            if (mod(k, 50) == 0)
                out_tuning = update_tuning_vec(k, sigma_accept_batch, sigma_tune)
                sigma_accept_batch = out_tuning["accept"]
                sigma_tune = out_tuning["tune"]
            end
        end


        #
        # sample eta
        #

        if (sample_eta)
            for j = 1:(J-1)

                # initial time
                A = (1.0 + rho[j]^2) * Sigma_inv[j] + Diagonal(omega[:, j, 1])
                b =
                    Sigma_inv[j] *
                    ((1.0 - rho[j] + rho[j]^2) * Xbeta[:, j] + rho[j] * eta[:, j, 2]) +
                    kappa[:, j, 1]
                eta[:, j, 1] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)

                for t = 2:(n_time-1)
                    A = (1.0 + rho[j]^2) * Sigma_inv[j] + Diagonal(omega[:, j, t])
                    b =
                        Sigma_inv[j] * (
                            (1.0 - rho[j])^2 * Xbeta[:, j] +
                            rho[j] * (eta[:, j, t-1] + eta[:, j, t+1])
                        ) + kappa[:, j, t]
                    eta[:, j, t] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
                end

                # final time
                A = Sigma_inv[j] + Diagonal(omega[:, j, n_time])
                b =
                    Sigma_inv[j] *
                    ((1.0 - rho[j]) * Xbeta[:, j] + rho[j] * eta[:, j, n_time-1]) +
                    kappa[:, j, n_time]
                eta[:, j, n_time] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)

            end
        end

        #
        # sample rho
        #

        if (sample_rho)
            for j = 1:(J-1)
                rho_star = rand(Normal(rho[j], rho_tune[j]))

                mh1 =
                    sum([
                        logpdf(
                            MvNormal(
                                Xbeta[:, j] + rho_star * eta[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) + logpdf(Beta(priors["alpha_rho"], priors["beta_rho"]), rho_star)[1]

                mh2 =
                    sum([
                        logpdf(
                            MvNormal(
                                Xbeta[:, j] + rho[j] * eta[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            eta[:, j, t],
                        ) for t = 2:n_time
                    ]) + logpdf(Beta(priors["alpha_rho"], priors["beta_rho"]), rho[j])[1]

                mh = exp(mh1 - mh2)
                if mh > rand(Uniform(0, 1))
                    rho[j] = rho_star[1]
                    if k <= params["n_adapt"]
                        rho_accept_batch[j] += 1.0 / 50.0
                    else
                        rho_accept[j] += 1.0 / params["n_mcmc"]
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
        # Save MCMC parameters
        #

        if (k > params["n_adapt"])
            if (mod(k, params["n_thin"]) == 0)
                save_idx = div(k - params["n_adapt"], params["n_thin"])
                beta_save[save_idx, :, :, :] = beta
                eta_save[save_idx, :, :, :] = eta
                theta_save[save_idx, :] = theta
                tau_save[save_idx, :] = tau 
                sigma_save[save_idx, :] = sigma
                rho_save[save_idx, :] = rho
                if (save_omega)
                    omega_save[save_idx, :, :] = omega
                end
                for t = 1:n_time
                    pi_save[save_idx, :, :, t] =
                        reduce(hcat, map(eta_to_pi, eachrow(eta[:, :, t])))'
                end
            end
        end
    end

    #
    # end MCMC loop
    #

    toc = now()

    if (save_omega)
        out = Dict(
            "beta" => beta_save,
            "eta" => eta_save,
            "omega" => omega_save,
            "theta" => theta_save,
            "tau" => tau_save,
            "sigma" => sigma_save,
            "pi" => pi_save,
            "rho" => rho_save,
            "runtime" => toc - tic
        )
    else
        out = Dict(
            "beta" => beta_save,
            "eta" => eta_save,
            "theta" => theta_save,
            "tau" => tau_save,
            "sigma" => sigma_save,
            "pi" => pi_save,
            "rho" => rho_save,
            "runtime" => toc - tic
        )
    end


    #    # convert Dict to DataFrame
    #    out_df = DataFrame()
    #    # add betas to out_df
    #    for i in 1:p
    #    	for j in 1:(J-1)
    #	  out_df[!, "beta[$i,$j]"] = beta_save[:, i, j]
    #	end
    #    end
    #    # add etas to out_df
    #    for i in 1:N
    #    	for j in 1:(J-1)
    #          out_df[!, "eta[$i,$j]"] = eta_save[:, i, j]
    #	end
    #    end
    #    if (save_omega)
    #        # add omega to out_df
    #        for i in 1:N
    #            for j in 1:(J-1)
    #                out_df[!, "omega[$i,$j]"] = omega_save[:, i, j]
    #	    end
    #	end
    #    end
    #
    #    return(out_df)

    return (out)

end
