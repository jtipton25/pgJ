# function pg_stlm(Y, X, locs, params, priors, n_cores)

export pg_stlm_latent

"""
    pg_stlm_latent(Y, X, locs, params, priors)

Return the MCMC output for a linear model with A 2-D Array of observations of Ints `Y` (with `missing` values), a 2-D Array of covariates `X`, A 2-D Array of locations `locs`, a `Dict` of model parameters `params`, and a `Dict` of prior parameter values `priors`
"""
function pg_stlm_latent(Y, X, locs, params, priors)

    tic = now()

    # check input (TODO)
    # check params (TODO)    

    N = size(Y, 1)
    J = size(Y, 2)
    n_time = size(Y, 3)
    p = size(X, 2)

    tX = X'
    tXX = tX * X

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
    sigma = rand(Gamma(priors["alpha_sigma"], priors["beta_sigma"]), J - 1)
    sigma[sigma.>5] .= 5

    # initialize theta (log-scale)
    # TODO add in Matern priors
    theta_mean = priors["mean_range"]
    theta_var = priors["sd_range"]^2

    theta = rand(Normal(theta_mean, sqrt(theta_var)), J - 1)
    theta[theta.<-2] .= -2
    theta[theta.>0.1] .= 0.1
    # TODO: check if initial values are supplied

    # initilaize tau
    tau = rand(InverseGamma(params["alpha_tau"], params["beta_tau"]), J - 1)
    tau[tau.>10] .= 10

    # TODO: check if initial values are supplied

    # initialize rho
    rho = rand(Uniform(0, 1), J - 1)

    # TODO: check if initial values are supplied


    # setup the GP covariance
    # TODO: setup Matern covariance
    D = pairwise(Euclidean(), locs, locs, dims = 1)
    I = Diagonal(ones(N))


    R = [exp.(-D / exp(v)) for v in theta]
    Sigma = [tau[j]^2 * R[j] for j = 1:(J-1)]
    R_chol = [cholesky(v) for v in R]
    Sigma_chol = copy(R_chol)
    for j = 1:(J-1)
        Sigma_chol[j].U .*= tau[j]
    end

    Sigma_inv = [inv(v) for v in Sigma_chol]

    # initialize psi
    psi = Array{Float64}(undef, (N, J - 1, n_time))
    for j = 1:(J-1)
        psi[:, j, 1] = rand(MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol[j])), 1)
        for t = 2:n_time
            psi[:, j, t] =
                rand(MvNormal(rho[j] * psi[:, j, t-1], PDMat(Sigma[j], Sigma_chol[j])), 1)
        end
    end

    # initialize eta
    eta = Array{Float64}(undef, (N, J - 1, n_time))
    for j = 1:(J-1)
        eta[:, j, 1] = Xbeta[:, j] + psi[:, j, 1] + rand(Normal(0, sigma[j]), N)
        for t = 2:n_time
            eta[:, j, t] = Xbeta[:, j] + psi[:, j, t] + rand(Normal(0, sigma[j]), N)
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
    sample_psi = true
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
    psi_save = Array{Float64}(undef, (n_save, N, J - 1, n_time))
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
            omega[nonzero_idx] = ThreadsX.collect(
                rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i = 1:n_nonzero
            )
        end

        #
        # sample beta
        #

        if (sample_beta)
            for j = 1:(J-1)
                A = n_time * tXX / (sigma[j]^2) + Sigma_beta_inv
                b = dropdims(
                    sum(tX * (eta[:, j, :] - psi[:, j, :]), dims = 2) / (sigma[j]^2) +
                    Sigma_beta_inv_mu_beta,
                    dims = 2,
                )
                beta[:, j] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
            end
        end

        # update Xbeta
        Xbeta = X * beta

        #
        # sample eta
        #

        if (sample_eta)
            Threads.@threads for j = 1:(J-1)
                for t = 1:n_time
                    sigma2_tilde = 1.0 ./ (1.0 / sigma[j]^2 .+ omega[:, j, t])
                    mu_tilde =
                        1.0 / sigma[j]^2 * (Xbeta[:, j] + psi[:, j, t]) + kappa[:, j, t]
                    sigma2_mu_tilde = sigma2_tilde .* mu_tilde
                    eta[:, j, t] = [
                        rand(Normal(sigma2_mu_tilde[i], sqrt(sigma2_tilde[i]))) for i = 1:N
                    ]
                end
            end
        end
        
        #
        # Sample tau
        #

        if (sample_tau)
            for j = 1:(J-1)
                devs = Array{Float64}(undef, (N, n_time))
                devs[:, 1] = psi[:, j, 1]
                for t = 2:n_time
                    devs[:, t] = psi[:, j, t] - rho[j] * psi[:, j, t-1]
                end
                SS = sum([
                    devs[:, t]' * (tau[j]^2 * Sigma_inv[j] * devs[:, t]) for t = 1:n_time
                ])
                tau[j] = sqrt(
                    rand(
                        InverseGamma(
                            0.5 * n_time * N + priors["alpha_tau"],
                            0.5 * SS + priors["beta_tau"],
                        ),
                    ),
                )

                Sigma[j] = tau[j]^2 * R[j]
                Sigma_chol[j] = copy(R_chol[j])
                Sigma_chol[j].U .*= tau[j]
                Sigma_inv[j] = inv(Sigma_chol[j])
            end
        end

        #
        # sample theta
        #

        # TODO: add in Matern
        if (sample_theta)
            for j = 1:(J-1)
                theta_star = rand(Normal(theta[j], theta_tune[j]))
                R_star = exp.(-D / exp(theta_star))
                Sigma_star = tau[j]^2 * R_star
                R_chol_star = cholesky(R_star)
                Sigma_chol_star = copy(R_chol_star)
                Sigma_chol_star.U .*= tau[j]

                mh1 =
                    logpdf(
                        MvNormal(zeros(N), PDMat(Sigma_star, Sigma_chol_star)),
                        psi[:, j, 1],
                    ) +
                    sum([
                        logpdf(
                            MvNormal(
                                rho[j] * psi[:, j, t-1],
                                PDMat(Sigma_star, Sigma_chol_star),
                            ),
                            psi[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(Normal(theta_mean, sqrt(theta_var)), theta_star)

                mh2 =
                    logpdf(
                        MvNormal(zeros(N), PDMat(Sigma[j], Sigma_chol[j])),
                        psi[:, j, 1],
                    ) +
                    sum([
                        logpdf(
                            MvNormal(
                                rho[j] * psi[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            psi[:, j, t],
                        ) for t = 2:n_time
                    ]) +
                    logpdf(Normal(theta_mean, sqrt(theta_var)), theta[j])

                mh = exp(mh1 - mh2)
                if mh > rand(Uniform(0, 1))
                    theta[j] = theta_star
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
            if (mod(k, 50) == 0)
                out_tuning = update_tuning_vec(k, theta_accept_batch, theta_tune)
                theta_accept_batch = out_tuning["accept"]
                theta_tune = out_tuning["tune"]
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
                                rho_star * psi[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            psi[:, j, t],
                        ) for t = 2:n_time
                    ]) + logpdf(Beta(priors["alpha_rho"], priors["beta_rho"]), rho_star)

                mh2 =
                    sum([
                        logpdf(
                            MvNormal(
                                rho[j] * psi[:, j, t-1],
                                PDMat(Sigma[j], Sigma_chol[j]),
                            ),
                            psi[:, j, t],
                        ) for t = 2:n_time
                    ]) + logpdf(Beta(priors["alpha_rho"], priors["beta_rho"]), rho[j])

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
            for j = 1:(J-1)
                SS = sum(
                    sum(
                        [(eta[:, j, t] - Xbeta[:, j] - psi[:, j, t]).^2 for t = 1:n_time]                    
                    )
                )
                sigma[j] = sqrt(
                    rand(
                        InverseGamma(
                            0.5 * n_time * N + priors["alpha_sigma"],
                            0.5 * SS + priors["beta_sigma"],
                        ),
                    ),
                )

            end
        end

        #
        # sample psi
        #

        if (sample_psi)
            for j = 1:(J-1)

                # initial time
                A = (1.0 + rho[j]^2) * Sigma_inv[j] + 1.0 / sigma[j]^2 * I
                b =
                    Sigma_inv[j] * (rho[j] * psi[:, j, 2]) +
                    1.0 / sigma[j]^2 * (eta[:, j, 1] - Xbeta[:, j])
                psi[:, j, 1] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)

                for t = 2:(n_time-1)
                    A = (1.0 + rho[j]^2) * Sigma_inv[j] + 1.0 / sigma[j]^2 * I
                    b =
                        Sigma_inv[j] * rho[j] * (psi[:, j, t-1] + psi[:, j, t+1]) +
                        1.0 / sigma[j]^2 * (eta[:, j, t] - Xbeta[:, j])
                    psi[:, j, t] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
                end

                # final time
                A = Sigma_inv[j] + 1.0 / sigma[j]^2 * I
                b = 
                    Sigma_inv[j] * rho[j] * psi[:, j, n_time-1] +
                    1.0 / sigma[j]^2 * (eta[:, j, n_time] - Xbeta[:, j])
                psi[:, j, n_time] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
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
                psi_save[save_idx, :, :, :] = psi
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
            "psi" => psi_save,
            "omega" => omega_save,
            "theta" => theta_save,
            "tau" => tau_save,
            "sigma" => sigma_save,
            "pi" => pi_save,
            "rho" => rho_save,
            "runtime" => Int(Dates.value(toc - tic)) # milliseconds runtime as an Int
        )
    else
        out = Dict(
            "beta" => beta_save,
            "eta" => eta_save,
            "psi" => psi_save,
            "theta" => theta_save,
            "tau" => tau_save,
            "sigma" => sigma_save,
            "pi" => pi_save,
            "rho" => rho_save,
            "runtime" => Int(Dates.value(toc - tic)) # milliseconds runtime as an Int
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
