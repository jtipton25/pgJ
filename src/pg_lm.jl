using Random, Distributions, PolyaGammaSamplers;

# function pg_lm(Y, X, params, priors, n_cores)
function pg_lm(Y, X, params)

    # check input (TODO)
    # check params (TODO)

    N = size(Y, 1)
    J = size(Y, 2)
    p = size(X, 2)

    tX = X'

    Mi = [calc_Mi(v) for v in eachrow(Y)]
    Mi = reduce(hcat, Mi)'

    missing_idx = [any(ismissing.(v)) for v in eachrow(Y)]

    print(
        "There are ",
        ifelse(sum(missing_idx) > 0, sum(missing_idx), "no"),
        " observations with missing count vectors \n",
    )

    nonzero_idx = [v .!= 0 for v in eachrow(Mi)]
    nonzero_idx = reduce(hcat, nonzero_idx)'
    n_nonzero = sum(nonzero_idx)
    kappa = [calc_kappa(Y[i, :], Mi[i, :]) for i = 1:N]
    kappa = reduce(hcat, kappa)'
    tXkappa = X' * kappa

    # default priors
    mu_beta = zeros(p)
    Sigma_beta = Diagonal(10.0 .* ones(p))

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

    # initialize eta
    eta = X * beta

    # initialize omega

    omega = zeros(N, J - 1)
    Mi_nonzero = Mi[nonzero_idx]
    eta_nonzero = eta[nonzero_idx]
    omega[nonzero_idx] =
        [rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i = 1:n_nonzero]

    # TODO: check if initial values for omega are supplied


    # TODO: setup config
    sample_beta = true
    sample_omega = true
    save_omega = false

    # setup save variables
    n_save = div(params["n_mcmc"], params["n_thin"])
    beta_save = Array{Float64}(undef, (n_save, p, J - 1))
    eta_save = Array{Float64}(undef, (n_save, N, J - 1))
    if (save_omega)
        omega_save = Array{Float64}(undef, (n_save, N, J - 1))
    end


    println(
        "Starting MCMC. Running for ",
        params["n_adapt"],
        " adaptive iterations and ",
        params["n_mcmc"],
        " fitting iterations",
    )

    # MCMC loop
    for k = 1:(params["n_adapt"]+params["n_mcmc"])

        if (k == params["n_adapt"] + 1)
            println("Starting MCMC fitting. Running for ", params["n_mcmc"], " iterations")
        end
        if (mod(k, params["n_message"]) == 0)
            if (k <= params["n_adapt"])
                println("MCMC adaptation iteration ", k, " out of ", params["n_adapt"])
            else
                println(
                    "MCMC fitting iteration ",
                    k - params["n_adapt"],
                    " out of ",
                    params["n_mcmc"],
                )
            end
        end

        #
        # sample omega
        #

        if (sample_omega)
            eta_nonzero = eta[nonzero_idx]
            omega[nonzero_idx] =
                [rand(PolyaGamma(Mi_nonzero[i], eta_nonzero[i])) for i = 1:n_nonzero]
        end

        #
        # sample beta
        #

        if (sample_beta)
            for j = 1:(J-1)
                A = Sigma_beta_inv + tX * (omega[:, j] .* X)
                # A = Sigma_beta_inv + tX * (Diagonal(omega[:, j]) * X)
                b = Sigma_beta_inv_mu_beta + tXkappa[:, j]
                beta[:, j] = rand(MvNormalCanon(b, PDMat(Matrix(Hermitian(A)))), 1)
            end
        end

        # update eta
        eta = X * beta

        if (k > params["n_adapt"])
            if (mod(k, params["n_thin"]) == 0)
                save_idx = div(k - params["n_adapt"], params["n_thin"])
                beta_save[save_idx, :, :] = beta
                eta_save[save_idx, :, :] = eta
                if (save_omega)
                    omega_save[save_idx, :, :] = omega
                end
            end
        end


    end # end MCMC loop



    #    if (save_omega)
    #       out = Dict("beta" => beta_save, "eta" => eta_save, "omega" => omega_save)
    #    else
    #       out = Dict("beta" => beta_save, "eta" => eta_save)
    #    end


    # convert Dict to DataFrame
    out_df = DataFrame()
    # add betas to out_df
    for i = 1:p
        for j = 1:(J-1)
            out_df[!, "beta[$i,$j]"] = beta_save[:, i, j]
        end
    end
    # add etas to out_df
    for i = 1:N
        for j = 1:(J-1)
            out_df[!, "eta[$i,$j]"] = eta_save[:, i, j]
        end
    end
    if (save_omega)
        # add omega to out_df
        for i = 1:N
            for j = 1:(J-1)
                out_df[!, "omega[$i,$j]"] = omega_save[:, i, j]
            end
        end
    end

    return (out_df)

end
