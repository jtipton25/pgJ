export subset_posterior

"""
    subset_posterior(out, model)

Returns the subset posterior of `out` for the MCMC model `model` for predictions
"""
function subset_posterior(out, model)
    @assert (model == "matern") | (model == "overdispersed") | (model == "latent") "model must equal \"matern\", \"overdispersed\", or \"latent\", "
    n_adapt = out["params"]["n_adapt"]
    n_mcmc = out["params"]["n_mcmc"]
    n_thin = out["params"]["n_thin"]
    keepers = (n_adapt+1):n_thin:(n_adapt+n_mcmc)

    # filter out the parameters    
    setindex!(out, out["beta"][keepers, :, :], "beta")
    setindex!(out, out["theta"][keepers, :, :], "theta")
    setindex!(out, out["tau"][keepers, :], "tau")
    setindex!(out, out["eta"][keepers, :, :, :], "eta")
    setindex!(out, out["pi"][keepers, :, :, :], "pi")
    setindex!(out, out["rho"][keepers, :], "rho")
        # trim off the variables not needed outside the MCMC
    delete!(out, "omega")
    delete!(out, "k")
    delete!(out, "runtime")
    delete!(out, "checkpoint_idx")
    delete!(out, "lambda_theta")
    delete!(out, "Sigma_theta_tune")
    delete!(out, "Sigma_theta_tune_chol")
    delete!(out, "theta_accept")
    delete!(out, "rho_accept")
    delete!(out, "rho_tune")

    if (model == "overdispersed")
        setindex!(out, out["sigma"][keepers, :], "sigma")
        delete!(out, "sigma_accept")
        delete!(out, "sigma_tune")      
        delete!(out, "tau_accept")
        delete!(out, "tau_tune")      
    end
    if (model == "latent")
        setindex!(out, out["sigma"][keepers, :], "sigma")
        setindex!(out, out["psi"][keepers, :, :, :], "psi")
    end

    return out
end