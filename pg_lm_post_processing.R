# install.packages("BiocManager")
# BiocManager::install("rhdf5")
library(rhdf5)
library(tidyverse)
library(patchwork)

plot_eta <- function(dat_sim, fit_sim) {
    eta <- dat_sim$eta
    dims <- dim(eta)
    dimnames(eta) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2])
    dat_sim_eta <- as.data.frame.table(eta, responseName = "eta")
    
    eta_fit <- fit_sim$eta
    dims <- dim(eta_fit)
    dimnames(eta_fit) <- list(
        iteration = 1:dims[1], 
        locs = 1:dims[2], 
        species = 1:dims[3])
    dat_fit_eta <- as.data.frame.table(eta_fit, responseName = "eta")

    sites_to_plot <- sample(1:dims[1], 30)
    dat_fit_eta %>%
        filter(locs %in% sites_to_plot) %>%
        group_by(species, locs) %>%
        summarize(mean_eta = mean(eta), 
                  lower_eta = quantile(eta, prob = 0.025), 
                  upper_eta = quantile(eta, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_eta)) + 
        geom_errorbar(aes(x = locs, ymin = lower_eta, ymax = upper_eta)) +
        facet_wrap( ~ species, ncol = 1) +
        geom_point(data = filter(dat_sim_eta, locs %in% sites_to_plot),
                   aes(x = locs, y = eta), color = "orange", alpha = 0.5)
}


plot_pi <- function(dat_sim, fit_sim) {
    pi <- dat_sim$pi
    dims <- dim(pi)
    dimnames(pi) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2])
    dat_sim_pi <- as.data.frame.table(pi, responseName = "pi")
    
    pi_fit <- fit_sim$pi
    dims <- dim(pi_fit)
    dimnames(pi_fit) <- list(
        iteration = 1:dims[1], 
        locs = 1:dims[2], 
        species = 1:dims[3])
    dat_fit_pi <- as.data.frame.table(pi_fit, responseName = "pi")
    
    sites_to_plot <- sample(1:dims[1], 30)
    
    dat_fit_pi %>%
        filter(locs %in% sites_to_plot) %>%
        group_by(species, locs) %>%
        summarize(mean_pi = mean(pi), 
                  lower_pi = quantile(pi, prob = 0.025), 
                  upper_pi = quantile(pi, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_pi)) + 
        geom_errorbar(aes(x = locs, ymin = lower_pi, ymax = upper_pi)) +
        facet_wrap( ~ species, ncol = 1) +
        geom_point(data = filter(dat_sim_pi, locs %in% sites_to_plot),
                   aes(x = locs, y = pi), color = "orange", alpha = 0.5)
}


plot_beta <- function(dat_sim, fit_sim) {
    beta <- dat_sim$beta
    dims <- dim(beta)
    dimnames(beta) <- list(
        parameter = 1:dims[1], 
        species = 1:dims[2])
    dat_sim_beta <- as.data.frame.table(beta, responseName = "beta")
    
    beta_fit <- fit_sim$beta
    dims <- dim(beta_fit)
    dimnames(beta_fit) <- list(
        iterations = 1:dims[1],
        parameter = 1:dims[2], 
        species = 1:dims[3])
    dat_fit_beta <- as.data.frame.table(beta_fit, responseName = "beta")
    
    dat_fit_beta %>%
        group_by(species, parameter) %>%
        summarize(mean_beta = mean(beta), 
                  lower_beta = quantile(beta, prob = 0.025), 
                  upper_beta = quantile(beta, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = parameter, y = mean_beta)) + 
        geom_errorbar(aes(x = parameter, ymin = lower_beta, ymax = upper_beta)) +
        facet_wrap( ~ species, ncol = 1) +
        geom_point(data = dat_sim_beta,
                   aes(x = parameter, y = beta), color = "orange", alpha = 0.5)
}


# h5ls("./output/overdispersed_sim.jld")

# load the julia simulation parameters



# matern simulation ----


dat_sim <- readRDS("./output/pg_lm_sim_data.RDS")
fit_sim <- readRDS("./output/pg_lm_sim_fit.RDS")


library(tidyverse)


plot_beta(dat_sim, fit_sim)

plot_eta(dat_sim, fit_sim)

plot_pi(dat_sim, fit_sim)


# load the julia MCMC output for simulated data

# h5ls("./output/overdispersed_sim_data.jld")
# tmp_sim = h5read("./output/overdispersed_sim_data.jld", name="_refs")
# dat_sim = vector(mode='list', length = length(tmp_sim[[1]]))
# names(out) <- tmp_out[[1]]


# h5ls("./output/overdispersed_sim_fit.jld")
# tmp_out = h5read("./output/overdispersed_sim_fit.jld", name="_refs")
# out = vector(mode='list', length = length(tmp_out[[1]]))
# names(out) <- tmp_out[[1]]

# for (i in 1:length(tmp_out[[1]]))  {
#     out[[i]] <- tmp_out[[i+2]]
# }
