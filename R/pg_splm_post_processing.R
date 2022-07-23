# install.packages("BiocManager")
# BiocManager::install("rhdf5")
library(rhdf5)
library(tidyverse)
library(patchwork)

plot_eta_sim <- function(dat_sim) {
    eta <- dat_sim$eta
    dims <- dim(eta)
    dimnames(eta) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2])
    dat_plot <- as.data.frame.table(eta, responseName = "eta") 
    dat_plot$x = locs[, 1]
    dat_plot$y = locs[, 2]  
    
    ggplot(dat_plot, aes(x = x, y = y, fill = eta)) +
        geom_raster() +
        facet_wrap(~ species) +
        scale_fill_viridis_c()
}

plot_psi_sim <- function(dat_sim) {
    psi <- dat_sim$psi
    dims <- dim(psi)
    dimnames(psi) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2])
    dat_plot <- as.data.frame.table(psi, responseName = "psi") 
    dat_plot$x = locs[, 1]
    dat_plot$y = locs[, 2]  
    
    ggplot(dat_plot, aes(x = x, y = y, fill = psi)) +
        geom_raster() +
        facet_wrap(~ species) +
        scale_fill_viridis_c()
}

plot_Xbeta_sim <- function(dat_sim) {
    Xbeta <- dat_sim$X %*% dat_sim$beta
    dims <- dim(Xbeta)
    dimnames(Xbeta) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2])
    dat_plot <- as.data.frame.table(Xbeta, responseName = "Xbeta") 
    dat_plot$x = locs[, 1]
    dat_plot$y = locs[, 2]  
    
    ggplot(dat_plot, aes(x = x, y = y, fill = Xbeta)) +
        geom_raster() +
        facet_wrap(~ species) +
        scale_fill_viridis_c()
    }

plot_pi_sim <- function(dat_sim) {
    pi <- dat_sim$pi
    dims <- dim(pi)
    dimnames(pi) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2])
    dat_plot <- as.data.frame.table(pi, responseName = "pi") 
    dat_plot$x = locs[, 1]
    dat_plot$y = locs[, 2]  

    ggplot(dat_plot, aes(x = x, y = y, fill = pi)) +
        geom_raster() +
        facet_wrap(~ species) +
        scale_fill_viridis_c()
}

plot_eta <- function(dat_sim, fit_sim) {
    
    eta <- dat_sim$eta
    dims <- dim(eta)
    dimnames(eta) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2])
    dat_true <- as.data.frame.table(eta, responseName = "eta") 

    eta_fit <- fit_sim$eta
    dims <- dim(eta_fit)
    dimnames(eta_fit) <- list(
        iteration = 1:dims[1],
        locs = 1:dims[2], 
        species = 1:dims[3])
    dat_fit <- as.data.frame.table(eta_fit, responseName = "eta") 

    sites_to_plot = sample(1:dims[1], 50)
    # Plot these in increasing order
    dat_fit_plot <- dat_fit %>%
        filter(locs %in% sites_to_plot) %>%
        group_by(species, locs) %>%
        summarize(mean_eta = mean(eta), 
                  lower_eta_95 = quantile(eta, prob = 0.025), 
                  upper_eta_95 = quantile(eta, prob = 0.975),
                  lower_eta_80 = quantile(eta, prob = 0.1), 
                  upper_eta_80 = quantile(eta, prob = 0.9)) 
    dat_true_plot <- dat_true %>%
        filter(locs %in% sites_to_plot)
    dat_plot <- dat_fit_plot %>% 
        left_join(dat_true_plot, by = c("species", "locs"))
    
    
    dat_plot %>%
        mutate(locs = droplevels(locs)) %>%
        mutate(locs = reorder_within(locs, eta, within = species)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_eta)) + 
        geom_errorbar(aes(x = locs, ymin = lower_eta_95, ymax = upper_eta_95),
                      width = 0, lwd = 0.5) +
        geom_errorbar(aes(x = locs, ymin = lower_eta_80, ymax = upper_eta_80),
                      width = 0, lwd = 0.75) +
        facet_wrap( ~ species, scales = "free") +
        geom_point(aes(x = locs, y = eta), color = "orange", alpha = 0.5) +
        scale_x_reordered()

    # dat_fit %>%
    #     filter(locs %in% sites_to_plot) %>%
    #     group_by(species, locs) %>%
    #     summarize(mean_eta = mean(eta), 
    #               lower_eta = quantile(eta, prob = 0.025), 
    #               upper_eta = quantile(eta, prob = 0.975)) %>%
    #     ggplot() +
    #     geom_point(aes(x = locs, y = mean_eta)) + 
    #     geom_errorbar(aes(x = locs, ymin = lower_eta, ymax = upper_eta)) +
    #     facet_wrap( ~ species) +
    #     geom_point(data = filter(dat_true, locs %in% sites_to_plot),
    #                aes(x = locs, y = eta), color = "orange", alpha = 0.5)
}

plot_pi <- function(dat_sim, fit_sim) {
    
    pi <- dat_sim$pi
    dims <- dim(pi)
    dimnames(pi) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2])
    dat_true <- as.data.frame.table(pi, responseName = "pi") 
    
    pi_fit <- fit_sim$pi
    dims <- dim(pi_fit)
    dimnames(pi_fit) <- list(
        iteration = 1:dims[1],
        locs = 1:dims[2], 
        species = 1:dims[3])
    dat_fit <- as.data.frame.table(pi_fit, responseName = "pi") 
    
    sites_to_plot = sample(1:dims[1], 50)
    # Plot these in increasing order
    dat_fit_plot <- dat_fit %>%
        filter(locs %in% sites_to_plot) %>%
        group_by(species, locs) %>%
        summarize(mean_pi = mean(pi), 
                  lower_pi_95 = quantile(pi, prob = 0.025), 
                  upper_pi_95 = quantile(pi, prob = 0.975),
                  lower_pi_80 = quantile(pi, prob = 0.1), 
                  upper_pi_80 = quantile(pi, prob = 0.9)) 
    dat_true_plot <- dat_true %>%
        filter(locs %in% sites_to_plot)
    dat_plot <- dat_fit_plot %>% 
        left_join(dat_true_plot, by = c("species", "locs"))
    

    dat_plot %>%
        mutate(locs = droplevels(locs)) %>%
        mutate(locs = reorder_within(locs, pi, within = species)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_pi)) + 
        geom_errorbar(aes(x = locs, ymin = lower_pi_95, ymax = upper_pi_95),
                      width = 0, lwd = 0.5) +
        geom_errorbar(aes(x = locs, ymin = lower_pi_80, ymax = upper_pi_80),
                      width = 0, lwd = 0.75) +
        
        facet_wrap( ~ species, scales = "free") +
        geom_point(aes(x = locs, y = pi), color = "orange", alpha = 0.5) +
        scale_x_reordered()
    # dat_fit %>%
    #     filter(locs %in% sites_to_plot) %>%
    #     group_by(species, locs) %>%
    #     summarize(mean_pi = mean(pi), 
    #               lower_pi = quantile(pi, prob = 0.025), 
    #               upper_pi = quantile(pi, prob = 0.975)) %>%
    #     ggplot() +
    #     geom_point(aes(x = locs, y = mean_pi)) + 
    #     geom_errorbar(aes(x = locs, ymin = lower_pi, ymax = upper_pi)) +
    #     facet_wrap( ~ species) +
    #     geom_point(data = filter(dat_true, locs %in% sites_to_plot),
    #                aes(x = locs, y = pi), color = "orange", alpha = 0.5)
}

plot_beta <- function(dat_sim, fit_sim) {
    
    beta <- dat_sim$beta
    dims <- dim(beta)
    dimnames(beta) <- list(
        parameter = 1:dims[1], 
        species = 1:dims[2])
    dat_true <- as.data.frame.table(beta, responseName = "beta") 
    
    beta_fit <- fit_sim$beta
    dims <- dim(beta_fit)
    dimnames(beta_fit) <- list(
        iteration = 1:dims[1],
        parameter = 1:dims[2], 
        species = 1:dims[3])
    dat_fit <- as.data.frame.table(beta_fit, responseName = "beta") 
    
    dat_fit %>%
        group_by(parameter, species) %>%
        summarize(mean_beta = mean(beta), 
                  lower_beta = quantile(beta, prob = 0.025), 
                  upper_beta = quantile(beta, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = parameter, y = mean_beta)) + 
        geom_errorbar(aes(x = parameter, ymin = lower_beta, ymax = upper_beta)) +
        facet_wrap(~ species) +
        geom_point(data = dat_true, aes(x = parameter, y = beta), color = "orange")
}

plot_theta <- function(dat_sim, fit_sim) {
    theta <- dat_sim$theta
    dat_true <- data.frame(theta = theta, species = 1:length(theta))
    
    theta_fit <- fit_sim$theta
    dims <- dim(theta_fit)
    dimnames(theta_fit) <- list(
        iteration = 1:dims[1],
        species = 1:dims[2])
    dat_fit <- as.data.frame.table(theta_fit, responseName = "theta")
    
    dat_fit %>%
        group_by(species) %>%
        summarize(mean_theta = mean(theta), 
                  lower_theta = quantile(theta, prob = 0.025), 
                  upper_theta = quantile(theta, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = species, y = mean_theta)) + 
        geom_errorbar(aes(x = species, ymin = lower_theta, ymax = upper_theta)) +
        geom_point(data = dat_true, aes(x = species, y = theta), color = "orange")
}

plot_tau <- function(dat_sim, fit_sim) {
    tau <- dat_sim$tau
    dat_true <- data.frame(tau = tau, species = 1:length(tau))
    
    tau_fit <- fit_sim$tau
    dims <- dim(tau_fit)
    dimnames(tau_fit) <- list(
        iteration = 1:dims[1],
        species = 1:dims[2])
    dat_fit <- as.data.frame.table(tau_fit, responseName = "tau")
    
    dat_fit %>%
        group_by(species) %>%
        summarize(mean_tau = mean(tau), 
                  lower_tau = quantile(tau, prob = 0.025), 
                  upper_tau = quantile(tau, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = species, y = mean_tau)) + 
        geom_errorbar(aes(x = species, ymin = lower_tau, ymax = upper_tau)) +
        geom_point(data = dat_true, aes(x = species, y = tau), color = "orange")
}



# pg_splm simulation ----

dat_sim <- readRDS("./output/pg_splm_sim_data.RDS")
fit_sim <- readRDS("./output/pg_splm_sim_fit.RDS")

library(tidyverse)

p_eta <- plot_eta_sim(dat_sim)
p_Xbeta <- plot_Xbeta_sim(dat_sim)
p_psi <- plot_psi_sim(dat_sim)
p_eta / p_Xbeta / p_psi
plot_pi_sim(dat_sim)

plot_beta(dat_sim, fit_sim)

plot_theta(dat_sim, fit_sim)

plot_tau(dat_sim, fit_sim)

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
