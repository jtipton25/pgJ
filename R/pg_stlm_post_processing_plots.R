# Plotting functions for post-processing
library(tidyverse)
library(patchwork)

plot_eta_sim <- function(dat_sim) {
    eta <- dat_sim$eta
    dims <- dim(eta)
    dimnames(eta) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2], 
        time = 1:dims[3])
    dat_plot <- as.data.frame.table(eta, responseName = "eta") 
    dat_plot$x = dat_sim$locs[, 1]
    dat_plot$y = dat_sim$locs[, 2]  
    
    ggplot(dat_plot, aes(x = x, y = y, fill = eta)) +
        geom_raster() +
        facet_grid(time ~ species) +
        scale_fill_viridis_c()
}

plot_psi_sim <- function(dat_sim) {
    psi <- dat_sim$psi
    dims <- dim(psi)
    dimnames(psi) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2], 
        time = 1:dims[3])
    dat_plot <- as.data.frame.table(psi, responseName = "psi") 
    dat_plot$x = dat_sim$locs[, 1]
    dat_plot$y = dat_sim$locs[, 2]  
    
    ggplot(dat_plot, aes(x = x, y = y, fill = psi)) +
        geom_raster() +
        facet_grid(time ~ species) +
        scale_fill_viridis_c()
}

plot_Xbeta_sim <- function(dat_sim) {
    Xbeta <- dat_sim$X %*% dat_sim$beta
    dims <- dim(Xbeta)
    dimnames(Xbeta) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2])
    dat_plot <- as.data.frame.table(Xbeta, responseName = "Xbeta") 
    dat_plot$x = dat_sim$locs[, 1]
    dat_plot$y = dat_sim$locs[, 2]  
    
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
        species = 1:dims[2],
        time = 1:dims[3])
    dat_plot <- as.data.frame.table(pi, responseName = "pi") 
    dat_plot$x = dat_sim$locs[, 1]
    dat_plot$y = dat_sim$locs[, 2]  
    
    ggplot(dat_plot, aes(x = x, y = y, fill = pi)) +
        geom_raster() +
        facet_grid(time~ species) +
        scale_fill_viridis_c()
}

plot_eta <- function(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
                     fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) {
    
    eta_matern <- dat_sim_matern$eta
    dims <- dim(eta_matern)
    dimnames(eta_matern) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2],
        time = 1:dims[3])
    dat_eta_matern <- as.data.frame.table(eta_matern, responseName = "eta") %>% 
        mutate(model = "matern")
    
    eta_overdispersed <- dat_sim_overdispersed$eta
    dims <- dim(eta_overdispersed)
    dimnames(eta_overdispersed) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2],
        time = 1:dims[3])
    dat_eta_overdispersed <- as.data.frame.table(eta_overdispersed, responseName = "eta") %>%
        mutate(model = "overdispersed")
    
    eta_latent <- dat_sim_latent$eta
    dims <- dim(eta_latent)
    dimnames(eta_latent) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2],
        time = 1:dims[3])
    dat_eta_latent <- as.data.frame.table(eta_latent, responseName = "eta") %>%
        mutate(model = "latent")
    
    dat_true <- rbind(dat_eta_matern, dat_eta_overdispersed, dat_eta_latent)
    
    eta_matern <- fit_sim_matern$eta
    dims <- dim(eta_matern)
    dimnames(eta_matern) <- list(
        iteration = 1:dims[1],
        locs = 1:dims[2], 
        species = 1:dims[3],
        time = 1:dims[4])
    dat_eta_matern <- as.data.frame.table(eta_matern, responseName = "eta") %>% 
        mutate(model = "matern")
    
    eta_overdispersed <- fit_sim_overdispersed$eta
    dims <- dim(eta_overdispersed)
    dimnames(eta_overdispersed) <- list(
        iteration = 1:dims[1],
        locs = 1:dims[2], 
        species = 1:dims[3],
        time = 1:dims[4])
    dat_eta_overdispersed <- as.data.frame.table(eta_overdispersed, responseName = "eta") %>%
        mutate(model = "overdispersed")
    
    eta_latent <- fit_sim_latent$eta
    dims <- dim(eta_latent)
    dimnames(eta_latent) <- list(
        iteration = 1:dims[1],
        locs = 1:dims[2], 
        species = 1:dims[3],
        time = 1:dims[4])
    dat_eta_latent <- as.data.frame.table(eta_latent, responseName = "eta") %>%
        mutate(model = "latent")
    
    dat_fit <- rbind(dat_eta_matern, dat_eta_overdispersed, dat_eta_latent)
    
    # dims = dim(dat_sim_matern$eta)
    # iters <- dim(fit_sim_latent$beta)[1]
    # dat_true <- data.frame(model = c("matern", "overdispersed", "latent"))
    # dat_fit <- data.frame(iteration = 1:iters, model = rep(c("matern", "overdispersed", "latent"), each = iters))
    # 
    # # process the output into a data.frame
    # for (j in 1:dims[1]) {
    #     for (p in 1:dims[2]) {
    #         for (k in 1:dims[3]) {
    #             dat_true[[paste0("eta[", j, ", ", p, ", ", k, "]")]] = 
    #                 c(dat_sim_matern$eta[j, p, k], dat_sim_overdispersed$eta[j, p, k], dat_sim_latent$eta[j, p, k])
    #             dat_fit[[paste0("eta[", j, ", ", p, ", ", k, "]")]] = 
    #                 c(fit_sim_matern$eta[, j, p, k], fit_sim_overdispersed$eta[, j, p, k], fit_sim_latent$eta[, j, p, k])
    #         }
    #     }
    # }
    
    sites_to_plot = sample(1:dims[1], 30)
    times_to_plot = sample(1:dims[4], 4)
    p_matern <- dat_fit %>%
        filter(model == "matern") %>%
        filter(locs %in% sites_to_plot) %>%
        filter(time %in% times_to_plot) %>%
        group_by(species, locs, time) %>%
        summarize(mean_eta = mean(eta), 
                  lower_eta = quantile(eta, prob = 0.025), 
                  upper_eta = quantile(eta, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_eta)) + 
        geom_errorbar(aes(x = locs, ymin = lower_eta, ymax = upper_eta)) +
        facet_grid(time ~ species) +
        geom_point(data = filter(dat_true, model == "matern") %>% 
                       filter(locs %in% sites_to_plot) %>%
                       filter(time %in% times_to_plot),
                   aes(x = locs, y = eta), color = "orange", alpha = 0.5) +
        ggtitle("Matern")
    p_overdispersed <- dat_fit %>%
        filter(model == "overdispersed") %>%
        filter(locs %in% sites_to_plot) %>%
        filter(time %in% times_to_plot) %>%
        group_by(species, locs, time) %>%
        summarize(mean_eta = mean(eta), 
                  lower_eta = quantile(eta, prob = 0.025), 
                  upper_eta = quantile(eta, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_eta)) + 
        geom_errorbar(aes(x = locs, ymin = lower_eta, ymax = upper_eta)) +
        facet_grid(time ~ species) +
        geom_point(data = filter(dat_true, model == "overdispersed") %>% 
                       filter(locs %in% sites_to_plot) %>%
                       filter(time %in% times_to_plot),
                   aes(x = locs, y = eta), color = "orange", alpha = 0.5) +
        ggtitle("Overdispersed")
    p_latent <- dat_fit %>%
        filter(model == "latent") %>%
        filter(locs %in% sites_to_plot) %>%
        filter(time %in% times_to_plot) %>%
        group_by(species, locs, time) %>%
        summarize(mean_eta = mean(eta), 
                  lower_eta = quantile(eta, prob = 0.025), 
                  upper_eta = quantile(eta, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_eta)) + 
        geom_errorbar(aes(x = locs, ymin = lower_eta, ymax = upper_eta)) +
        facet_grid(time ~ species) +
        geom_point(data = filter(dat_true, model == "latent") %>% 
                       filter(locs %in% sites_to_plot) %>%
                       filter(time %in% times_to_plot),
                   aes(x = locs, y = eta), color = "orange", alpha = 0.5) +
        ggtitle("Latent")
    
    p_matern / p_overdispersed / p_latent
}


plot_pi <- function(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
                    fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) {
    pi_matern <- dat_sim_matern$pi
    dims <- dim(pi_matern)
    dimnames(pi_matern) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2],
        time = 1:dims[3])
    dat_pi_matern <- as.data.frame.table(pi_matern, responseName = "pi") %>% 
        mutate(model = "matern")
    
    pi_overdispersed <- dat_sim_overdispersed$pi
    dims <- dim(pi_overdispersed)
    dimnames(pi_overdispersed) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2],
        time = 1:dims[3])
    dat_pi_overdispersed <- as.data.frame.table(pi_overdispersed, responseName = "pi") %>%
        mutate(model = "overdispersed")
    
    pi_latent <- dat_sim_latent$pi
    dims <- dim(pi_latent)
    dimnames(pi_latent) <- list(
        locs = 1:dims[1], 
        species = 1:dims[2],
        time = 1:dims[3])
    dat_pi_latent <- as.data.frame.table(pi_latent, responseName = "pi") %>%
        mutate(model = "latent")
    
    dat_true <- rbind(dat_pi_matern, dat_pi_overdispersed, dat_pi_latent)
    
    pi_matern <- fit_sim_matern$pi
    dims <- dim(pi_matern)
    dimnames(pi_matern) <- list(
        iteration = 1:dims[1],
        locs = 1:dims[2], 
        species = 1:dims[3],
        time = 1:dims[4])
    dat_pi_matern <- as.data.frame.table(pi_matern, responseName = "pi") %>% 
        mutate(model = "matern")
    
    pi_overdispersed <- fit_sim_overdispersed$pi
    dims <- dim(pi_overdispersed)
    dimnames(pi_overdispersed) <- list(
        iteration = 1:dims[1],
        locs = 1:dims[2], 
        species = 1:dims[3],
        time = 1:dims[4])
    dat_pi_overdispersed <- as.data.frame.table(pi_overdispersed, responseName = "pi") %>%
        mutate(model = "overdispersed")
    
    pi_latent <- fit_sim_latent$pi
    dims <- dim(pi_latent)
    dimnames(pi_latent) <- list(
        iteration = 1:dims[1],
        locs = 1:dims[2], 
        species = 1:dims[3],
        time = 1:dims[4])
    dat_pi_latent <- as.data.frame.table(pi_latent, responseName = "pi") %>%
        mutate(model = "latent")
    
    dat_fit <- rbind(dat_pi_matern, dat_pi_overdispersed, dat_pi_latent)
    
    sites_to_plot = sample(1:dims[1], 30)
    times_to_plot = sample(1:dims[3], 4)
    p_matern <- dat_fit %>%
        filter(model == "matern") %>%
        filter(locs %in% sites_to_plot) %>%
        filter(time %in% times_to_plot) %>%
        group_by(species, locs, time) %>%
        summarize(mean_pi = mean(pi), 
                  lower_pi = quantile(pi, prob = 0.025), 
                  upper_pi = quantile(pi, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_pi)) + 
        geom_errorbar(aes(x = locs, ymin = lower_pi, ymax = upper_pi)) +
        facet_grid(time ~ species) +
        geom_point(data = filter(dat_true, model == "matern") %>% 
                       filter(locs %in% sites_to_plot) %>%
                       filter(time %in% times_to_plot),
                   aes(x = locs, y = pi), color = "orange", alpha = 0.5) +
        ggtitle("Matern")
    p_overdispersed <- dat_fit %>%
        filter(model == "overdispersed") %>%
        filter(locs %in% sites_to_plot) %>%
        filter(time %in% times_to_plot) %>%
        group_by(species, locs, time) %>%
        summarize(mean_pi = mean(pi), 
                  lower_pi = quantile(pi, prob = 0.025), 
                  upper_pi = quantile(pi, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_pi)) + 
        geom_errorbar(aes(x = locs, ymin = lower_pi, ymax = upper_pi)) +
        facet_grid(time ~ species) +
        geom_point(data = filter(dat_true, model == "overdispersed") %>% 
                       filter(locs %in% sites_to_plot) %>%
                       filter(time %in% times_to_plot),
                   aes(x = locs, y = pi), color = "orange", alpha = 0.5) +
        ggtitle("Overdispersed")
    p_latent <- dat_fit %>%
        filter(model == "latent") %>%
        filter(locs %in% sites_to_plot) %>%
        filter(time %in% times_to_plot) %>%
        group_by(species, locs, time) %>%
        summarize(mean_pi = mean(pi), 
                  lower_pi = quantile(pi, prob = 0.025), 
                  upper_pi = quantile(pi, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = locs, y = mean_pi)) + 
        geom_errorbar(aes(x = locs, ymin = lower_pi, ymax = upper_pi)) +
        facet_grid(time ~ species) +
        geom_point(data = filter(dat_true, model == "latent") %>% 
                       filter(locs %in% sites_to_plot) %>%
                       filter(time %in% times_to_plot),
                   aes(x = locs, y = pi), color = "orange", alpha = 0.5) +
        ggtitle("Latent")
    
    p_matern / p_overdispersed / p_latent
}


plot_beta <- function(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
                      fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) {
    dims = dim(dat_sim_matern$beta)
    iters <- dim(fit_sim_latent$beta)[1]
    dat_true <- data.frame(model = c("matern", "overdispersed", "latent"))
    dat_fit <- data.frame(iteration = 1:iters, model = rep(c("matern", "overdispersed", "latent"), each = iters))
    
    # process the output into a data.frame
    for (j in 1:dims[1]) {
        for (p in 1:dims[2]) {
            dat_true[[paste0("beta[", j, ", ", p, "]")]] = 
                c(dat_sim_matern$beta[j, p], dat_sim_overdispersed$beta[j, p], dat_sim_latent$beta[j, p])
            dat_fit[[paste0("beta[", j, ", ", p, "]")]] = 
                c(fit_sim_matern$beta[, j, p], fit_sim_overdispersed$beta[, j, p], fit_sim_latent$beta[, j, p])
        }
    }
    dat_true <- dat_true %>%
        pivot_longer(cols = starts_with("beta"), names_to = "parameter", values_to = "beta") 
    
    dat_fit %>%
        pivot_longer(cols = starts_with("beta"), names_to = "parameter", values_to = "beta") %>%
        group_by(model, parameter) %>%
        summarize(mean_beta = mean(beta), 
                  lower_beta = quantile(beta, prob = 0.025), 
                  upper_beta = quantile(beta, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = parameter, y = mean_beta)) + 
        geom_errorbar(aes(x = parameter, ymin = lower_beta, ymax = upper_beta)) +
        facet_wrap(~ model) +
        geom_point(data = dat_true, aes(x = parameter, y = beta), color = "orange")
}

plot_theta <- function(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
                       fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) {
    dims = length(dat_sim_matern$theta)
    iters <- dim(fit_sim_latent$theta)[1]
    dat_true <- data.frame(model = c("matern", "overdispersed", "latent"))
    dat_fit <- data.frame(iteration = 1:iters, model = rep(c("matern", "overdispersed", "latent"), each = iters))
    
    # process the output into a data.frame
    for (j in 1:dims) {
        dat_true[[paste0("theta[", j, "]")]] = 
            c(dat_sim_matern$theta[j], dat_sim_overdispersed$theta[j], dat_sim_latent$theta[j])
        # fitted thetas on log scale
        dat_fit[[paste0("theta[", j, "]")]] = 
            exp(c(fit_sim_matern$theta[, j], fit_sim_overdispersed$theta[, j], fit_sim_latent$theta[, j]))
    }
    dat_true <- dat_true %>%
        pivot_longer(cols = starts_with("theta"), names_to = "parameter", values_to = "theta") 
    
    dat_fit %>%
        pivot_longer(cols = starts_with("theta"), names_to = "parameter", values_to = "theta") %>%
        group_by(model, parameter) %>%
        summarize(mean_theta = mean(theta), 
                  lower_theta = quantile(theta, prob = 0.025), 
                  upper_theta = quantile(theta, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = parameter, y = mean_theta)) + 
        geom_errorbar(aes(x = parameter, ymin = lower_theta, ymax = upper_theta)) +
        facet_wrap(~ model) +
        geom_point(data = dat_true, aes(x = parameter, y = theta), color = "orange")
}



plot_rho <- function(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
                     fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) {
    dims = length(dat_sim_matern$rho)
    iters <- dim(fit_sim_latent$rho)[1]
    dat_true <- data.frame(model = c("matern", "overdispersed", "latent"))
    dat_fit <- data.frame(iteration = 1:iters, model = rep(c("matern", "overdispersed", "latent"), each = iters))
    
    # process the output into a data.frame
    for (j in 1:dims) {
        dat_true[[paste0("rho[", j, "]")]] = 
            c(dat_sim_matern$rho[j], dat_sim_overdispersed$rho[j], dat_sim_latent$rho[j])
        dat_fit[[paste0("rho[", j, "]")]] = 
            c(fit_sim_matern$rho[, j], fit_sim_overdispersed$rho[, j], fit_sim_latent$rho[, j])
    }
    dat_true <- dat_true %>%
        pivot_longer(cols = starts_with("rho"), names_to = "parameter", values_to = "rho") 
    
    dat_fit %>%
        pivot_longer(cols = starts_with("rho"), names_to = "parameter", values_to = "rho") %>%
        group_by(model, parameter) %>%
        summarize(mean_rho = mean(rho), 
                  lower_rho = quantile(rho, prob = 0.025), 
                  upper_rho = quantile(rho, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = parameter, y = mean_rho)) + 
        geom_errorbar(aes(x = parameter, ymin = lower_rho, ymax = upper_rho)) +
        facet_wrap(~ model) +
        geom_point(data = dat_true, aes(x = parameter, y = rho), color = "orange")
}



plot_tau <- function(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
                     fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) {
    dims = length(dat_sim_matern$tau)
    iters <- dim(fit_sim_latent$tau)[1]
    dat_true <- data.frame(model = c("matern", "overdispersed", "latent"))
    dat_fit <- data.frame(iteration = 1:iters, model = rep(c("matern", "overdispersed", "latent"), each = iters))
    
    # process the output into a data.frame
    for (j in 1:dims) {
        dat_true[[paste0("tau[", j, "]")]] = 
            c(dat_sim_matern$tau[j], dat_sim_overdispersed$tau[j], dat_sim_latent$tau[j])
        dat_fit[[paste0("tau[", j, "]")]] = 
            c(fit_sim_matern$tau[, j], fit_sim_overdispersed$tau[, j], fit_sim_latent$tau[, j])
    }
    dat_true <- dat_true %>%
        pivot_longer(cols = starts_with("tau"), names_to = "parameter", values_to = "tau") 
    
    dat_fit %>%
        pivot_longer(cols = starts_with("tau"), names_to = "parameter", values_to = "tau") %>%
        group_by(model, parameter) %>%
        summarize(mean_tau = mean(tau), 
                  lower_tau = quantile(tau, prob = 0.025), 
                  upper_tau = quantile(tau, prob = 0.975)) %>%
        ggplot() +
        geom_point(aes(x = parameter, y = mean_tau)) + 
        geom_errorbar(aes(x = parameter, ymin = lower_tau, ymax = upper_tau)) +
        facet_wrap(~ model) +
        geom_point(data = dat_true, aes(x = parameter, y = tau), color = "orange")
}
