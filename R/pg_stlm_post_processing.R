# install.packages("BiocManager")
# BiocManager::install("rhdf5")
library(rhdf5)
library(tidyverse)
library(patchwork)

source(here::here("R", "pg_stlm_post_processing_plots.R"))
# h5ls("./output/overdispersed_sim.jld")

# load the julia simulation parameters


# matern simulation ----


dat_sim_matern <- readRDS("./output/matern_sim_data.RDS")
fit_sim_matern <- readRDS("./output/matern_sim_fit.RDS")

plot_eta_sim(dat_sim_matern)
plot_psi_sim(dat_sim_matern)
plot_Xbeta_sim(dat_sim_matern)
plot_pi_sim(dat_sim_matern)

# thin the matern samples by hand
# thin_idx <- seq(5, 5000, 5)
# fit_sim_matern$theta <- fit_sim_matern$theta[thin_idx, ]
# fit_sim_matern$eta <- fit_sim_matern$eta[thin_idx, , , ]
# fit_sim_matern$tau <- fit_sim_matern$tau[thin_idx, ]
# fit_sim_matern$pi <- fit_sim_matern$pi[thin_idx, , , ]
# fit_sim_matern$rho <- fit_sim_matern$rho[thin_idx, ]
# fit_sim_matern$beta <- fit_sim_matern$beta[thin_idx, , ]

# overdispersed simulation ----


dat_sim_overdispersed <- readRDS("./output/overdispersed_sim_data.RDS")
fit_sim_overdispersed <- readRDS("./output/overdispersed_sim_fit.RDS")

plot_eta_sim(dat_sim_overdispersed)
plot_psi_sim(dat_sim_overdispersed)
plot_Xbeta_sim(dat_sim_overdispersed)
plot_pi_sim(dat_sim_overdispersed)


# latent simulation ----

dat_sim_latent <- readRDS("./output/latent_sim_data.RDS")
fit_sim_latent <- readRDS("./output/latent_sim_fit.RDS")

plot_eta_sim(dat_sim_latent)
plot_psi_sim(dat_sim_latent)
plot_Xbeta_sim(dat_sim_latent)
plot_pi_sim(dat_sim_latent)


str(fit_sim_matern)
str(fit_sim_overdispersed)
str(fit_sim_latent)


library(tidyverse)


plot_beta(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
          fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) 

plot_theta(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
           fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) 

plot_rho(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
         fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) 

plot_tau(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
          fit_sim_matern, fit_sim_overdispersed, fit_sim_latent) 

plot_eta(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
         fit_sim_matern, fit_sim_overdispersed, fit_sim_latent)

plot_pi(dat_sim_matern, dat_sim_overdispersed, dat_sim_latent, 
         fit_sim_matern, fit_sim_overdispersed, fit_sim_latent)


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
