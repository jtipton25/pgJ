# Setup cross-validation data

# generate the cross-validation datasets
if(!dir.exists(here::here("output"))) {
    dir.create(here::here("output"))
}
if(!dir.exists(here::here("output", "cross-validate"))) {
    dir.create(here::here("output", "cross-validate"))
}



# library(raster)
# library(geosphere)
library(tidyverse)
# library(sp)
# library(rgdal)
# library(rdist)
# library(rgeos)
# library(ggplot2)
# library(invgamma)
# library(mvnfast)
# library(splines)
# library(pgdraw)
# library(fields)
# library(geoR)
# library(dplyr)
# library(data.table)
# library(xtable)
# library(pgR)



version='5.0'

#### DATA PREP #### -- renamed version 5.0
# y <- readRDS(here::here('data', paste0('paleo_pollen_dat_', version, '.RDS')))
# taxa.keep <- readRDS(here::here('data', paste0('pollen_taxa_', version, '.RDS')))
# locs <- readRDS(here::here('data', paste0('paleo_pollen_locs_', version, '.RDS')))
y <- readRDS(here::here('data', paste0('pollen_dat_', version, '.RDS')))
taxa.keep <- readRDS(here::here('data', paste0('taxa_', version, '.RDS')))
locs <- readRDS(here::here('data', paste0('pollen_locs_', version, '.RDS')))
rescale <- 1e4

locs_scaled <- locs/rescale
N_locs_dat = nrow(locs_scaled)
X <- matrix(rep(1, N_locs_dat),N_locs_dat, 1)

# setup cross-validation ------------------------------------------------------

set.seed(2022)
K <- 8
total_obs <- dim(y)[1] * dim(y)[3]
# make sure this is an exact integer (which it is in our case -- could be more general but is not needed)
n_cv <- total_obs / K

if (file.exists(here::here("output", "cross-validate", paste0("cv_idx_", version, ".RData")))) {
    load(here::here("output", "cross-validate", paste0("cv_idx_", version, ".RData")))
} else {
    cv_idx <- matrix(sample(1:total_obs), n_cv, K)
    obs_idx <- expand_grid(row_id = 1:dim(y)[1], col_id = 1:dim(y)[3])
    save(cv_idx, obs_idx, file = here::here("output", "cross-validate", paste0("cv_idx_", version, ".RData")))
}

# generate the data and save to file for Julia
for (k in 1:K) {
    # set up the cross-validation
    y_train <- y
    y_test <- array(NA, dim = dim(y))
    obs_fold <- obs_idx[cv_idx[, k], ]
    for (i in 1:n_cv) {
        y_train[obs_fold$row_id[i], , obs_fold$col_id[i]] <- NA
        y_test[obs_fold$row_id[i], , obs_fold$col_id[i]] <- y[obs_fold$row_id[i], , obs_fold$col_id[i]] 
    }
    saveRDS(y_train, file = here::here("output", "cross-validate", paste0("pollen_", version, "_train_fold_", k, ".rds")))
    saveRDS(y_test, file = here::here("output", "cross-validate", paste0("pollen_", version, "_test_fold_", k, ".rds")))
}

