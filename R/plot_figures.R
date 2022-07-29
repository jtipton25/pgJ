# plot figures
# TODO revise, make a function of data type that accepts command line arguments
library(tidyverse)
library(patchwork)
require(rasterVis)
require(fields)
require(rgdal)
require(raster)
require(enmSdm)
require(rgeos)
library(stringr)


version='5.0'

# load the species names
species_names <- readRDS(here::here('data', paste0('taxa_', version, '.RDS'))) %>%
    # species_names <- readRDS(here::here('data', paste0('taxa_', version, '.RDS'))) %>%
    tolower() %>%
    str_replace("\\.", " ") %>%
    tools::toTitleCase()

# load the preds data
# preds <- readRDS(here::here("output", paste0('polya-gamma-predictions_', version, '.RDS')))
preds <- readRDS(here::here('output', 'pollen', 'pollen_matern_predictions.RDS'))

# load the preds grid
locs_grid <- readRDS(here::here('data', paste0('grid_', version, '.rds')))

pi_mean <- apply(preds$pi, c(2, 3, 4), mean)
pi_sd <- apply(preds$pi, c(2, 3, 4), sd)

# modify this based on Alissa's input
dimnames(pi_mean) <- list(
    location = 1:dim(pi_mean)[1],
    species = species_names,
    time = 1:dim(pi_mean)[3]
)


dat_pi_mean <- as.data.frame.table(pi_mean, responseName = "pi_mean") %>%
    mutate(location = as.numeric(location), time = as.numeric(time)) %>%
    left_join(locs_grid %>%
                  mutate(location = 1:dim(locs_grid)[1]))

dimnames(pi_sd) <- list(
    location = 1:dim(pi_sd)[1],
    species = species_names,
    time = 1:dim(pi_sd)[3]
)

dat_pi_sd <- as.data.frame.table(pi_sd, responseName = "pi_sd") %>%
    mutate(location = as.numeric(location), time = as.numeric(time)) %>%
    left_join(locs_grid %>%
                  mutate(location = 1:dim(locs_grid)[1]))

#### READ MAP DATA ####
# read raster masks (provides masks for spatial domain/resolution of genetic + ENM data)
# stack <- stack(here::here('data', 'map-data', 'study_region_daltonIceMask_lakesMasked_linearIceSheetInterpolation.tif'))
# names(stack) <- 1:701
# sub <- seq(1, 701, by = 33)
# stack_sub <- subset(stack, subset = paste0('X', sub))  # only want mask every 990 years
# stack_sub[stack_sub > 0.6] <- NA
# proj <- proj4string(stack)
proj <- "+proj=aea +lat_0=37.5 +lon_0=-96 +lat_1=29.5 +lat_2=45.5 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs"

na_shp <- readOGR(here::here("data", "map-data", "NA_States_Provinces_Albers.shp"), layer = "NA_States_Provinces_Albers")
na_shp <- sp::spTransform(na_shp, proj)
cont_shp <- subset(na_shp,
                   (NAME_0 %in% c("United States of America", "Mexico", "Canada")))
lake_shp <- readOGR(here::here("data", "map-data", "Great_Lakes.shp"), "Great_Lakes")
lake_shp <- sp::spTransform(lake_shp, proj)

# getting bounding box to specify spatial domain when plotting
bbox_tran <- function(x, coord_formula = '~ x + y', from, to) {
    sp::coordinates(x) <- formula(coord_formula)
    sp::proj4string(x) <- sp::CRS(from)
    bbox <- as.vector(sp::bbox(sp::spTransform(x, CRSobj = sp::CRS(to))))
    return(bbox)
}

grid_box <- bbox_tran(locs_grid, '~ x + y',
                      proj,
                      proj)
xlim = c(grid_box[1], grid_box[3])
ylim = c(grid_box[2], grid_box[4])



dat_cont_shp <- fortify(cont_shp)
dat_lake_shp <- fortify(lake_shp)
# glimpse(dat_ar_shp)


# plot the precincts
map <- ggplot() +
    geom_polygon(data = dat_cont_shp,
                 aes(x = long, y = lat, group = group), fill = NA,
                 color = 'black', size = .2) +
    geom_polygon(data = dat_lake_shp,
                 aes(x = long, y = lat, group = group), fill = "blue",
                 color = 'black', size = .2)
map


# change the time numbering
time_bins <- seq(-285, 21990, by=990)[-1]

# time_vec <- paste(1:21)
# names(time_vec) <- paste(time_bins[1:21], "ybp")
time_vec <- paste(time_bins[1:22], "ybp")
names(time_vec) <- paste(1:22)
time_vec

# generate the plots
base_size <- 14



for (species_to_plot in unique(dat_pi_mean$species)) {
    
    # if (!file.exists(here::here("figures", "matern", paste0("predictions-", species_to_plot, ".png")))) {
    p_mean <- dat_pi_mean %>%
        # filter(species %in% species_to_plot) %>%
        filter(species == species_to_plot) %>%
        ggplot(aes(x = x, y = y, fill = pi_mean)) +
        geom_raster() +
        geom_polygon(data = dat_cont_shp,
                     aes(x = long, y = lat, group = group), fill = NA,
                     color = 'black', size = .2) +
        geom_polygon(data = dat_lake_shp,
                     aes(x = long, y = lat, group = group), fill = "black",
                     color = 'black', size = .2) +
        facet_wrap(~ time, nrow = 3, labeller = as_labeller(time_vec)) +
        scale_fill_viridis_c() +
        ggtitle(substitute(paste("Posterior proportion mean of ", italic(x), " for the ", italic("Matern"), " model"), list(x = as.character(species_to_plot)))) +
        theme_bw(base_size = base_size) +
        theme(axis.text.x = element_blank(),
              axis.ticks.x=element_blank(),
              axis.text.y = element_blank(),
              axis.ticks.y=element_blank()) +
        xlab("") +
        ylab("") +         
        labs(fill = "p") +
        coord_cartesian(xlim = xlim, ylim = ylim)
    
    p_sd <- dat_pi_sd %>%
        # filter(species %in% species_to_plot) %>%
        filter(species == species_to_plot) %>%
        ggplot(aes(x = x, y = y, fill = pi_sd)) +
        geom_raster() +
        geom_polygon(data = dat_cont_shp,
                     aes(x = long, y = lat, group = group), fill = NA,
                     color = 'black', size = .2) +
        geom_polygon(data = dat_lake_shp,
                     aes(x = long, y = lat, group = group), fill = "black",
                     color = 'black', size = .2) +
        facet_wrap(~ time, nrow = 3, labeller = as_labeller(time_vec)) +
        scale_fill_viridis_c() +
        ggtitle(substitute(paste("Posterior proportion sd of ", italic(x), " for the ", italic("Matern"), " model"), list(x = as.character(species_to_plot)))) +
        theme_bw(base_size = base_size) +
        theme(axis.text.x = element_blank(),
              axis.ticks.x=element_blank(),
              axis.text.y = element_blank(),
              axis.ticks.y=element_blank()) +
        xlab("") +
        ylab("") +
        labs(fill = "sd") +
        coord_cartesian(xlim = xlim, ylim = ylim)
    
    # ggsave(p_mean / p_sd, 
    #        file = here::here("figures", "matern", paste0("predictions-", species_to_plot, "-version-", version, ".png")), 
    #        height = 12,
    #        width = 16)
    # }
}
