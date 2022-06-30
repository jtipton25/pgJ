# install.packages("BiocManager")
BiocManager::install("rhdf5")
library(rhdf5)

h5ls("./output/overdispersed_sim.jld")

# load the julia simulation parameters

# load the julia MCMC output
tmp_out = h5read("./output/overdispersed_sim.jld", name="_refs")
out = vector(mode='list', length = length(tmp_out[[1]]))
names(out) <- tmp_out[[1]]

for (i in 1:length(tmp_out[[1]]))  {
    out[[i]] <- tmp_out[[i+2]]
}
    