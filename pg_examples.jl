using Random, Distributions, PolyaGammaSamplers;

Random.seed!(123); # Setting the seed

# generate a random polya-gamma sampler
s = PolyaGammaPSWSampler(2, 3.0)

# generate samples from the sampler
x = rand(s, 200)

