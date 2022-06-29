using LinearAlgebra, Random, Distributions

Random.seed!(1)

A = rand(Normal(0, 1), 10, 10)
A = A * A'
D = Diagonal(10 .* ones(10))
inv(D)
inv(D + A)

cholesky(D)
cholesky(D + A)

inv(cholesky(D).U)
inv(cholesky(D+A).U)

inv(cholesky(D))
inv(cholesky(D+A))