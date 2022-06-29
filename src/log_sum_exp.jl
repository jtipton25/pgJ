function log_sum_exp(X)
    a = maximum(X)  # Find maximum value in X
    log(sum(exp.(X .- a))) + a
end