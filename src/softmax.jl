function softmax(X)
    exp.(X .- log_sum_exp(X))
end