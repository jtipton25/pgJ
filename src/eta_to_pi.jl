function expit(x)
    1.0 / (1.0 + exp(-x))
end

function eta_to_pi(eta)
    J = length(eta) + 1
    pi = Array{Float64}(undef, J)
    stick = 1
    for j in 1:(J-1)
        pi[j] = expit(eta[j]) * stick
	stick = stick - pi[j]
    end
    pi[J] = stick
    pi
end