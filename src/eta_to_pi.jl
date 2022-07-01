function expit(x)
    1.0 / (1.0 + exp(-x))
end


export eta_to_pi

"""
    eta_to_pi(eta)

Return the transformation of a `J-1` vector of unconstrained values `eta` to a `J` dimesional simplex (sum-to-one) vector `pi`
"""
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