function calc_kappa(Y, Mi)
    J = length(Y)
    kappa = Array{Integer}(undef, J-1)
    if any(ismissing.(Y))
        kappa = zeros(J-1)
    else
        kappa = Y[1:(J-1)] .- (Mi / 2.0)
    end
    kappa
end