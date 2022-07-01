export calc_kappa

"""
    calc_kappa(Y, Mi)

Returns the transformation of an observation `Y` and the transformed count `Mi` to a constant `kappa` needed for Polya-gamma data augmentation
"""
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