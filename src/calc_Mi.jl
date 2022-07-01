export calc_Mi

"""
    calc_mi(Y)

Returns the transformation of an observation `Y` to a constant `Mi` needed for Polya-gamma data augmentation
"""
function calc_Mi(Y)
    J = length(Y)
    Mi = Array{Integer}(undef, J-1)
    if any(ismissing.(Y))
        Mi = zeros(Int64, J-1)
    else
        if J == 2
	    Mi[1] = sum(Y)
	else
	    Mi = sum(Y) .- append!([0], cumsum(Y)[1:(J-2)])
	end
    end
    Mi
end
