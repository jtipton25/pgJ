import SpecialFunctions.besselk
import SpecialFunctions.gamma

export matern

"""
    matern()

Return an isotropic and stationary Matern covariance matrix with variance `tau2`, smoothness parameter `nu`, and range `phi` for a distance `d`
"""
function matern(d, nu, phi)
    dd = d / phi
    if isapprox(dd, 0.0, atol=1e-8)
        # 1.0
        dd = 1e-8
    end
    return 1.0 / (2.0^(nu - 1.0) * gamma(nu)) * (dd^nu) * besselk(nu, dd)
end
# function matern(d, nu, phi)
#     if isapprox(d, 0.0, atol=1e-10)
#         1.0
#     else
#         inner = sqrt(2.0 * nu) * d / phi
#         return 2.0^(1.0 - nu) / gamma(nu) * (inner)^nu * besselk(nu, inner)
#     end
# end

# using Plots
# d = LinRange(0.01, 3, 100)

# # changing nu
# cov1 = [matern(v, 1.0, 0.5, 0.5) for v in d]
# cov2 = [matern(v, 1.0, 1.5, 0.5) for v in d]
# cov3 = [matern(v, 1.0, 2.5, 0.5) for v in d]
# plot(d, cov1)
# plot!(d, cov2)
# plot!(d, cov3)

# # changing phi
# cov1 = [matern(v, 1.0, 0.5, 0.5) for v in d]
# cov2 = [matern(v, 1.0, 0.5, 1.5) for v in d]
# cov3 = [matern(v, 1.0, 0.5, 2.5) for v in d]
# plot(d, cov1)
# plot!(d, cov2)
# plot!(d, cov3)


export correlation_function

"""
correlation_function()

Return an isotropic and stationary correlation matrix with range or smoothness and range parameter `theta` range (`corr_fun = exponential`) or smoothness and range parameter `theta` (`corr_fun = "matern"`, the Matern smoothness parameter `nu` is `theta[1]` and range parameter `phi` is `theta[2]`) for a distance `d`
"""
function correlation_function(d, theta; corr_fun="exponential")
    @assert d >= 0 "d must be a nonnegative scalar"    
    @assert (corr_fun == "exponential") | (corr_fun == "matern")
    # "corr_fun must be either \"exponential\" or \"matern\""
        if (corr_fun == "exponential")
        # exponential covariance function
        @assert length(theta) == 1 "for exponential covariance function, theta must be a positive scalar"
        @assert all(theta .> 0) "for exponential covariance function, theta must be a positive scalar"
        return exp(-d / theta[1])
    else (corr_fun == "matern")
        # matern covariance function
        @assert length(theta) == 2 "for matern covariance function, theta must be a vector of length 2 of positive numbers"
        @assert all(theta .> 0) "for matern covariance function, theta must be a vector of length 2 of positive numbers"
        return matern(d, theta[1], theta[2])
    end

end

export covariance_function

"""
covariance_function()

Return an isotropic and stationary covariance matrix with variance `tau2`, range (`corr_fun = exponential`) or smoothness and range parameter `theta` (`corr_fun = "matern"`, the Matern smoothness parameter `nu` is `theta[1]` and range parameter `phi` is `theta[2]`), for a distance `d`
"""
function covariance_function(d, tau2, theta; corr_fun="exponential")
    # add in checks
    @assert tau2 > 0 "tau2 must be a positive scalar"
    @assert length(tau2) ==1 "tau2 must be a positive scalar"
    @assert (corr_fun == "exponential") | (corr_fun == "matern")
    # "corr_fun must be either \"exponential\" or \"matern\""
        if (corr_fun == "exponential")
        # exponential covariance function
        return tau2 * exp(-d / theta[1])
    else (corr_fun == "matern")
        # matern covariance function
        return tau2 * matern(d, theta[1], theta[2])
    end

end


# covariance_function(-0.1, 0.1, 0.1) # error    
# covariance_function(0, 0.1, 0.1) # 0.1
# covariance_function(0, 0.1, 0.1, corr_fun = "a") # error
# covariance_function(0.1, 0.1, 0.1) # 0.036787944117144235
# covariance_function(0.1, 0.1, 0.1, corr_fun="exponential") # 0.036787944117144235
# covariance_function(0.1, 0.1, [0.1, 0.2], corr_fun="exponential") #error
# covariance_function(0.1, 0.1, [0.1, 0.2], corr_fun="matern") # 0.028094297248043486
# covariance_function(0.1, 0.1, 0.1, corr_fun="matern") #error


# l = @layout [grid(4, 3)]
# d = LinRange(0.01, 2, 100)

# #
# cov1 = [covariance_function(v, 1.0, [0.5, 0.25], corr_fun = "matern") for v in d]
# cov2 = [covariance_function(v, 1.0, [0.5, 0.5], corr_fun = "matern") for v in d]
# cov3 = [covariance_function(v, 1.0, [0.5, 0.75], corr_fun = "matern") for v in d]
# p1 = plot(d, cov1)
# p2 = plot(d, cov2)
# p3 = plot(d, cov3)
# #
# cov4 = [covariance_function(v, 1.0, [1.5, 0.25], corr_fun = "matern") for v in d]
# cov5 = [covariance_function(v, 1.0, [1.5, 0.5], corr_fun = "matern") for v in d]
# cov6 = [covariance_function(v, 1.0, [1.5, 0.75], corr_fun = "matern") for v in d]
# p4 = plot(d, cov4)
# p5 = plot(d, cov5)
# p6 = plot(d, cov6)
# #
# cov7 = [covariance_function(v, 1.0, [2.5, 0.25], corr_fun = "matern") for v in d]
# cov8 = [covariance_function(v, 1.0, [2.5, 0.5], corr_fun = "matern") for v in d]
# cov9 = [covariance_function(v, 1.0, [2.5, 0.75], corr_fun = "matern") for v in d]
# p7 = plot(d, cov7)
# p8 = plot(d, cov8)
# p9 = plot(d, cov9)

# # changing nu
# cov10 = [covariance_function(v, 1.0, 0.25, corr_fun = "exponential") for v in d]
# cov11 = [covariance_function(v, 1.0, 0.5, corr_fun = "exponential") for v in d]
# cov12 = [covariance_function(v, 1.0, 0.75, corr_fun = "exponential") for v in d]
# p10 = plot(d, cov10)
# p11 = plot(d, cov11)
# p12 = plot(d, cov12)

# plot(p10, p11, p12, p1, p2, p3, p4, p5, p6, p7, p8, p9, layout = l,
#     title=reshape(["exponential, phi = 0.25",
#     "exponential, phi = 0.5",
#     "exponential, phi = 0.75",
#     "matern, nu = 0.5, phi = 0.25",
#     "matern, nu = 0.5, phi = 0.5",
#     "matern, nu = 0.5, phi = 0.75",
#     "matern, nu = 1.5, phi = 0.25",
#     "matern, nu = 1.5, phi = 0.5",
#     "matern, nu = 1.5, phi = 0.75",
#     "matern, nu = 2.5, phi = 0.25",
#     "matern, nu = 2.5, phi = 0.5",
#     "matern, nu = 2.5, phi = 0.75",
    
#     ], 1, 12), 
#     titlefont = font(8),
#     legend = false)


