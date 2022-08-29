# NOT TESTED
function update_tuning(k, accept, tune)
    delta = 1.0 / sqrt(k)
    tune_out = 0.0
    if accept > 0.44
        tune_out = exp(log(tune) + delta)
    else
        tune_out = exp(log(tune) - delta)
    end
    accept_out = 0.0
    Dict{Sring, Float64}("accept" => accept_out, "tune" => tune_out)
end

# TESTED
function update_tuning_vec(k, accept, tune)
    delta = 1.0 / sqrt(k)
    n = length(tune)
    tune_out = zeros(n)
    for i in 1:n
        if accept[i] > 0.44
            tune_out[i] = exp(log(tune[i]) + delta)
        else
            tune_out[i] = exp(log(tune[i]) - delta)
	end
    end
    accept_out = zeros(n)
    Dict{String, Any}("accept" => accept_out, "tune" => tune_out)
end

# NOT TESTED
function update_tuning_mat(k, accept, tune)
    delta = 1.0 / sqrt(k)
    n = size(tune, 1)
    p = size(tune, 2)
    tune_out = zeros(n, p)
    for i in 1:n
    	for j in 1:p
            if accept[i, j] > 0.44
                 tune_out[i, j] = exp(log(tune[i, j]) + delta)
             else
                 tune_out[i, j] = exp(log(tune[i, j]) - delta)
	     end
	 end
    end
    accept_out = zeros(n, p)
    Dict{String, Any}("accept" => accept_out, "tune" => tune_out)
end

# NOT TESTED
function update_tuning_mv(k, accept, lambda, batch_samples, Sigma_tune, Sigma_tune_chol)
    # determine optimal acceptance rates based on dimension of parameter
    arr = [0.44, 0.35, 0.32, 0.25, 0.234] # optimal tuning rates
    dimension = size(batch_samples, 1)
    if dimension >= 5
        dimension = 5
    end
    optimal_accept = arr[dimension]

    # setup tuning adjustment
    batch_size = size(batch_samples, 1)
    d = size(batch_samples, 2)
    times_adapted = k / 50
    gamma1 = 1.0 / ((times_adapted + 3.0)^(0.8))
    gamma2 = 10.0 * gamma1
    adapt_factor = exp(gamma2 * (accept - optimal_accept))
    lambda_out = lambda * adapt_factor
    batch_samples_out = copy(batch_samples)
    for j in 1:d
    	mean_batch = mean(batch_samples[:, j]) # check the dims
	for i in 1:batch_size
	    batch_samples_out[i, j] = batch_samples[i, j] - mean_batch
	end
    end
    Sigma_tune_out = Sigma_tune + gamma1 * (batch_samples_out' * batch_samples_out / (50.0 - 1.0) - Sigma_tune)
    Sigma_tune_chol_out = cholesky(Matrix(Hermitian(Sigma_tune_out)))
    accept_out = 0.0
    batch_samples_out = zeros(batch_size, d)
    Dict{String, Any}("accept" => accept_out, "lambda" => lambda_out,
                      "batch_samples" => batch_samples_out,
		      "Sigma_tune" => Sigma_tune_out,
		      "Sigma_tune_chol" => Sigma_tune_chol_out)

end


# NOT TESTED OR WRITTEN -- WILL BE DEPRECATED WITHOUT CHOLESKY INSTEAD WILL USE PDMat
# function updated_tuning_mv_mat
# NOT TESTED
function update_tuning_mv_mat(k, accept, lambda, batch_samples, Sigma_tune, Sigma_tune_chol)
    # determine optimal acceptance rates based on dimension of parameter
    arr = [0.44, 0.35, 0.32, 0.25, 0.234] # optimal tuning rates
    dimension = size(batch_samples, 2)
    if dimension >= 5
        dimension = 5
    end
    optimal_accept = arr[dimension]

    # setup tuning adjustment
    batch_size = size(batch_samples, 1)
    d = size(batch_samples, 2)
    p = size(batch_samples, 3)
    times_adapted = k / 50
    gamma1 = 1.0 / ((times_adapted + 3.0)^(0.8))
    gamma2 = 10.0 * gamma1
    adapt_factor = exp.(gamma2 * (accept .- optimal_accept))
    lambda_out = lambda .* adapt_factor
    batch_samples_out = copy(batch_samples)
    Sigma_tune_out = copy(Sigma_tune)
    Sigma_tune_chol_out = copy(Sigma_tune_chol)
    for j in 1:d
        for k in 1:p
    	    mean_batch = mean(batch_samples[:, j, k]) # check the dims
	        batch_samples_out[:, j, k] = batch_samples[:, j, k] .- mean_batch
	    end
    end
    for j in 1:p
        Sigma_tune_out[j] = Sigma_tune[j] .+ gamma1 .* (batch_samples_out[:, j, :]' * batch_samples_out[:, j, :] ./ (50.0-1.0) .- Sigma_tune[j])
        Sigma_tune_chol_out[j] = cholesky(Matrix(Hermitian(Sigma_tune_out[j])))        
    end
    accept_out = zeros(size(accept))
    batch_samples_out = zeros(batch_size, d, p)
    Dict{String, Any}("accept" => accept_out, 
                    "lambda" => lambda_out,
                    "batch_samples" => batch_samples_out,
                    "Sigma_tune" => Sigma_tune_out,
		            "Sigma_tune_chol" => Sigma_tune_chol_out)

end


# function updated_tuning_mv_mat
# NOT TESTED
function update_tuning_mv_mat(k, accept, lambda, batch_samples, Sigma_tune)
    # determine optimal acceptance rates based on dimension of parameter
    arr = [0.44, 0.35, 0.32, 0.25, 0.234] # optimal tuning rates
    dimension = size(batch_samples, 2)
    if dimension >= 5
        dimension = 5
    end
    optimal_accept = arr[dimension]

    # setup tuning adjustment
    batch_size = size(batch_samples, 1)
    d = size(batch_samples, 2)
    p = size(batch_samples, 3)
    times_adapted = k / 50
    gamma1 = 1.0 / ((times_adapted + 3.0)^(0.8))
    gamma2 = 10.0 * gamma1
    adapt_factor = exp.(gamma2 * (accept .- optimal_accept))
    lambda_out = lambda .* adapt_factor
    batch_samples_out = copy(batch_samples)
    Sigma_tune_out = copy(Sigma_tune)
    Sigma_tune_chol_out = copy(Sigma_tune_chol)
    for j in 1:d
        for k in 1:p
    	    mean_batch = mean(batch_samples[:, j, k]) # check the dims
	        batch_samples_out[:, j, k] = batch_samples[:, j, k] .- mean_batch
	    end
    end
    for j in 1:p
        Sigma_tune_out[j] = PDMat(Sigma_tune[j].mat .+ gamma1 .* (batch_samples_out[:, j, :]' * batch_samples_out[:, j, :] ./ (50.0-1.0) .- Sigma_tune[j].mat))
    end
    accept_out = zeros(size(accept))
    batch_samples_out = zeros(batch_size, d, p)
    Dict{String, Any}("accept" => accept_out, 
                    "lambda" => lambda_out,
                    "batch_samples" => batch_samples_out,
                    "Sigma_tune" => Sigma_tune_out
		            )

end