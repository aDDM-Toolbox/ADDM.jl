"""
    grid_search(data, likelihood_fn, param_grid, return_grid_likelihoods = false; 
                likelihood_args =  (timeStep = 10.0, approxStateStep = 0.1), return_trial_likelihoods = false)

Compute the likelihood of either observed or simulated data for all parameter combinations in paramGrid.

# Arguments

## Required 

- `data`: Data for which the sum of negative log likelihoods will be computed for each trial.
  Should be a vector of `ADDM.Trial` objects.
- `likelihood_fn`: Name of likelihood function to be used to compute likelihoods. 
  The toolbox has `ADDM.aDDM_get_trial_likelihood` and `ADDM.aDDM_get_trial_likelihood` defined.
- `param_grid`: Grid of parameter combinations for which the sum of nll's for the `data` is 
  computed.
- `fixed_params`: Default `Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0)`.
  Parameters required by the `likelihood_fn` that are not specified to vary across likelihood 
  computations.

## Optional 

- `return_grid_likelihoods`: Default `true`. If true, will return the sum of nll's for 
  each parameter combination
  in the grid search.
- `likelihood_args`: Default `(timeStep = 10.0, approxStateStep = 0.1)`. Additional 
  arguments to be passed onto `likelihood_fn`. 
- `return_trial_likelihoods`: Default `false`. If true, will return the likelihood of each 
  trial in `data` for each parameter combination in `param_grid`.

# Returns
- `best_part`: `Dict` containing the parameter combination with the lowest nll.
- `all_nll_df`: `DataFrame` containing sum of nll's for each parameter combination.
- `trial_likelihoods`: Likelihood for each trial for each parameter combination.

"""
function grid_search(data, likelihood_fn, param_grid, 
                    fixed_params = Dict(:θ=>1.0, :η=>0.0, :barrier=>1, :decay=>0, :nonDecisionTime=>0, :bias=>0.0); 
                    return_grid_likelihoods = true,
                    likelihood_args = (timeStep = 10.0, approxStateStep = 0.1), 
                    return_trial_likelihoods = false)

  n = length(param_grid) # number of parameter combinations specified in param_grid
  all_nll = Vector{}(undef, n)
  trial_likelihoods = Vector{}(undef, n)

  # Pass fixed parameters to the model
  # These don't need to be updated for each combination of the parameter grid
  model = ADDM.aDDM()
  # model = aDDM()
  for (k,v) in fixed_params setproperty!(model, k, v) end

  # What should the structure of param_grid be?
  # Currently it is a dictionary of NamedTuples 
  for (k, cur_grid_params) in param_grid
    
    # Update the model with the current parameter combination
    if !(cur_grid_params isa Dict)
      for (k,v) in pairs(cur_grid_params) setproperty!(model, k, v) end
    else
      for (k,v) in cur_grid_params setproperty!(model, k, v) end
    end
    
    # Make sure param names are converted to Greek symbols
    convert_param_symbols(model)

    if return_trial_likelihoods
      all_nll[k], trial_likelihoods[k] = compute_trials_nll(model, data, likelihood_fn, likelihood_args; 
                              return_trial_likelihoods = return_trial_likelihoods)
    else
      all_nll[k] = compute_trials_nll(model, data, likelihood_fn, likelihood_args)
    end
  
  end

  # Wrangle likelihood data and extract best pars robustly before returning

  # Extract best pars
  # TODo: Convert param names to greek letters?
  minIdx = argmin(all_nll)
  best_fit_pars = Dict(pairs(param_grid[minIdx]))
  best_pars = merge(best_fit_pars, fixed_params)
  best_pars[:nll] = all_nll[minIdx]

  if return_grid_likelihoods
      # Add param info to all_nll
      all_nll_df = DataFrame()
      for (k, v) in param_grid
        row = DataFrame(Dict(pairs(v)))
        row.nll .= all_nll[k]
        append!(all_nll_df, row)
      end

    if return_trial_likelihoods 
        # TODo: Add param info to trial_likelihoods 
      return best_pars, all_nll_df, trial_likelihoods
    else
      return best_pars, all_nll_df
    end
  else
    return best_pars
  end

end
