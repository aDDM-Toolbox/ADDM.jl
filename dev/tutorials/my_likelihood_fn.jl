using Distributions
using LinearAlgebra

function my_likelihood_fn(;model::ADDM.aDDM, trial::ADDM.Trial, timeStep::Number = 10.0, 
                                   stateStep::Number = 0.1)
    
    # Iterate over the fixations and discount the non-decision time.
    if model.nonDecisionTime > 0
        correctedFixItem = Number[]
        correctedFixTime = Number[]
        remainingNDT = model.nonDecisionTime
        for (fItem, fTime) in zip(trial.fixItem, trial.fixTime) # iterate through each fixation in the trial
            if remainingNDT > 0
                push!(correctedFixItem, 0)
                push!(correctedFixTime, min(remainingNDT, fTime)) # if the fTime is smaller push that otherwise push ndt
                push!(correctedFixItem, fItem)
                push!(correctedFixTime, max(fTime - remainingNDT, 0))
                remainingNDT = remainingNDT - fTime
            else
                push!(correctedFixItem, fItem)
                push!(correctedFixTime, fTime)
            end
        end
    else
        correctedFixItem = trial.fixItem
        correctedFixTime = trial.fixTime
    end
    
    # Iterate over the fixations and get the number of time steps for this trial.
    numTimeSteps = 0
    
    for fTime in correctedFixTime
        numTimeSteps += Int64(fTime ÷ timeStep)
    end
    
    if numTimeSteps < 1
        throw(RuntimeError("Trial response time is smaller than time step."))
    end
    numTimeSteps += 1
    
    # The values of the barriers can change over time.
    barrierUp = exp.(-model.λ .* (0:numTimeSteps-1))
    barrierDown = -exp.(-model.λ .* (0:numTimeSteps-1))
    
    # Obtain correct state step.
    halfNumStateBins = ceil(model.barrier / stateStep)
    stateStep = model.barrier / (halfNumStateBins + 0.5)
    
    # The vertical axis is divided into states.
    states = range(-1*(model.barrier) + stateStep / 2, 1*(model.barrier) - stateStep/2, step=stateStep)
    
    # Find the state corresponding to the bias parameter.
    biasState = argmin(abs.(states .- model.bias))
    
    # Initial probability for all states is zero, except the bias state,
    # for which the initial probability is one.
    prStates = zeros(length(states), numTimeSteps)
    prStates[biasState,1] = 1
    
    # The probability of crossing each barrier over the time of the trial.
    probUpCrossing = zeros(numTimeSteps)
    probDownCrossing = zeros(numTimeSteps)
    
    time = 1
    
    # Dictionary of μ values from fItem.
    μDict = Dict{Number, Number}()
    for fItem in 0:2
        if fItem == 1
            μ = model.d * ((trial.valueLeft + model.η) - (model.θ * trial.valueRight))
        elseif fItem == 2
            μ = model.d * ((model.θ * trial.valueLeft) - (trial.valueRight + model.η))
        else
            μ = 0
        end
        
        μDict[fItem] = μ
    end 
    
    changeMatrix = states .- reshape(states, 1, :)
    changeUp = (barrierUp .- reshape(states, 1, :))'
    changeDown = (barrierDown .- reshape(states, 1, :) )'
    
    pdfDict = Dict{Number, Any}()
    cdfUpDict = Dict{Number, Any}()
    cdfDownDict = Dict{Number, Any}() 
    
    for fItem in 0:2
        normpdf = similar(changeMatrix)
        cdfUp = similar(changeUp[:, time])
        cdfDown = similar(changeDown[:, time])
        
        @. normpdf = pdf(Normal(μDict[fItem], model.σ), changeMatrix)
        @. cdfUp = cdf(Normal(μDict[fItem], model.σ), changeUp[:, time])
        @. cdfDown = cdf(Normal(μDict[fItem], model.σ), changeDown[:, time])
        pdfDict[fItem] = normpdf
        cdfUpDict[fItem] = cdfUp
        cdfDownDict[fItem] = cdfDown
    end
    
    # Iterate over all fixations in this trial.
    for (fItem, fTime) in zip(correctedFixItem, correctedFixTime)
        # We use a normal distribution to model changes in RDV
        # stochastically. The mean of the distribution (the change most
        # likely to occur) is calculated from the model parameters and from
        # the item values.
        μ = μDict[fItem]
        normpdf = pdfDict[fItem]
        cdfUp = cdfUpDict[fItem]
        cdfDown = cdfDownDict[fItem]
        
        # Iterate over the time interval of this fixation.
        for t in 1:Int64(fTime ÷ timeStep)
            # Update the probability of the states that remain inside the 
            # barriers. The probability of being in state B is the sum, 
            # over all states A, of the probability of being in A at the 
            # previous timestep times the probability of changing from A to
            # B. We multiply the probability by the stateStep to ensure
            # that the area under the curves for the probability 
            # distributions probUpCrossing and probDownCrossing add up to 1.
            prStatesNew = stateStep * (normpdf * prStates[:,time])
            prStatesNew[(states .>= barrierUp[time]) .| (states .<= barrierDown[time])] .= 0
            
            # Calculate the probabilities of crossing the up barrier and
            # the down barrier. This is given by the sum, over all states
            # A, of the proability of being in A at the previous timestep
            # times the probability of crossing the barrier if A is the
            # previous state.
            tempUpCross = dot(prStates[:,time], 1 .- cdfUp)
            tempDownCross = dot(prStates[:,time], cdfDown)
            
            # Renormalize to cope with numerical approximations.
            sumIn = sum(prStates[:,time])
            sumCurrent = sum(prStatesNew) + tempUpCross + tempDownCross
            prStatesNew = prStatesNew * sumIn / sumCurrent
            tempUpCross = tempUpCross * sumIn / sumCurrent
            tempDownCross = tempDownCross * sumIn / sumCurrent

            # Update the probabilities of each state and the probabilities of
            # crossing each barrier at this timestep
            prStates[:,time+1] = prStatesNew
            probUpCrossing[time+1] = tempUpCross
            probDownCrossing[time+1] = tempDownCross
            
            time += 1
        end
    end
    
    # Compute the likelihood contribution of this trial based on the final
    # choice.
    likelihood = 0
    if trial.choice == -1 # Choice was left.
        if probUpCrossing[end] > 0
            likelihood = probUpCrossing[end]
        end
    elseif trial.choice == 1 # Choice was right.
        if probDownCrossing[end] > 0 
            likelihood = probDownCrossing[end]
        end
    end
    
    return likelihood
end