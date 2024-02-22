using ADDM

function my_trial_simulator(;model::ADDM.aDDM, fixationData::ADDM.FixationData, 
                        valueLeft::Number, valueRight::Number, 
                        timeStep::Number=10.0, numFixDists::Int64=3, cutOff::Number=100000)
    
    fixUnfixValueDiffs = Dict(1 => valueLeft - valueRight, 2 => valueRight - valueLeft)
    
    fixItem = Number[]
    fixTime = Number[]
    fixRDV = Number[]

    RDV = model.bias
    trialTime = 0
    choice = 0
    tRDV = Number[RDV]
    RT = 0
    uninterruptedLastFixTime = 0
    ndtTimeSteps = Int64(model.nonDecisionTime ÷ timeStep)

    # The values of the barriers can change over time.
    # In this case we include an exponential decay
    # Due to the shape of the exponential decay function the starting point for the decay is exp(0) = 1
    barrierUp = exp.(-model.λ .* (0:cutOff-1))
    barrierDown = -exp.(-model.λ .* (0:cutOff-1))
    
    # Sample and iterate over the latency for this trial.
    latency = rand(fixationData.latencies)
    remainingNDT = model.nonDecisionTime - latency

    # This will not change anything (i.e. move the RDV) if there is no latency data in the fixations
    for t in 1:Int64(latency ÷ timeStep)
        # Sample the change in RDV from the distribution.
        RDV += rand(Normal(0, model.σ))
        push!(tRDV, RDV)

        # If the RDV hit one of the barriers, the trial is over.
        # No barrier decay before decision-related accummulation
        if abs(RDV) >= model.barrier
            choice = RDV >= 0 ? -1 : 1
            push!(fixRDV, RDV)
            push!(fixItem, 0)
            push!(fixTime, t * timeStep)
            trialTime += t * timeStep
            RT = trialTime
            uninterruptedLastFixTime = latency
            trial = ADDM.Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
            trial.fixItem = fixItem 
            trial.fixTime = fixTime 
            trial.fixRDV = fixRDV
            trial.uninterruptedLastFixTime = uninterruptedLastFixTime
            trial.RDV = tRDV
            return trial
        end
    end

    # Add latency to this trial's data
    push!(fixRDV, RDV)
    push!(fixItem, 0)
    push!(fixTime, latency - (latency % timeStep))
    trialTime += latency - (latency % timeStep)

    fixNumber = 1
    prevFixItem = -1
    currFixLocation = 0
    decisionReached = false

    # Begin decision related accummulation
    cumTimeStep = 0
    while true
        if currFixLocation == 0
            # This is an item fixation; sample its location.
            if prevFixItem == -1
                # Sample the first item fixation for this trial.
                currFixLocation = rand(Bernoulli(1 - fixationData.probFixLeftFirst)) + 1
            elseif prevFixItem in [1, 2]
                currFixLocation = abs(3 - prevFixItem)
            end
            prevFixItem = currFixLocation

            # Sample the duration of this item fixation.
            valueDiff = fixUnfixValueDiffs[currFixLocation]
            #[1] is here to make sure it's not sampling from 1-element Vector but from the array inside it
            currFixTime = rand(fixationData.fixations[fixNumber][valueDiff][1]) 
            

            if fixNumber < numFixDists
                fixNumber += 1
            end

        else
            # This is a transition.
             currFixLocation = 0
            # Sample the duration of this transition. The fixation data used below does not have transition information so ignoring this.
            # currFixTime = rand(fixationData.transitions)
            currFixTime = 0
        end

        # Iterate over the remaining non-decision time remaining after the latency
        # This can span more than first fixation depending on the first fixation duration
        # That's why it's not conditioned over the fixation number
        if remainingNDT > 0
            for t in 1:Int64(remainingNDT ÷ timeStep)
                # Sample the change in RDV from the distribution.
                RDV += rand(Normal(0, model.σ))
                push!(tRDV, RDV)

                # If the RDV hit one of the barriers, the trial is over.
                # No barrier decay before decision-related accummulation
                if abs(RDV) >= model.barrier
                    choice = RDV >= 0 ? -1 : 1
                    push!(fixRDV, RDV)
                    push!(fixItem, currFixLocation)
                    push!(fixTime, t * timeStep)
                    trialTime += t * timeStep
                    RT = trialTime
                    uninterruptedLastFixTime = currFixTime
                    decisionReached = true
                    break
                end
            end
        end

        # Break out of the while loop if decision reached during NDT
        # The break above only breaks from the NDT for loop
        if decisionReached
            break
        end

        remainingFixTime = max(0, currFixTime - max(0, remainingNDT))
        remainingNDT -= currFixTime

        # Iterate over the duration of the current fixation.
        # Does not move RDV if there is no fixation time left due to NDT
        for t in 1:Int64(remainingFixTime ÷ timeStep)
            # We use a distribution to model changes in RDV
            # stochastically. The mean of the distribution (the change
            # most likely to occur) is calculated from the model
            # parameters and from the values of the two items.
            if currFixLocation == 0
                μ = 0
            elseif currFixLocation == 1
                μ = model.d * ( (valueLeft + model.η) - (model.θ * valueRight))
            elseif currFixLocation == 2
                μ = model.d * ((model.θ * valueLeft) - (valueRight + model.η))
            end

            # Sample the change in RDV from the distribution.
            RDV += rand(Normal(μ, model.σ))
            push!(tRDV, RDV)

            # Increment cumulative timestep to look up the correct barrier value in case there has been a decay
            # Decay in this case only happens during decision-related accummulation (not before)
            # Don't want to use t here because this is reset for each fixation throughout a trial but the barrier is not
            cumTimeStep += 1

            # If the RDV hit one of the barriers, the trial is over.
            # Decision related accummulation here so barrier might have decayed
            if abs(RDV) >= barrierUp[cumTimeStep]
                choice = RDV >= 0 ? -1 : 1
                push!(fixRDV, RDV)
                push!(fixItem, currFixLocation)
                push!(fixTime, t * timeStep)
                trialTime += t * timeStep
                RT = trialTime
                uninterruptedLastFixTime = currFixTime
                decisionReached = true
                break
            end
        end

        # Break out of the while loop if decision reached during NDT
        # The break above only breaks from the curFixTime for loop
        if decisionReached
            break
        end

        # Add fixation to this trial's data.
        push!(fixRDV, RDV)
        push!(fixItem, currFixLocation)
        push!(fixTime, currFixTime - (currFixTime % timeStep))
        trialTime += currFixTime - (currFixTime % timeStep)

    end

    trial = ADDM.Trial(choice = choice, RT = RT, valueLeft = valueLeft, valueRight = valueRight)
    trial.fixItem = fixItem 
    trial.fixTime = fixTime 
    trial.fixRDV = fixRDV
    trial.uninterruptedLastFixTime = uninterruptedLastFixTime
    trial.RDV = tRDV
    return trial
end