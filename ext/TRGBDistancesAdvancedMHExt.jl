module TRGBDistancesAdvancedMHExt

# Loaded when AdvancedMH is available.
# Provides random-walk Metropolis-Hastings posterior sampling via
# _sample(::AdvancedMHJL, ...).

using TRGBDistances: TRGBDistances, AdvancedMHJL, TRGBChain
import TRGBDistances: _sample
using AdvancedMH: AdvancedMH
using Distributions: MvNormal
using LinearAlgebra: I
using Statistics: mean
import Random

function _sample(backend::AdvancedMHJL, logposterior, x0;
                 rng=Random.default_rng(), kwargs...)
    n = length(x0)
    x0f = float.(x0)
    parallel = isnothing(backend.parallel) ? (backend.nwalkers == 1 ? AdvancedMH.MCMCSerial() : AdvancedMH.MCMCThreads()) : backend.parallel

    model = AdvancedMH.DensityModel(logposterior)
    proposal = AdvancedMH.RandomWalkProposal(
                    MvNormal(zeros(n), backend.proposal_scale^2 * I(n)))
    sampler  = AdvancedMH.MetropolisHastings(proposal)

    total = backend.nburnin + backend.nsamples
    full_chain = AdvancedMH.sample(rng, model, sampler, parallel, total, backend.nwalkers;
                                   initial_params = [x0f for i in 1:backend.nwalkers],
                                   progress = backend.progress,
                                   thinning = backend.nthin)

    # Flatten across walkers and discard burn-in
    chain = mapreduce(chain -> chain[(backend.nburnin + 1):end], vcat, full_chain)
    samples  = reduce(hcat, [t.params for t in chain])  # n_params × n_samples
    logprobs = [t.lp for t in chain]
    acceptance_fraction = mean(t.accepted for t in chain)

    return TRGBChain(samples, logprobs, acceptance_fraction, backend, chain)
end

end # module
