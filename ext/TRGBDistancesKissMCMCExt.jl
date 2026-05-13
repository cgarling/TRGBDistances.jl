module TRGBDistancesKissMCMCExt

using TRGBDistances: TRGBDistances, KissMCMCJL, TRGBChain
import TRGBDistances: _sample
using KissMCMC: KissMCMC
using Statistics: mean

function _sample(backend::KissMCMCJL, logposterior, x0; kwargs...)
    x0f = float.(x0)
    n = length(x0f)
    # Initialize ensemble of walkers near x0
    nwalkers = isnothing(backend.nwalkers) ? max(2 * n, 8) : backend.nwalkers
    # theta0s = [x0f .+ 0.01 .* randn(n) for _ in 1:nwalkers]
    theta0s = KissMCMC.make_theta0s(x0f, 1e-2, logposterior, nwalkers)

    # emcee returns (thetas, accept_ratios, logps, blobs)
    # thetas is Vector{Vector{Vector{Float64}}}: thetas[walker][sample] = param vector
    thetas, accept_ratios, logps, _ = KissMCMC.emcee(
        logposterior,
        theta0s;
        niter=backend.nsamples + backend.nburnin,
        nburnin=backend.nburnin,
        nthin=backend.nthin,
        kwargs...
    )

    # # Squash walkers into a flat sequence of samples
    # # thetas[w][s] is a parameter vector; collect all samples across walkers
    all_samples = [thetas[w][s] for w in eachindex(thetas) for s in eachindex(thetas[w])]
    sample_matrix = reduce(hcat, all_samples)  # n_params × n_total_samples

    # logps has one entry per walker (mean or last logP — use what's available)
    acceptance_fraction = mean(mean(ar) for ar in accept_ratios)

    # all_samples, acceptance_fraction, _b, logps = KissMCMC.squash_walkers(thetas, accept_ratios, _b, logps) # puts all walkers into one

    return TRGBChain(
        sample_matrix,
        reduce(vcat, logps),
        acceptance_fraction,
        backend,
        (thetas=thetas, logps=logps, accept_ratios=accept_ratios),
    )
end

end # module
