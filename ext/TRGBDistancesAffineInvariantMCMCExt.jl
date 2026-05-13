module TRGBDistancesAffineInvariantMCMCExt

# Loaded when AffineInvariantMCMC is available.
# Provides affine-invariant ensemble sampling via
# _sample(::AffineInvariantMCMCJL, ...).

using TRGBDistances: TRGBDistances, AffineInvariantMCMCJL, TRGBChain
import TRGBDistances: _sample
using AffineInvariantMCMC: AffineInvariantMCMC
using Statistics: mean
import Random

function _sample(backend::AffineInvariantMCMCJL, logposterior, x0;
                 rng=Random.default_rng(), kwargs...)
    n = length(x0)
    nwalkers = isnothing(backend.nwalkers) ? max(2 * n, 8) : backend.nwalkers
    x0f = float.(x0)

    # Build initial ensemble: perturb x0 slightly per walker
    x0_matrix = x0f .+ 1e-3 .* Random.randn(rng, n, nwalkers)

    chain_3d, logls_2d = AffineInvariantMCMC.sample(
        logposterior, nwalkers, x0_matrix, backend.nsamples + backend.nburnin, backend.nthin; rng)

    chain_3d = chain_3d[:, :, (backend.nburnin + 1):end]
    logls_2d = logls_2d[:, (backend.nburnin + 1):end]

    # chain_3d : (n_params, n_walkers, n_samples)
    # logls_2d : (n_walkers, n_samples)
    n_params, nw, ns = size(chain_3d)

    # Flatten across walkers: (n_params, n_walkers * n_samples)
    samples  = reshape(chain_3d, n_params, nw * ns)
    logprobs = reshape(logls_2d, nw * ns)

    # Approximate acceptance fraction: fraction of consecutive-walker
    # samples that changed (any parameter differs)
    n_accepted = 0
    n_total = 0
    for w in 1:nw, s in 2:ns
        n_total += 1
        if any(chain_3d[:, w, s] .!= chain_3d[:, w, s - 1])
            n_accepted += 1
        end
    end
    acceptance_fraction = n_total > 0 ? n_accepted / n_total : 0.0

    return TRGBChain(samples, logprobs, acceptance_fraction, backend,
                     (chain = chain_3d, logls = logls_2d))
end

end # module
