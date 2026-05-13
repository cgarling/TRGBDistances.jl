module TRGBDistancesDynamicHMCExt

# Loaded when DynamicHMC and LogDensityProblems are both available.
# Provides HMC-based posterior sampling via _sample(::DynamicHMCJL, ...).
# Gradient computation delegates to build_gradient(backend.ad, ...), which is
# provided by TRGBDistancesForwardDiffExt depending
# on which AD package is loaded alongside DynamicHMC.

using TRGBDistances: TRGBDistances, DynamicHMCJL, TRGBChain, build_gradient
import TRGBDistances: _sample
using DynamicHMC: DynamicHMC
using LogDensityProblems: LogDensityProblems
import Random
using Statistics: mean

# ---------------------------------------------------------------------------
# LogDensityProblems wrapper
# ---------------------------------------------------------------------------

struct TRGBLogDensity{F,G}
    logposterior::F   # θ -> log p(θ | data)
    gradient::G       # θ -> ∇ log p(θ | data)  (non-mutating)
    dim::Int
end

LogDensityProblems.logdensity(ℓ::TRGBLogDensity, θ) = ℓ.logposterior(θ)
LogDensityProblems.dimension(ℓ::TRGBLogDensity) = ℓ.dim
LogDensityProblems.capabilities(::Type{<:TRGBLogDensity}) = LogDensityProblems.LogDensityOrder{1}()

function LogDensityProblems.logdensity_and_gradient(ℓ::TRGBLogDensity, θ)
    return ℓ.logposterior(θ), ℓ.gradient(θ)
end

# ---------------------------------------------------------------------------
# _sample implementation
# ---------------------------------------------------------------------------

function _sample(backend::DynamicHMCJL, logposterior, x0;
                 rng=Random.default_rng(), reporter=DynamicHMC.NoProgressReport(),
                 kwargs...)
    # Build non-mutating gradient of the log-posterior (not the objective).
    g, _ = build_gradient(backend.ad, logposterior)
    ℓ = TRGBLogDensity(logposterior, g, length(x0))

    result = DynamicHMC.mcmc_with_warmup(
        rng, ℓ, backend.nsamples;
        initialization = (q = float.(x0),),
        warmup_stages  = DynamicHMC.default_warmup_stages(; doubling_stages=backend.n_warmup ÷ 50),
        reporter       = reporter,
        kwargs...,
    )

    samples = result.posterior_matrix  # n_params × n_samples
    logps   = [logposterior(samples[:, i]) for i in axes(samples, 2)]
    acc     = mean(ts.acceptance_rate for ts in result.tree_statistics)

    return TRGBChain(samples, logps, acc, backend, result)
end

end # module
