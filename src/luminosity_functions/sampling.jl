"""
    AbstractSamplerBackend

Abstract supertype for all sampler backends used with [`sample`](@ref).
"""
abstract type AbstractSamplerBackend end

"""
    TRGBChain

Result returned by [`sample`](@ref).

Fields:
- `samples`: Matrix of samples, size `(n_params, n_samples)`.
- `logprob`: Vector of log-posterior values at each sample.
- `acceptance_fraction`: Fraction of proposals accepted.
- `backend`: The backend used.
- `result`: Raw backend result (backend-specific).
"""
struct TRGBChain{S,L,B,R}
    samples::S
    logprob::L
    acceptance_fraction::Float64
    backend::B
    result::R
end

Base.show(io::IO, c::TRGBChain) =
    print(io, "TRGBChain(n_samples=$(size(c.samples, 2)), acceptance_fraction=$(round(c.acceptance_fraction; digits=3)))")


"""
    sample(model_factory, mags, err_func, complete_func, bias_func, x0;
           backend=KissMCMCJL(), prior=nothing, int_width=1.0, kwargs...)

Sample from the posterior of a [`LuminosityFunction`](@ref) model given observed magnitudes.

# Arguments

- `model_factory`: Callable that accepts a parameter vector and returns a `LuminosityFunction`.
- `mags`: Vector of observed (dereddened) apparent magnitudes.
- `err_func`: Returns photometric error at magnitude `m`.
- `complete_func`: Returns photometric completeness at `m`.
- `bias_func`: Returns photometric bias at `m`.
- `x0`: Initial parameter vector (starting point for the sampler).

# Keyword Arguments

- `backend`: A sampler backend instance (default: `KissMCMCJL()`).
- `prior`: Optional prior — `nothing`, a `Function`, or a `Tuple` of per-parameter distributions.
- `int_width`: Half-width of integration window (default `1.0` mag).

# Returns

A [`TRGBChain`](@ref TRGBDistances.TRGBChain).
"""
function sample(model_factory, mags, err_func, complete_func, bias_func, x0;
                backend=KissMCMCJL(), prior=nothing, int_width=1.0, kwargs...)
    # Use fixed quadrature when an AD backend is present (adaptive is not differentiable).
    quad = (hasproperty(backend, :ad) && !isnothing(backend.ad)) ? :fixed : :adaptive
    objective = _build_objective(model_factory, mags, err_func, complete_func, bias_func, prior, int_width; quad)
    logposterior = θ -> -objective(θ) # Sampling methods typically take log-posterior (negative objective)
    return _sample(backend, logposterior, x0; kwargs...)
end

function _sample(backend::AbstractSamplerBackend, logposterior, x0; kwargs...)
    error("Backend $(typeof(backend)) requires loading the corresponding package. " *
          "For KissMCMCJL, use `using KissMCMC`.")
end

################################################################################
# Backends

"""
    KissMCMCJL(; nsamples=10_000, nburnin=1_000, nwalkers=nothing, nthin=1)

Backend for posterior sampling using [KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl).
Requires the `KissMCMC` package to be loaded (via the `TRGBDistancesKissMCMCExt` extension).

Draws `nsamples` samples from the posterior using `nwalkers` chains, with the first
`nburnin` samples discarded from each chain as burn-in. The chain will be thinned by
keeping only every `nthin`-th sample.  If `nwalkers` is `nothing`, a default value of
`max(2 * n_params, 8)` is used. 

# Example
```julia
using KissMCMC
chain = sample(BrokenPowerLaw, mags, err, compl, bias, x0;
               prior=prior, backend=KissMCMCJL(nsamples=50_000))
```
"""
Base.@kwdef struct KissMCMCJL <: AbstractSamplerBackend
    nsamples::Int = 10_000
    nburnin::Int = 1_000
    nwalkers::Union{Int, Nothing} = nothing
    nthin::Int = 1
end

"""
    DynamicHMCJL(; nsamples=1000, n_warmup=500, ad=ADTypes.AutoForwardDiff())

Backend for posterior sampling using Hamiltonian Monte Carlo via
[DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl).
Requires `using DynamicHMC, LogDensityProblems` (loads the
`TRGBDistancesDynamicHMCExt` extension).

Gradient computation is required and is handled by the `ad` backend:
- `ADTypes.AutoForwardDiff()` (default): requires `using ForwardDiff`.

# Example
```julia
using DynamicHMC, LogDensityProblems, ForwardDiff, ADTypes
chain = sample(BrokenPowerLaw, mags, err, compl, bias, x0;
               prior=prior, backend=DynamicHMCJL(nsamples=1000, ad=ADTypes.AutoForwardDiff()))
```
"""
Base.@kwdef struct DynamicHMCJL <: AbstractSamplerBackend
    nsamples::Int = 1000
    n_warmup::Int = 500
    ad::Union{ADTypes.AbstractADType,Nothing} = ADTypes.AutoForwardDiff()
end

"""
    AdvancedMHJL(; nsamples=10_000, nburnin=1_000, nwalkers=1, 
                   proposal_scale=0.01, nthin=1, parallel=nothing, progress=false)

Backend for posterior sampling using random-walk Metropolis-Hastings via
[AdvancedMH.jl](https://github.com/TuringLang/AdvancedMH.jl).
Requires `using AdvancedMH` (loads the `TRGBDistancesAdvancedMHExt` extension).

Computes `n_samples` samples from the posterior using `nwalkers` chains, with the first
`nburnin` samples discarded from each chain as burn-in. The chain will be thinned by keeping only every `nthin`-th sample.
By default if `nwalkers==1` sampling will be single-threaded,
if `nwalkers>1`, then sampling will be multi-threaded if you started Julia
with multiple threads. Distributed parallelism can be used by passing
`parallel=AbstractMCMC.MCMCDistributed()`
(see [AbstractMCMC.jl](https://turinglang.org/AbstractMCMC.jl/stable/api/#Sampling-multiple-chains-in-parallel)).
If `progress=true`, a progress bar will be shown during sampling.

The proposal is an isotropic Gaussian with standard deviation `proposal_scale`
in each parameter direction.  For well-constrained problems the proposal scale
should be on the order of the marginal posterior standard deviations.

# Example
```julia
using AdvancedMH
chain = sample(BrokenPowerLaw, mags, err, compl, bias, x0;
               prior=prior, backend=AdvancedMHJL(nsamples=10_000, proposal_scale=0.01))
```
"""
Base.@kwdef struct AdvancedMHJL{M} <: AbstractSamplerBackend
    nsamples::Int = 10_000
    nburnin::Int = 1_000
    nwalkers::Int = 1
    proposal_scale::Float64 = 0.01
    nthin::Int = 1
    parallel::M = nothing
    progress::Bool = false
end

"""
    AffineInvariantMCMCJL(; nsamples=10_000, nburnin=1_000, nwalkers=nothing, nthin=1)

Backend for posterior sampling using the affine-invariant ensemble sampler
(Goodman & Weare 2010) via
[AffineInvariantMCMC.jl](https://github.com/madsjulia/AffineInvariantMCMC.jl).
Requires `using AffineInvariantMCMC` (loads the
`TRGBDistancesAffineInvariantMCMCExt` extension).

`nwalkers` defaults to `max(2 * n_params, 8)`. Each walker generates `nsamples` samples.
The first `nburnin` samples from each walker are discarded as burn-in, and the chain is thinned by keeping only every `nthin`-th sample.
The affine-invariant sampler maintains an ensemble of `nwalkers` walkers that explore the posterior
simultaneously; it is invariant to affine reparametrizations and performs well
when parameters are correlated.

# Example
```julia
using AffineInvariantMCMC
chain = sample(BrokenPowerLaw, mags, err, compl, bias, x0;
               prior=prior, backend=AffineInvariantMCMCJL(nsamples=5_000))
```
"""
Base.@kwdef struct AffineInvariantMCMCJL <: AbstractSamplerBackend
    nsamples::Int = 10_000
    nburnin::Int = 1_000
    nwalkers::Union{Int,Nothing} = nothing
    nthin::Int = 1
end
