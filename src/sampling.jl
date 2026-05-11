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

# -----------------------------------------------------------------------
# KissMCMC backend
# -----------------------------------------------------------------------

"""
    KissMCMCJL(; nsamples=10_000, nburnin=1_000, nthin=1, nchain=1)

Backend for posterior sampling using [KissMCMC.jl](https://github.com/mauro3/KissMCMC.jl).
Requires the `KissMCMC` package to be loaded (via the `TRGBDistancesKissMCMCExt` extension).

# Example
```julia
using KissMCMC
chain = sample(BrokenPowerLaw, mags, err, compl, bias, x0;
               prior=prior, backend=KissMCMCJL(nsamples=50_000))
```
"""
struct KissMCMCJL <: AbstractSamplerBackend
    nsamples::Int
    nburnin::Int
    nthin::Int
    nchain::Int
end
KissMCMCJL(; nsamples=10_000, nburnin=1_000, nthin=1, nchain=1) =
    KissMCMCJL(nsamples, nburnin, nthin, nchain)

"""
    DynamicHMCJL(; nsamples=1000, n_warmup=500, ad=ForwardDiffAD())

Backend for posterior sampling using Hamiltonian Monte Carlo via
[DynamicHMC.jl](https://github.com/tpapp/DynamicHMC.jl).
Requires `using DynamicHMC, LogDensityProblems` (loads the
`TRGBDistancesDynamicHMCExt` extension).

Gradient computation is required and is handled by the `ad` backend:
- [`ForwardDiffAD()`](@ref) (default): requires `using ForwardDiff`.
- [`ZygoteAD()`](@ref): requires `using Zygote`.

# Example
```julia
using DynamicHMC, LogDensityProblems, ForwardDiff
chain = sample(BrokenPowerLaw, mags, err, compl, bias, x0;
               prior=prior, backend=DynamicHMCJL(nsamples=1000, ad=ForwardDiffAD()))
```
"""
struct DynamicHMCJL <: AbstractSamplerBackend
    nsamples::Int
    n_warmup::Int
    ad::Union{ADTypes.AbstractADType,Nothing}
end
DynamicHMCJL(; nsamples=1000, n_warmup=500, ad=ADTypes.AutoForwardDiff()) =
    DynamicHMCJL(nsamples, n_warmup, ad)

# -----------------------------------------------------------------------
# sample() — public API
# -----------------------------------------------------------------------

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

A [`TRGBChain`](@ref).
"""
function sample(model_factory, mags, err_func, complete_func, bias_func, x0;
                backend=KissMCMCJL(), prior=nothing, int_width=1.0, kwargs...)
    # Use fixed quadrature when an AD backend is present (adaptive is not differentiable).
    quad = (hasproperty(backend, :ad) && !isnothing(backend.ad)) ? :fixed : :adaptive
    objective = _build_objective(model_factory, mags, err_func, complete_func, bias_func, prior, int_width; quad)
    logposterior = θ -> -objective(θ)
    return _sample(backend, logposterior, x0; kwargs...)
end

function _sample(backend::AbstractSamplerBackend, logposterior, x0; kwargs...)
    error("Backend $(typeof(backend)) requires loading the corresponding package. " *
          "For KissMCMCJL, use `using KissMCMC`.")
end
